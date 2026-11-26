"""Composable inference steps for multi-model runtime pipelines."""

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping, Optional

from abraia.inference.geometry import point_in_polygon, segments_intersect

from .anomaly import (
    ANOMALY_MEASUREMENT_REGIONS,
    _consolidate_anomaly_results,
    _inspect_anomaly_map,
    _regions_without_masks,
)
from .results import round_result_value


_MODEL_STEP_FACTORIES = {}
_PIPELINE_STEP_FACTORIES = {}


def register_model_step_factory(kind, factory):
    """Register composition behavior for a model kind.

    The factory receives ``(step_id, model, model_kwargs, *, primary)`` and
    returns a callable step with ``close()`` and a ``model`` attribute. A step
    receives ``FrameContext``, updates it and may return it. It owns its model
    after construction; the runtime closes it on completion, failure or
    cancellation. Construction failures leave model cleanup to the runtime.
    Primary steps wrapping streaming models also implement
    ``apply_results(context, results)`` for precomputed model outputs.
    Register in each worker process; re-registration replaces the factory.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("Model step kind must be a non-empty string")
    if not callable(factory):
        raise TypeError("Model step factory must be callable")
    _MODEL_STEP_FACTORIES[kind] = factory


def create_model_step(step_id, model, spec, model_kwargs, *, primary):
    """Build a registered model step or the built-in composition step."""
    factory = _MODEL_STEP_FACTORIES.get(spec.kind)
    if factory is not None:
        return factory(step_id, model, model_kwargs, primary=primary)
    if primary:
        step_type = PrimaryAnomalyModelStep if spec.kind == "anomaly" else ModelStep
        return step_type(step_id, model, model_kwargs=model_kwargs)
    if spec.kind == "anomaly":
        return BoundAnomalyModelStep(step_id, model, model_kwargs=model_kwargs)
    return BoundModelStep(
        step_id, model, field=spec.result_field, model_kwargs=model_kwargs
    )


def register_pipeline_step_factory(step_type, factory):
    """Register an application-owned non-model pipeline step factory."""
    if not isinstance(step_type, str) or not step_type:
        raise ValueError("Pipeline step type must be a non-empty string")
    if not callable(factory):
        raise TypeError("Pipeline step factory must be callable")
    _PIPELINE_STEP_FACTORIES[step_type] = factory


def create_pipeline_step(step_type, step_id, configuration, *, models=None):
    """Build a registered application-owned pipeline step."""
    factory = _PIPELINE_STEP_FACTORIES.get(step_type)
    if factory is None:
        raise ValueError(f"No factory registered for pipeline step '{step_type}'")
    return factory(step_id, configuration, models=models or {})


def count_objects(results):
    """Count object occurrences across detection results."""
    counts = {}
    for result in results:
        label = result["label"]
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "count": count} for label, count in counts.items()]


class LineCounter:
    def __init__(self, line, minimum_crossing_threshold=2, max_absent_frames=30):
        self.line = line
        self.in_count = 0
        self.out_count = 0
        self.minimum_crossing_threshold = minimum_crossing_threshold
        self.max_absent_frames = max_absent_frames
        self.crossing_state = {}

    def update(self, results):
        current_ids = set()
        direction = lambda point: (
            (self.line[1][0] - self.line[0][0])
            * (point[1] - self.line[0][1])
            - (self.line[1][1] - self.line[0][1])
            * (point[0] - self.line[0][0])
        )
        for result in results:
            track_id = result.get("track_id")
            if track_id is None:
                continue
            current_ids.add(track_id)
            x, y, width, height = result["box"]
            side_value = direction((x + width / 2, y + height / 2))
            if side_value == 0:
                continue
            side = side_value > 0
            state = self.crossing_state.get(track_id)
            if state is None:
                self.crossing_state[track_id] = {
                    "side": side, "candidate": None, "run": 0, "absent": 0,
                    "point": (x + width / 2, y + height / 2),
                }
                continue
            state["absent"] = 0
            point = (x + width / 2, y + height / 2)
            if side == state["side"]:
                state["candidate"] = None
                state["run"] = 0
                state["point"] = point
                continue
            if state["candidate"] != side:
                state["candidate"] = side
                state["run"] = 1
                state["candidate_crosses_line"] = bool(segments_intersect(
                    self.line[0], self.line[1], state["point"], point
                ))
            else:
                state["run"] += 1
            if state["run"] >= self.minimum_crossing_threshold:
                if state["candidate_crosses_line"]:
                    if state["side"]:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                state["side"] = side
                state["candidate"] = None
                state["run"] = 0
            state["point"] = point
        for track_id, state in list(self.crossing_state.items()):
            if track_id not in current_ids:
                state["absent"] += 1
                if state["absent"] >= self.max_absent_frames:
                    del self.crossing_state[track_id]
        return self.in_count, self.out_count


def filter_region(results, polygon):
    """Split detections according to whether their centers lie in a polygon."""
    in_objects, out_objects = [], []
    for result in results:
        box = result.get("box")
        if box:
            x, y, width, height = box
            center = x + width / 2, y + height / 2
            target = in_objects if point_in_polygon(center, polygon) else out_objects
            target.append(result)
    return in_objects, out_objects


class RegionTimer:
    def __init__(self, polygon):
        self.region = polygon
        self.timers = {}
        self.absent_frames = {}
        self.max_absent_frames = 30

    def update(self, results, frame_time):
        in_objects, out_objects = [], []
        current_ids = set()
        for result in results:
            box = result.get("box")
            track_id = result.get("track_id")
            start_time = self.timers.get(track_id)
            if not box or track_id is None:
                continue
            current_ids.add(track_id)
            self.absent_frames[track_id] = 0
            x, y, width, height = box
            center = x + width / 2, y + height / 2
            if point_in_polygon(center, self.region):
                in_objects.append(result)
                if start_time is None:
                    self.timers[track_id] = frame_time
                else:
                    result["time"] = round(frame_time - start_time, 2)
                    result["label"] = f"time {result['time']}s"
                    result.pop("score", None)
            else:
                out_objects.append(result)
                if start_time is not None:
                    del self.timers[track_id]
                    self.absent_frames.pop(track_id, None)
        for track_id in list(self.timers):
            if track_id not in current_ids:
                self.absent_frames[track_id] = self.absent_frames.get(track_id, 0) + 1
                if self.absent_frames[track_id] >= self.max_absent_frames:
                    del self.timers[track_id]
                    del self.absent_frames[track_id]
        return in_objects, out_objects


@dataclass
class RegionInput:
    """An image region that retains its source detection identity."""

    id: str
    image: Any
    source_box: tuple[float, float, float, float]
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _as_box(value):
    """Return a numeric ``x, y, width, height`` box or raise clearly."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Pipeline regions require a four-value 'box'")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("Pipeline region boxes must contain numbers") from error


def crop_region(image, box, padding=0.0):
    """Crop a clipped ``x, y, width, height`` region from an image."""
    source_box = _as_box(box)
    if not hasattr(image, "shape") or len(image.shape) < 2:
        raise ValueError("Pipeline crop inputs must expose an image shape")
    height, width = image.shape[:2]
    x, y, region_width, region_height = source_box
    padding = max(0.0, float(padding))
    x1 = max(0, int(round(x - region_width * padding)))
    y1 = max(0, int(round(y - region_height * padding)))
    x2 = min(width, int(round(x + region_width * (1 + padding))))
    y2 = min(height, int(round(y + region_height * (1 + padding))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Pipeline crop region is empty")
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _result_id(result, prefix, index):
    value = result.get("id") if isinstance(result, Mapping) else None
    return str(value or f"{prefix}-{index}")


def _normalize_results(value, prefix):
    if value is None:
        return []
    if not isinstance(value, list):
        return value
    normalized = []
    for index, result in enumerate(value):
        if isinstance(result, Mapping):
            result = dict(result)
            result.setdefault("id", _result_id(result, prefix, index))
        normalized.append(result)
    return normalized


def _region_result(value):
    """Normalize a model result while preserving OCR-like text results."""
    if isinstance(value, Mapping):
        result = dict(value)
    elif isinstance(value, list):
        result = {"items": value}
    else:
        result = {"value": value}
    if "text" not in result and isinstance(result.get("items"), list):
        text_items = [item for item in result["items"] if isinstance(item, Mapping)]
        texts = [str(item.get("text", "")) for item in text_items if item.get("text")]
        if texts:
            result["text"] = "\n".join(texts)
            scores = [float(item.get("score", 0.0)) for item in text_items if item.get("text")]
            if scores:
                result["text_score"] = max(scores)
    return result


def _select_results(values, labels, min_confidence):
    """Keep mapping results that match the configured label and score."""
    selected = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        if labels and str(value.get("label", "")) not in labels:
            continue
        if min_confidence is not None:
            score = value.get("score", value.get("confidence", 0.0))
            if float(score) < float(min_confidence):
                continue
        selected.append(value)
    return selected


def _run_regions(step_id, model, model_kwargs, regions):
    """Run a model over regions and preserve their source identities."""
    if not regions:
        return []
    run_batch = getattr(model, "run_batch", None)
    if callable(run_batch):
        raw_results = run_batch(
            [region.image for region in regions], **model_kwargs
        )
        if len(raw_results) != len(regions):
            raise ValueError(
                f"Model step '{step_id}' returned {len(raw_results)} results "
                f"for {len(regions)} regions"
            )
    else:
        raw_results = [
            model.run(region.image, **model_kwargs) for region in regions
        ]

    return [
        {
            "item_id": region.id,
            "parent_id": region.parent_id,
            "source_box": list(region.source_box),
            **_region_result(raw),
        }
        for region, raw in zip(regions, raw_results)
    ]


def _close_model(model):
    close = getattr(model, "close", None)
    if callable(close):
        close()


def _initialize_model_step(step, model, model_kwargs):
    step.model = model
    step.model_kwargs = dict(model_kwargs or {})


def _map_region_result(value, source_box):
    """Translate region-local boxes and points back to frame coordinates."""
    if not isinstance(value, Mapping) or not source_box:
        return dict(value) if isinstance(value, Mapping) else value
    x, y = float(source_box[0]), float(source_box[1])
    mapped = dict(value)
    box = mapped.get("box")
    try:
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            mapped["box"] = [
                x + float(box[0]),
                y + float(box[1]),
                float(box[2]),
                float(box[3]),
            ]
    except (TypeError, ValueError, IndexError):
        pass
    for field in ("points", "polygon"):
        points = mapped.get(field)
        if hasattr(points, "copy") and getattr(points, "ndim", 0) == 2:
            points = points.astype(float, copy=True)
            points[:, 0] += x
            points[:, 1] += y
            mapped[field] = points
        elif isinstance(points, (list, tuple)):
            mapped[field] = [
                [x + float(point[0]), y + float(point[1])]
                for point in points
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
    if isinstance(mapped.get("items"), list):
        mapped["items"] = [
            _map_region_result(item, source_box)
            for item in mapped["items"]
        ]
    return mapped


def _record_inference_time(context, started):
    elapsed = (time.perf_counter() - started) * 1000
    context.metrics["inference_ms"] = context.metrics.get("inference_ms", 0) + elapsed


class CompositionStep:
    """Base class for a named step in a composed pipeline."""

    def __init__(self, step_id):
        self.step_id = str(step_id)

    def close(self):
        return None

    def _set_current(self, context, values, *, update_results=False):
        """Store the current ordered output and its optional public results."""
        context.current = values
        if update_results:
            context.results = values


class ObjectCountStep(CompositionStep):
    """Count current objects in a region or unique line crossings."""

    def __init__(self, step_id, mode, *, polygon=None, line=None, tracker=None,
                 max_absent_frames=30):
        super().__init__(step_id)
        self.mode = mode
        self.polygon = polygon or []
        self.line = line or []
        self.tracker = tracker
        self.counter = (
            LineCounter(
                self.line,
                max_absent_frames=max_absent_frames,
            )
            if mode == "line_crossing" else None
        )

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(f"Step '{self.step_id}' requires detection results")
        if self.mode == "region":
            inside, outside = filter_region(values, self.polygon)
            count = len(inside)
            context.views["in_region"] = inside
            context.views["out_region"] = outside
        else:
            if self.tracker is None:
                raise ValueError("Line-crossing counts require an internal tracker")
            self.tracker.update(values)
            in_count, out_count = self.counter.update(values)
            context.metrics["line_counter"] = {"in": in_count, "out": out_count}
            count = in_count + out_count
        context.metrics["object_count"] = count
        context.metrics["count_mode"] = self.mode
        self._set_current(context, values, update_results=True)
        return context


class DecisionStep(CompositionStep):
    """Evaluate a threshold or inclusive range against pipeline outputs."""

    def __init__(self, step_id, metric, operator, threshold, upper_threshold=None):
        super().__init__(step_id)
        self.metric = metric
        self.operator = operator
        self.threshold = float(threshold)
        self.upper_threshold = (
            float(upper_threshold) if upper_threshold is not None else None
        )

    def __call__(self, context):
        if self.metric == "object_count":
            results = ()
            values = [context.metrics.get("object_count")]
        else:
            results = context.results or ()
            values = []
            for result in results:
                value = (
                    result.get(self.metric)
                    if self.metric in {"score", "confidence"}
                    else (result.get("measurements") or {}).get(self.metric)
                )
                values.append(value)
        compare = {
            "gt": lambda value: value > self.threshold,
            "gte": lambda value: value >= self.threshold,
            "lt": lambda value: value < self.threshold,
            "lte": lambda value: value <= self.threshold,
            "eq": lambda value: value == self.threshold,
            "range": lambda value: self.threshold <= value <= self.upper_threshold,
        }[self.operator]
        evaluated = []
        for value in values:
            try:
                evaluated.append(
                    None if value is None else compare(
                        round_result_value(self.metric, value)
                    )
                )
            except (TypeError, ValueError):
                evaluated.append(None)
        if self.metric != "object_count":
            for result, passed in zip(results, evaluated):
                if passed is None:
                    result.setdefault("decision_result", "Not evaluated")
                else:
                    result["decision_result"] = "PASS" if passed else "FAIL"
        booleans = [passed for passed in evaluated if passed is not None]
        decision = {
            "status": (
                "Not evaluated" if not booleans
                else "PASS" if all(booleans)
                else "FAIL"
            ),
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "upper_threshold": self.upper_threshold,
            "values": [
                float(value)
                for value, passed in zip(values, evaluated)
                if passed is not None
            ],
            "pass_count": sum(booleans),
            "fail_count": sum(passed is False for passed in evaluated),
        }
        context.metrics["decision"] = decision
        context.artifacts[f"{self.step_id}.decision"] = decision
        return context


class ModelStep(CompositionStep):
    """Run a model on the frame or on each input region."""

    def __init__(self, step_id, model, model_kwargs=None, *, _raw_results=False):
        super().__init__(step_id)
        _initialize_model_step(self, model, model_kwargs)
        self._raw_results = _raw_results

    def __call__(self, context):
        inputs = context.current
        if inputs is context.frame:
            started = time.perf_counter()
            results = self.model.run(inputs, **self.model_kwargs)
            _record_inference_time(context, started)
            return self.apply_results(
                context,
                results,
            )

        if not isinstance(inputs, Iterable) or isinstance(inputs, (str, bytes)):
            raise ValueError(f"Model step '{self.step_id}' requires region inputs")
        regions = list(inputs)
        started = time.perf_counter()
        output = _run_regions(
            self.step_id, self.model, self.model_kwargs, regions
        )
        _record_inference_time(context, started)
        self._set_current(context, output, update_results=True)
        return context

    def apply_results(self, context, results):
        """Store precomputed frame results from a streaming model adapter."""
        if self._raw_results:
            self._set_current(context, results, update_results=True)
            return context
        output = _normalize_results(results, self.step_id)
        if isinstance(output, list):
            self._set_current(context, output, update_results=True)
        else:
            self._set_current(context, output)
        return context

    def close(self):
        _close_model(self.model)


class PrimaryAnomalyModelStep(ModelStep):
    """Publish image anomaly decisions for both regular and streaming inference."""

    inspection = None
    measure_defects = False

    def apply_results(self, context, results):
        if isinstance(results, list):
            results, regions = _consolidate_anomaly_results(
                results, self.model.last_normalized_anomaly_map,
                context.frame.shape, self.inspection, measure_defects=self.measure_defects,
            )
            if self.measure_defects:
                context.artifacts[ANOMALY_MEASUREMENT_REGIONS] = regions
        return super().apply_results(context, results)


class BoundModelStep(CompositionStep):
    """Run a model on selected detections and attach results automatically."""

    def __init__(
        self,
        step_id,
        model,
        labels=None,
        min_confidence=None,
        crop_padding=0.0,
        field="result",
        model_kwargs=None,
    ):
        super().__init__(step_id)
        _initialize_model_step(self, model, model_kwargs)
        self.labels = {str(label) for label in labels or ()}
        self.min_confidence = min_confidence
        self.crop_padding = crop_padding
        self.field = str(field)

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(
                f"Bound model step '{self.step_id}' requires detection results"
            )
        selected = _select_results(
            values, self.labels, self.min_confidence
        )

        regions = []
        for index, value in enumerate(selected):
            parent_id = str(value.get("id") or f"{self.step_id}-parent-{index}")
            cropped, source_box = crop_region(
                context.frame,
                value.get("box"),
                padding=self.crop_padding,
            )
            regions.append(RegionInput(
                id=f"{self.step_id}-{index}",
                image=cropped,
                source_box=source_box,
                parent_id=parent_id,
                metadata=dict(value),
            ))
        started = time.perf_counter()
        output = _run_regions(
            self.step_id, self.model, self.model_kwargs, regions
        )
        _record_inference_time(context, started)
        by_id = {
            str(value.get("id")): value
            for value in values
            if isinstance(value, Mapping) and value.get("id") is not None
        }
        for value in output:
            target = by_id.get(str(value.get("parent_id")))
            if target is None or not isinstance(target, dict):
                continue
            attached = _map_region_result(value, value.get("source_box"))
            attached.pop("item_id", None)
            attached.pop("parent_id", None)
            attached.pop("source_box", None)
            target[self.field] = attached
        self._set_current(context, values, update_results=True)
        return context

    def close(self):
        _close_model(self.model)


class BoundAnomalyModelStep(BoundModelStep):
    """Inspect each detection crop and map defect regions to the frame."""

    inspection = None
    measure_defects = False

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(
                f"Bound model step '{self.step_id}' requires detection results"
            )
        inspection = dict(self.inspection or {})
        inspection.pop("roi", None)
        measurement_regions = []
        started = time.perf_counter()
        for value in values:
            if not isinstance(value, dict) or value.get("box") is None:
                continue
            crop, source_box = crop_region(context.frame, value["box"])
            predictions = self.model.run(crop, **self.model_kwargs)
            value[self.field] = _map_region_result(
                _region_result(predictions), source_box,
            )
            anomaly_map = self.model.last_normalized_anomaly_map
            score, regions = None, []
            decision = "review"
            if anomaly_map is not None:
                summary = _inspect_anomaly_map(
                    anomaly_map, crop.shape, predictions, inspection,
                    include_masks=self.measure_defects,
                )
                score = summary["score"]
                decision = summary["decision"]
                regions = [
                    _map_region_result(region, source_box)
                    for region in summary["defect_regions"]
                ]
                if self.measure_defects:
                    measurement_regions.extend(regions)
                    regions = _regions_without_masks(regions)
            prediction = {"pass": "normal", "fail": "anomaly"}.get(decision, "review")
            value["decision_result"] = decision.upper()
            value["anomaly"] = {
                "prediction": prediction,
                "decision": decision,
                "score": score,
                "defect_regions": regions,
            }
        if self.measure_defects:
            context.artifacts[ANOMALY_MEASUREMENT_REGIONS] = measurement_regions
        _record_inference_time(context, started)
        self._set_current(context, values, update_results=True)
        return context


class FilterStep(CompositionStep):
    """Filter detection-like records before a downstream region step."""

    def __init__(self, step_id, labels=None, min_confidence=None,
                 region_enabled=False, polygon=None):
        super().__init__(step_id)
        self.labels = {str(label) for label in labels or ()}
        self.min_confidence = min_confidence
        self.region_enabled = region_enabled
        self.polygon = polygon or []

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(f"Filter step '{self.step_id}' requires a result list")
        output = _select_results(
            values, self.labels, self.min_confidence
        )
        if self.region_enabled:
            output, _outside = filter_region(output, self.polygon)
        self._set_current(context, output, update_results=True)
        return context


class ProcessorStep(CompositionStep):
    """Apply a tracker, counter, or region operation to current results."""

    def __init__(self, step_id, processor, operation):
        super().__init__(step_id)
        self.processor = processor
        self.operation = operation

    def _results(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(
                f"Step '{self.step_id}' requires a result-list input"
            )
        return values

    def _emit(self, context, values):
        self._set_current(context, values, update_results=True)
        return context

    def close(self):
        close = getattr(self.processor, "close", None)
        if callable(close):
            close()

    def __call__(self, context):
        values = self._results(context)
        if self.operation == "tracker":
            output = self.processor.update(values)
        elif self.operation == "line_counter":
            in_count, out_count = self.processor.update(values)
            context.metrics["line_counter"] = {
                "in": in_count,
                "out": out_count,
            }
            output = values
        elif self.operation in ("region_filter", "region_timer"):
            if self.operation == "region_timer":
                inside, outside = self.processor.update(
                    values, context.frame_time
                )
                context.metrics["region"] = {
                    "count": len(inside),
                    "in_objects": inside,
                    "out_objects": outside,
                }
            elif self.operation == "region_filter":
                inside, outside = self.processor(values)
            context.views["in_region"] = inside
            context.views["out_region"] = outside
            output = inside
        else:
            raise ValueError(f"Unknown result processor: {self.operation}")
        return self._emit(context, output)


__all__ = [
    "BoundModelStep",
    "CompositionStep",
    "FilterStep",
    "LineCounter",
    "ModelStep",
    "ProcessorStep",
    "RegionInput",
    "RegionTimer",
    "count_objects",
    "crop_region",
    "filter_region",
    "create_pipeline_step",
    "register_model_step_factory",
    "register_pipeline_step_factory",
]
