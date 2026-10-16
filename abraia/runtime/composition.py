"""Composable inference steps for multi-model runtime pipelines."""

from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping, Optional

from abraia.inference.geometry import point_in_polygon, segments_intersect


def count_objects(results):
    """Count object occurrences across detection results."""
    counts = {}
    for result in results:
        label = result["label"]
        counts[label] = counts.get(label, 0) + 1
    return [{"label": label, "count": count} for label, count in counts.items()]


class LineCounter:
    def __init__(self, line):
        self.line = line
        self.in_count = 0
        self.out_count = 0
        self.prev_results = {}

    def update(self, results):
        for result in results:
            track_id = result.get("track_id")
            if track_id is None:
                continue
            prev_result = self.prev_results.get(track_id)
            if prev_result:
                x1, y1, w1, h1 = result["box"]
                x2, y2, w2, h2 = prev_result["box"]
                direction = segments_intersect(
                    self.line[0],
                    self.line[1],
                    [x1 + w1 / 2, y1 + h1 / 2],
                    [x2 + w2 / 2, y2 + h2 / 2],
                )
                if direction > 0:
                    self.in_count += 1
                elif direction < 0:
                    self.out_count += 1
            self.prev_results[track_id] = result
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

    def update(self, results, frame_time):
        in_objects, out_objects = [], []
        for result in results:
            box = result.get("box")
            track_id = result.get("track_id")
            start_time = self.timers.get(track_id)
            if not box or track_id is None:
                continue
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
    points = mapped.get("points")
    if hasattr(points, "copy") and getattr(points, "ndim", 0) == 2:
        points = points.astype(float, copy=True)
        points[:, 0] += x
        points[:, 1] += y
        mapped["points"] = points
    elif isinstance(points, list):
        mapped["points"] = [
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


class ModelStep(CompositionStep):
    """Run a model on the frame or on each input region."""

    def __init__(self, step_id, model, model_kwargs=None):
        super().__init__(step_id)
        _initialize_model_step(self, model, model_kwargs)

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
        context.artifacts[f"{self.step_id}.results"] = output
        self._set_current(context, output, update_results=True)
        return context

    def apply_results(self, context, results):
        """Store precomputed frame results from a streaming model adapter."""
        output = _normalize_results(results, self.step_id)
        context.artifacts[f"{self.step_id}.results"] = output
        if isinstance(output, list):
            self._set_current(context, output, update_results=True)
        else:
            self._set_current(context, output)
        return context

    def close(self):
        _close_model(self.model)


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
        spectral_cube = getattr(context, "spectral_cube", None)
        spectral_input = getattr(self.model, "input_kind", "image") == "spectral"
        if spectral_input and spectral_cube is None:
            raise ValueError(
                f"Bound model step '{self.step_id}' requires a spectral source"
            )
        for index, value in enumerate(selected):
            parent_id = str(value.get("id") or f"{self.step_id}-parent-{index}")
            crop_source = spectral_cube if spectral_input else context.frame
            cropped, source_box = crop_region(
                crop_source,
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
        context.artifacts[f"{self.step_id}.results"] = values
        self._set_current(context, values, update_results=True)
        return context

    def close(self):
        _close_model(self.model)


class FilterStep(CompositionStep):
    """Filter detection-like records before a downstream region step."""

    def __init__(self, step_id, labels=None, min_confidence=None):
        super().__init__(step_id)
        self.labels = {str(label) for label in labels or ()}
        self.min_confidence = min_confidence

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(f"Filter step '{self.step_id}' requires a result list")
        output = _select_results(
            values, self.labels, self.min_confidence
        )
        context.artifacts[f"{self.step_id}.results"] = output
        self._set_current(context, output)
        return context


class CropStep(CompositionStep):
    """Fan out detections into source-aware region inputs."""

    def __init__(self, step_id, padding=0.0):
        super().__init__(step_id)
        self.padding = padding

    def __call__(self, context):
        values = context.current
        if not isinstance(values, list):
            raise ValueError(f"Crop step '{self.step_id}' requires a result list")
        regions = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                continue
            parent_id = str(value.get("id") or f"{self.step_id}-parent-{index}")
            cropped, source_box = crop_region(
                context.frame,
                value.get("box"),
                padding=self.padding,
            )
            regions.append(RegionInput(
                id=f"{self.step_id}-{index}",
                image=cropped,
                source_box=source_box,
                parent_id=parent_id,
                metadata=dict(value),
            ))
        context.artifacts[f"{self.step_id}.items"] = regions
        self._set_current(context, regions)
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
        context.artifacts[f"{self.step_id}.results"] = values
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
    "CropStep",
    "FilterStep",
    "LineCounter",
    "ModelStep",
    "ProcessorStep",
    "RegionInput",
    "RegionTimer",
    "count_objects",
    "crop_region",
    "filter_region",
]
