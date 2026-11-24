"""Pipelines for frame-by-frame inference workflows."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
import json
import logging
import os
from pathlib import Path
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional

from numpy import ndarray

from abraia.inference import get_model_accelerator
from abraia.inference.model_config import ModelSpec

from .anomaly import anomaly_inspection_configuration, consolidate_anomaly_results
from .composition import (
    BoundAnomalyModelStep,
    DecisionStep,
    FilterStep,
    LineCounter,
    ObjectCountStep,
    PrimaryAnomalyModelStep,
    ProcessorStep,
    RegionTimer,
    create_pipeline_step,
    filter_region,
    create_model_step,
)
from .config import (
    SUPPORTED_STEP_TYPES,
    iter_pipeline_steps,
    is_valid_step_geometry,
    steps_missing_tracker,
    validate_step_structure,
)
from .results import result_records
from .sources import create_frame_source, infer_source_type, normalize_source_type


logger = logging.getLogger(__name__)


def _close_resources(resources: Iterable) -> None:
    """Close unique resources without masking the active pipeline error."""
    seen = set()
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            logger.warning("Failed to close runtime resource", exc_info=True)


def prepare_source(
    source_config, display_config, root: Path, input_source=None
):
    """Normalize source configuration into a source and constructor kwargs."""
    source = (
        source_config["src"] if input_source is None else input_source
    )
    configured_type = source_config.get("type")
    source_type = (
        normalize_source_type(configured_type) if configured_type else None
    )
    inferred_source_type = (
        infer_source_type(source) if input_source is None else None
    )
    numeric_source = inferred_source_type == "camera"
    camera_source = (
        source_type in ("camera", "usb_camera")
        or source_type == "genicam"
        or isinstance(source, int)
        or (numeric_source and source_type != "image")
    )
    if (
        camera_source
        and source_type != "genicam"
        and isinstance(source, str)
        and numeric_source
    ):
        source = int(source.strip())
    if (
        isinstance(source, str)
        and not camera_source
        and inferred_source_type != "stream"
        and source_type != "genicam"
    ):
        source_path = Path(source)
        if not source_path.is_absolute():
            source = str(root / source_path)

    resolution = source_config.get("resolution", (1920, 1080))
    if isinstance(resolution, list):
        resolution = tuple(resolution)

    destination = display_config.get("dest")
    if isinstance(destination, str) and not os.path.isabs(destination):
        destination = str(root / destination)

    video_kwargs = {
        "resolution": resolution,
        "fps": source_config.get("fps", 30),
        "dest": destination,
        "source_type": configured_type or source_type,
    }
    source_options = source_config.get("options", {})
    if not isinstance(source_options, dict):
        raise ValueError("Pipeline source 'options' must be an object")
    source_options = dict(source_options)
    if source_options:
        video_kwargs["source_options"] = source_options
    if "video_unpaced" in source_config:
        video_kwargs["video_unpaced"] = source_config["video_unpaced"]
    return source, video_kwargs


def _make_pipeline_renderer(
    components, render_results=True, render_metrics=True, render_anomaly=True,
):
    """Create a frame renderer from pipeline components and display options."""
    def render(context):
        out = context.frame.copy()
        if render_metrics:
            counter = components.get("line_counter")
            if counter:
                from ..utils.draw import render_counter

                out = render_counter(
                    out,
                    counter.line,
                    f"In: {counter.in_count} | Out: {counter.out_count}",
                )
            count_stage = components.get("count")
            if count_stage and count_stage.mode == "region":
                from ..utils.draw import render_region

                out = render_region(
                    out, count_stage.polygon,
                    f"Count: {context.metrics.get('object_count', 0)}",
                    color=(255, 255, 0),
                )
            elif count_stage and count_stage.mode == "line_crossing":
                from ..utils.draw import render_counter

                line_counts = context.metrics.get("line_counter", {})
                out = render_counter(
                    out, count_stage.line,
                    f"In: {line_counts.get('in', 0)} | Out: {line_counts.get('out', 0)}",
                )
            time_region = context.metrics.get("time_in_region")
            if time_region:
                from ..utils.draw import render_region

                out = render_region(
                    out, time_region["polygon"],
                    f"Count: {time_region['count']}", color=(255, 255, 0),
                )
            region_timer = components.get("region_timer")
            if region_timer:
                from ..utils.draw import render_region

                out = render_region(
                    out,
                    region_timer.region,
                    f"Count: {context.metrics['region']['count']}",
                    color=(255, 255, 0),
                )
        if render_results:
            from .rendering import render_pipeline_results

            decision = context.metrics.get("decision", {})
            frame_decision = (
                decision.get("status") if decision.get("metric") == "object_count" else None
            )
            out = render_pipeline_results(
                out, context.results, render_anomaly=render_anomaly,
                frame_decision=frame_decision,
            )
        return out

    return render


def _make_depth_renderer(renderer):
    """Replace the RGB frame with a colorized depth map when available."""
    def render(context):
        import cv2
        import numpy as np

        result = next((
            value for value in context.results or ()
            if value.get("depth_units") in ("m", "relative")
            and value.get("depth_map") is not None
        ), None)
        if result is None:
            return renderer(context)
        depth = np.asarray(result["depth_map"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        values = depth[valid]
        near, far = 0.0, 10.0
        if result["depth_units"] == "relative" and values.size:
            near, far = float(values.min()), float(values.max())
        intensity = np.zeros(depth.shape, dtype=np.uint8)
        if near == far:
            intensity[valid] = 128
        else:
            fraction = (np.clip(values, near, far) - near) / (far - near)
            intensity[valid] = np.rint((1 - fraction) * 255)
        colored = cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)
        colored[~valid] = 0
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return render


def _validate_pipeline_config(config, input_source):
    """Validate the document and return normalized runtime configuration parts."""
    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    if config.get("version") != 2:
        raise ValueError("Only version 2 pipeline configurations are supported")

    source_config = config.get("source") or {}
    display_config = config.get("display") or {}
    output_config = config.get("output")
    raw_steps = config.get("steps", []) or []
    if not isinstance(source_config, dict) or (
        "src" not in source_config and input_source is None
    ):
        raise ValueError("Pipeline source must define 'src'")
    if not isinstance(display_config, dict):
        raise ValueError("Pipeline display must be an object")
    if output_config is not None and not isinstance(output_config, dict):
        raise ValueError("Pipeline output must be an object")
    if output_config is not None and output_config.get("type") != "rtsp":
        raise ValueError("Pipeline output type must be 'rtsp'")
    if output_config is not None and not output_config.get("url"):
        raise ValueError("RTSP output must define 'url'")
    if not isinstance(raw_steps, list):
        raise ValueError("Pipeline 'steps' must be an array")

    steps_config = list(iter_pipeline_steps(raw_steps))
    structure_errors = validate_step_structure(
        steps_config, report_unsupported=False
    )
    if structure_errors:
        raise ValueError(structure_errors[0])
    dependency_errors = steps_missing_tracker(steps_config)
    if dependency_errors:
        index, step_type = dependency_errors[0]
        raise ValueError(
            f"Step {index}: {step_type} requires an earlier tracker."
        )
    for step in steps_config:
        if not step.is_active:
            continue
        if step.type not in SUPPORTED_STEP_TYPES:
            raise ValueError(f"Unknown composed pipeline step type: {step.type}")
        if not is_valid_step_geometry(step.type, step):
            if step.type == "line_counter":
                raise ValueError("line_counter requires a two-point 'line'")
            raise ValueError(f"{step.type} requires a 'polygon'")
    return source_config, display_config, output_config, steps_config


def _model_plans(steps_config):
    """Validate active model steps and return their construction plans."""
    plans = []
    for step in steps_config:
        if not step.is_active or step.type != "model":
            continue
        model_config = step.get("model")
        model_spec = ModelSpec.from_config(model_config)
        model_spec.require_runtime_valid()
        plans.append((model_config, model_spec))
    return plans


def _create_pipeline_models(model_plans, root, accelerator):
    """Construct planned models and clean up partial construction failures."""
    from ..inference.registry import (
        _create_model_from_spec, get_model_run_kwargs, resolve_model_backend,
    )

    selected = [
        (config, resolve_model_backend(spec, base_dir=root, accelerator=accelerator))
        for config, spec in model_plans
    ]
    created = []
    try:
        for model_config, model_spec in selected:
            model_kwargs = get_model_run_kwargs(model_config, model_spec)
            model = _create_model_from_spec(
                model_spec,
                model_config,
                base_dir=root,
                accelerator=accelerator,
            )
            created.append((model, model_kwargs, model_spec))
        return created
    except Exception:
        _close_resources([model for model, _kwargs, _spec in created])
        raise


def _build_pipeline_steps(
    steps_config, created_models, source, video_kwargs, calibration=None
):
    """Build runtime steps and shared rendering components."""
    from ..inference.tracker import Tracker

    steps = []
    components = {}
    model_iter = iter(created_models)
    model_index = 0
    model_step = None
    models = {}
    for step in steps_config:
        if not step.is_active:
            continue
        step_id = step.id
        step_type = step.type
        if step_type == "model":
            model, model_kwargs, model_spec = next(model_iter)
            models[model_spec.kind] = {
                "model": model,
                "configuration": step.get("model") or {},
            }
            model_step = create_model_step(
                step_id, model, model_spec, model_kwargs,
                primary=model_index == 0,
            )
            if isinstance(model_step, (PrimaryAnomalyModelStep, BoundAnomalyModelStep)):
                model_step.inspection = (step.get("model") or {}).get("inspection")
            steps.append(model_step)
            model_index += 1
        elif step_type == "filter":
            steps.append(FilterStep(
                step_id,
                labels=step.get("labels"),
                min_confidence=step.get("min_confidence"),
                region_enabled=step.get("region_enabled", False),
                polygon=step.get("polygon"),
            ))
        elif step_type == "measure":
            measure_config = dict(step.raw)
            if calibration:
                measure_config["calibration"] = calibration
            tracker = components.get("tracker")
            if tracker and measure_config.get("time_in_region"):
                measure_config["tracker_max_absent_frames"] = (
                    _tracker_max_absent_frames(tracker)
                )
            from .features import FeatureExtractionStep

            if isinstance(model_step, (PrimaryAnomalyModelStep, BoundAnomalyModelStep)):
                model_step.measure_defects = True
            steps.append(FeatureExtractionStep(step_id, measure_config, models=models))
        elif step_type == "count":
            mode = step.get("mode", "region")
            tracker = None
            if mode == "line_crossing":
                tracker = Tracker(
                    track_thresh=step.get("track_thresh", 0.25),
                    track_buffer=step.get("track_buffer", 30),
                    match_thresh=step.get("match_thresh", 0.8),
                    frame_rate=(
                        getattr(source, "frame_rate", 0)
                        or video_kwargs.get("fps", 30)
                    ),
                )
            count_stage = ObjectCountStep(
                step_id, mode, polygon=step.get("polygon"),
                line=step.get("line"), tracker=tracker,
                max_absent_frames=_tracker_max_absent_frames(tracker),
            )
            steps.append(count_stage)
            components["count"] = count_stage
        elif step_type == "decision":
            steps.append(DecisionStep(
                step_id,
                step.get("metric", "object_count"),
                step.get("operator", "gte"),
                step.get("lower_threshold", step.get("threshold", 1)),
                step.get("upper_threshold"),
            ))
        elif step_type == "tracker":
            tracker = Tracker(
                track_thresh=step.get("track_thresh", 0.25),
                track_buffer=step.get("track_buffer", 30),
                match_thresh=step.get("match_thresh", 0.8),
                frame_rate=(
                    getattr(source, "frame_rate", 0)
                    or video_kwargs.get("fps", 30)
                ),
            )
            steps.append(ProcessorStep(step_id, tracker, step_type))
            components["tracker"] = tracker
        elif step_type == "line_counter":
            counter = LineCounter(
                step.get("line"),
                max_absent_frames=_tracker_max_absent_frames(
                    components.get("tracker")
                ),
            )
            steps.append(ProcessorStep(step_id, counter, step_type))
            components["line_counter"] = counter
        elif step_type == "region_filter":
            region_filter = partial(
                filter_region, polygon=step.get("polygon")
            )
            steps.append(ProcessorStep(step_id, region_filter, step_type))
            components["region_filter"] = region_filter
        elif step_type == "region_timer":
            region_timer = RegionTimer(step.get("polygon"))
            if "tracker" in components:
                region_timer.max_absent_frames = (
                    _tracker_max_absent_frames(components["tracker"])
                )
            steps.append(ProcessorStep(step_id, region_timer, step_type))
            components["region_timer"] = region_timer
    return steps, components


def _tracker_max_absent_frames(tracker):
    """Return the tracker's configured expiry window when available."""
    implementation = getattr(tracker, "tracker", None)
    return getattr(implementation, "max_time_lost", 30)


@dataclass
class FrameContext:
    """State passed between pipeline steps for one frame.

    ``frame`` is the display/model image; ``data`` is opaque source extension
    data whose interpretation belongs to the extension. Steps read/write
    ``current``, publish detections through ``results`` and may store named
    intermediate values in ``artifacts`` or timings in ``metrics``.
    """

    frame: Any
    frame_index: int
    frame_time: float
    data: Dict[str, Any] = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)
    current: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    views: Dict[str, List[dict]] = field(default_factory=dict)

    def __post_init__(self):
        if self.current is None:
            self.current = self.results


def _source_frame_parts(source_frame):
    """Unpack an image and opaque extension data from a source item."""
    if isinstance(source_frame, Mapping):
        return source_frame["frame"], source_frame.get("data") or {}
    return source_frame, {}


def _result_key(value, index):
    if not isinstance(value, Mapping):
        return ("index", index)
    identity = value.get("parent_id") or value.get("id")
    if identity is not None:
        return ("id", str(identity))
    box = value.get("box")
    if isinstance(box, (list, tuple)):
        return ("box", tuple(box), value.get("label"))
    return ("index", index)


def _merge_pipeline_results(previous, current, published, step):
    previous = previous if isinstance(previous, list) else []
    values = current if isinstance(current, list) else published
    if not isinstance(values, list):
        values = previous

    previous_by_key = {
        _result_key(value, index): value
        for index, value in enumerate(previous)
        if isinstance(value, Mapping)
    }
    field = getattr(step, "field", None) or getattr(step, "result_field", None)
    if field and any(
        isinstance(value, Mapping) and value.get("parent_id") is not None
        for value in values
    ):
        rows = [
            dict(value) if isinstance(value, Mapping) else value
            for value in previous
        ]
        positions = {
            _result_key(value, index): index
            for index, value in enumerate(rows)
            if isinstance(value, Mapping)
        }
        for value in values:
            if not isinstance(value, Mapping):
                continue
            parent_index = positions.get(("id", str(value["parent_id"])))
            if parent_index is None:
                continue
            rows[parent_index][field] = {
                name: item for name, item in value.items()
                if name not in {"id", "parent_id", "item_id", "source_box"}
            }
        return rows

    rows = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            rows.append(value)
            continue
        result = dict(value)
        previous_value = previous_by_key.get(_result_key(value, index))
        if previous_value is not None:
            for name, item in previous_value.items():
                if name not in result:
                    result[name] = item
                elif (
                    isinstance(result[name], Mapping)
                    and isinstance(item, Mapping)
                ):
                    result[name] = {**item, **result[name]}
        rows.append(result)
    return rows


def _update_pipeline_results(context, previous, step):
    if isinstance(step, DecisionStep):
        # Decisions update published results. Keep matching step inputs in sync
        # so a later merge cannot restore their old decisions.
        if isinstance(context.current, list):
            decisions = {
                _result_key(record, index): record["decision_result"]
                for index, record in enumerate(context.results)
                if isinstance(record, Mapping) and "decision_result" in record
            }
            for index, record in enumerate(context.current):
                if isinstance(record, dict):
                    key = _result_key(record, index)
                    if key in decisions:
                        record["decision_result"] = decisions[key]
        return
    context.results = _merge_pipeline_results(
        previous, context.current, context.results, step
    )


class Pipeline:
    """Run a model and a sequence of processors over a frame source.

    Steps are called in the order supplied. Each step receives a
    :class:`FrameContext` and may mutate it or return a replacement context.
    The source is expected to be iterable and to expose ``frame_rate`` when
    timestamps based on a video frame rate are useful.
    """

    def __init__(
        self,
        source: Iterable,
        model: Any,
        steps: Optional[Iterable[Callable[[FrameContext], Optional[FrameContext]]]] = None,
        display: Any = None,
        render: Optional[Callable[[FrameContext], Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        components: Optional[Dict[str, Any]] = None,
        output: Any = None,
        result_calibration: Optional[Dict[str, Any]] = None,
        *,
        render_metrics: bool = True,
    ):
        self.source = source
        self.model = model
        self.steps = list(steps or [])
        self._stream_steps = tuple(self.steps[1:])
        self.display = display
        self.render = render
        self.model_kwargs = model_kwargs or {}
        self.on_frame = on_frame
        self.components = components or {}
        self.output = output
        self.result_calibration = result_calibration or {}
        self._closed = False
        self._stop_requested = threading.Event()
        self._render_output_owned = False
        self._render_stats = render_metrics and model is not None
        self._accelerator_label = (
            get_model_accelerator(model) if self._render_stats else None
        )
        self._last_frame_at = None
        self._frame_fps = None

        frame_rate = getattr(source, "frame_rate", None)
        self.frame_rate = float(frame_rate or 0)

    def stop(self):
        """Request cancellation and release a blocking source if possible."""
        self._stop_requested.set()
        stop_output = getattr(self.output, "stop", None)
        if callable(stop_output):
            stop_output()
        stop = getattr(self.source, "stop", None)
        if callable(stop):
            stop()
        else:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()

    def run(self, is_cancelled=None, *, hold_image=False) -> Optional[FrameContext]:
        """Process all source frames and return the last frame context.

        Models with an ``iter_inference(source)`` method may process frames
        asynchronously.  Results are still consumed in source order so
        stateful steps such as tracking and counting remain deterministic.
        Set ``hold_image`` to keep a single image preview open after inference.
        """
        if self._closed:
            raise RuntimeError("Pipeline has already been closed")
        watcher = None
        if is_cancelled is None:
            is_cancelled = lambda: False
        else:
            self._stop_requested.clear()

            def watch_cancellation():
                while not self._stop_requested.wait(0.05):
                    if is_cancelled():
                        self.stop()
                        return

            watcher = threading.Thread(target=watch_cancellation, daemon=True)
            watcher.start()
        try:
            last_context = self._run_frames(is_cancelled)
            if hold_image and last_context is not None:
                wait_for_preview = getattr(self.display, "wait_for_image_preview", None)
                if callable(wait_for_preview):
                    wait_for_preview(
                        self._stop_requested.is_set, source=self.source
                    )
            return last_context
        finally:
            self._stop_requested.set()
            if watcher is not None:
                watcher.join(timeout=0.2)
            self.close()

    def _run_frames(self, is_cancelled):
        """Run inference and ordered steps over each source frame."""
        last_context = None
        first_step = self.steps[0] if self.steps else None
        inference_model = (
            getattr(first_step, "model", None)
            if first_step is not None
            else self.model
        )
        stream_inference = getattr(inference_model, "iter_inference", None)
        streaming = callable(stream_inference)
        if streaming:
            from .inference import FrameResult

            model_kwargs = (
                first_step.model_kwargs
                if first_step is not None
                else self.model_kwargs
            )
            frame_results = stream_inference(self.source, **model_kwargs)
            frames = enumerate(frame_results)
        else:
            frames = enumerate(self.source)

        single_model = not self.steps and self.model is not None
        single_anomaly_model = False
        if single_model:
            from ..inference.models.anomaly import AnomalyModel

            single_anomaly_model = isinstance(self.model, AnomalyModel)
        for source_index, item in frames:
            if self._stop_requested.is_set() or is_cancelled():
                self.stop()
                break
            if streaming:
                if not isinstance(item, FrameResult):
                    raise TypeError("Async models must yield FrameResult instances")
                frame_index = item.record.index
                source_frame = item.record.frame
                inference_ms = item.inference_ms
                stream_elapsed_ms = item.elapsed_ms
            else:
                frame_index = source_index
                source_frame = item
                inference_ms = None
                stream_elapsed_ms = None

            frame, data = _source_frame_parts(
                source_frame
            )
            frame_time = (
                frame_index / self.frame_rate
                if self.frame_rate > 0
                else float(frame_index)
            )
            context = FrameContext(
                frame,
                frame_index,
                frame_time,
                data=data,
                current=frame,
            )
            if inference_ms is not None:
                context.metrics["inference_ms"] = inference_ms

            started = time.perf_counter()
            if streaming and first_step is not None:
                first_step.apply_results(context, item.results)
                _update_pipeline_results(context, [], first_step)
                steps = self._stream_steps
            elif single_model:
                if streaming:
                    results = item.results
                else:
                    inference_started = time.perf_counter()
                    results = self.model.run(frame, **self.model_kwargs)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    context.metrics["inference_ms"] = inference_ms
                if single_anomaly_model and isinstance(results, list):
                    results = consolidate_anomaly_results(
                        results, self.model.last_normalized_anomaly_map, frame.shape,
                    )
                context.results = results
                context.current = results
                steps = ()
            else:
                steps = self.steps

            for step in steps:
                previous_results = context.results
                context = step(context) or context
                _update_pipeline_results(context, previous_results, step)
            step_elapsed_ms = (time.perf_counter() - started) * 1000
            if single_model:
                elapsed_ms = stream_elapsed_ms if streaming else inference_ms
            elif streaming:
                elapsed_ms = (stream_elapsed_ms or 0) + step_elapsed_ms
            else:
                elapsed_ms = step_elapsed_ms
            if self.on_frame is not None and elapsed_ms is None:
                elapsed_ms = 0
            last_context = context
            if not self._finish_frame(context, elapsed_ms):
                break
        return last_context

    def _render_frame(self, context):
        """Render a frame for either a local display or a preview callback."""
        frame = self.render(context) if self.render else context.frame
        if self._render_stats and isinstance(frame, ndarray) and frame.ndim in (2, 3):
            from ..utils.draw import render_resolution, render_status

            if not self._render_output_owned or frame is context.frame:
                frame = frame.copy()
            render_status(frame, fps=self._frame_fps, accelerator=self._accelerator_label)
            render_resolution(frame)
        return frame

    def _finish_frame(self, context, elapsed_ms):
        """Report, render, and display one completed frame context."""
        if self._render_stats:
            now = time.perf_counter()
            if self._last_frame_at is not None and now > self._last_frame_at:
                self._frame_fps = 1 / (now - self._last_frame_at)
            self._last_frame_at = now
        context.results = result_records(
            context.results, calibration=self.result_calibration,
        )
        if self.on_frame is not None:
            self.on_frame(context, elapsed_ms)
        if self.output is not None:
            self.output.write(context.frame)
        if self.display is None:
            return True
        output = self._render_frame(context)
        show_rendered = getattr(self.display, "show_rendered", None)
        if callable(show_rendered):
            display_result = show_rendered(output)
        else:
            display_result = self.display.show(output)
        if display_result is False or getattr(self.display, "quit", False):
            self.stop()
            return False
        return True

    @classmethod
    def from_file(
        cls,
        path: str,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
        renderer_factory: Optional[Callable] = None,
    ) -> "Pipeline":
        """Build a pipeline from a JSON configuration file."""
        config_path = Path(path)
        with config_path.open("r") as config_file:
            config = json.load(config_file)
        return cls.from_dict(
            config,
            base_dir=config_path.parent,
            on_frame=on_frame,
            accelerator=accelerator,
            renderer_factory=renderer_factory,
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
        renderer_factory: Optional[Callable] = None,
        input_source: Optional[Iterable] = None,
    ) -> "Pipeline":
        """Assemble a pipeline from a supported versioned configuration."""
        source_config, display_config, output_config, steps_config = (
            _validate_pipeline_config(config, input_source)
        )
        model_plans = _model_plans(steps_config)

        root = Path(base_dir or os.getcwd())
        source, video_kwargs = prepare_source(
            source_config, display_config, root, input_source=input_source
        )
        frame_source = None
        display = None
        output = None
        steps = []
        created_models = []
        components = {}
        try:
            # Camera startup and model/session initialization are independent.
            # Starting them together avoids making users wait for both operations
            # serially while keeping each resource owned by this build operation.
            with ThreadPoolExecutor(max_workers=2) as executor:
                source_future = (
                    executor.submit(create_frame_source, source, video_kwargs)
                    if input_source is None else None
                )
                models_future = executor.submit(
                    _create_pipeline_models, model_plans, root, accelerator
                )
                try:
                    frame_source = (
                        source_future.result()
                        if source_future is not None
                        else input_source
                    )
                    created_models = models_future.result()
                except Exception:
                    if source_future is not None:
                        try:
                            frame_source = source_future.result()
                        except Exception:
                            pass
                    try:
                        created_models = models_future.result()
                    except Exception:
                        pass
                    raise

            steps, components = _build_pipeline_steps(
                steps_config, created_models, frame_source, video_kwargs,
                calibration=(config.get("source") or {}).get("calibration"),
            )
            primary_model = created_models[0][0] if created_models else None
            render_metrics = bool(display_config.get("render_metrics", True))
            renderer = _make_pipeline_renderer(
                components,
                render_results=display_config.get("render_results", True),
                render_metrics=render_metrics,
                render_anomaly=renderer_factory is None,
            )
            if model_plans and model_plans[0][1].task == "depth":
                renderer = _make_depth_renderer(renderer)
            if model_plans and model_plans[0][1].kind == "anomaly":
                from .rendering import AnomalyPipelineRenderer

                roi = (
                    anomaly_inspection_configuration(config)["roi"]
                    if renderer_factory is None else ()
                )
                renderer = AnomalyPipelineRenderer(renderer, primary_model, roi=roi)
            if output_config is not None:
                from ..media.outputs import RTSPOutput

                resolution = getattr(frame_source, "resolution", None) or video_kwargs.get(
                    "resolution", (1920, 1080)
                )
                output_size = (
                    int(getattr(frame_source, "width", 0) or resolution[0]),
                    int(getattr(frame_source, "height", 0) or resolution[1]),
                )
                output = RTSPOutput(
                    url=output_config["url"],
                    fps=getattr(frame_source, "frame_rate", None)
                    or video_kwargs.get("fps", 30),
                    size=output_size,
                    bitrate=output_config.get("bitrate", 4_000_000),
                )
            show_display = bool(display_config.get("show", True))
            if show_display or video_kwargs.get("dest"):
                from .video import VideoOutput

                resolution = getattr(frame_source, "resolution", None) or video_kwargs.get(
                    "resolution", (1920, 1080)
                )
                display = VideoOutput(
                    dest=video_kwargs.get("dest"),
                    fps=getattr(frame_source, "frame_rate", None)
                    or video_kwargs.get("fps", 30),
                    size=(
                        int(getattr(frame_source, "width", 0) or resolution[0]),
                        int(getattr(frame_source, "height", 0) or resolution[1]),
                    ),
                )
                if not show_display:
                    display.set_display_enabled(False)
            if renderer_factory is not None:
                renderer = renderer_factory(renderer, config)
            pipeline = cls(
                source=frame_source,
                model=primary_model,
                steps=steps,
                display=display,
                render=renderer,
                on_frame=on_frame,
                components=components,
                output=output,
                result_calibration=(source_config.get("calibration") or {}),
                render_metrics=render_metrics,
            )
            pipeline._render_output_owned = renderer_factory is None
            return pipeline
        except Exception:
            wrapped_model_ids = {
                id(step.model)
                for step in steps
                if getattr(step, "model", None) is not None
            }
            unwrapped_models = [
                model for model, _kwargs, _field in created_models
                if id(model) not in wrapped_model_ids
            ]
            _close_resources([*steps, *unwrapped_models, display, output, frame_source])
            raise

    def close(self):
        """Close the model and any closeable pipeline components."""
        if self._closed:
            return
        self._closed = True
        step_models = {
            id(step.model)
            for step in self.steps
            if getattr(step, "model", None) is not None
            and callable(getattr(step, "close", None))
        }
        resources = [self.output, self.display, self.source]
        if self.model is not None and id(self.model) not in step_models:
            resources.append(self.model)
        _close_resources([
            *resources,
            *self.steps,
            *self.components.values(),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "FrameContext",
    "Pipeline",
    "prepare_source",
]
