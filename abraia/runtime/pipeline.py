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
from typing import Any, Callable, Dict, Iterable, List, Optional
from collections.abc import Mapping

from abraia.inference.model_config import ModelSpec

from .composition import (
    CropStep,
    FilterStep,
    LineCounter,
    ProcessorStep,
    RegionTimer,
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
from .sources import create_source, infer_source_type


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


def _prepare_source(source_config, display_config, root: Path):
    """Normalize source configuration into a source and constructor kwargs."""
    source = source_config["src"]
    source_type = source_config.get("type")
    inferred_source_type = infer_source_type(source)
    numeric_source = inferred_source_type == "camera"
    camera_source = (
        source_type in ("camera", "usb_camera")
        or isinstance(source, int)
        or (numeric_source and source_type != "image")
    )
    if camera_source and isinstance(source, str) and numeric_source:
        source = int(source.strip())
    if (
        isinstance(source, str)
        and not camera_source
        and inferred_source_type != "stream"
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
        "source_type": source_type,
    }
    if "video_unpaced" in source_config:
        video_kwargs["video_unpaced"] = source_config["video_unpaced"]
    return source, video_kwargs


def _make_pipeline_renderer(components, render_results=True, render_metrics=True):
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
            from ..utils.draw import render_results as draw_results

            out = draw_results(out, context.results)
        return out

    return render


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
        return source_frame["frame"], source_frame["data"]
    return source_frame, {}


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
    ):
        self.source = source
        self.model = model
        self.steps = list(steps or [])
        self.display = display
        self.render = render
        self.model_kwargs = model_kwargs or {}
        self.on_frame = on_frame
        self.components = components or {}
        self._closed = False
        self._stop_requested = threading.Event()

        frame_rate = getattr(source, "frame_rate", None)
        self.frame_rate = float(frame_rate or 0)

    def stop(self):
        """Request cancellation and release a blocking source if possible."""
        self._stop_requested.set()
        stop = getattr(self.source, "stop", None)
        if callable(stop):
            stop()
        else:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()

    def run(self, is_cancelled=None) -> Optional[FrameContext]:
        """Process all source frames and return the last frame context.

        Models with an ``iter_inference(source)`` method may process frames
        asynchronously.  Results are still consumed in source order so
        stateful steps such as tracking and counting remain deterministic.
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
            return self._run_frames(is_cancelled)
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
                steps = self.steps[1:]
            elif single_model:
                if streaming:
                    results = item.results
                else:
                    inference_started = time.perf_counter()
                    results = self.model.run(frame, **self.model_kwargs)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    context.metrics["inference_ms"] = inference_ms
                context.results = results
                context.current = results
                steps = ()
            else:
                steps = self.steps

            for step in steps:
                context = step(context) or context
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

    def _finish_frame(self, context, elapsed_ms):
        """Report, render, and display one completed frame context."""
        if self.on_frame is not None:
            self.on_frame(context, elapsed_ms)
        if self.display is None:
            return True
        output = self.render(context) if self.render else context.frame
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
    ) -> "Pipeline":
        """Assemble a pipeline from a supported versioned configuration."""
        if not isinstance(config, dict):
            raise ValueError("Pipeline configuration must be a JSON object")
        if config.get("version") != 2:
            raise ValueError("Only version 2 pipeline configurations are supported")
        source_config = config.get("source") or {}
        display_config = config.get("display") or {}
        raw_steps = config.get("steps", []) or []
        if not isinstance(source_config, dict) or "src" not in source_config:
            raise ValueError("Pipeline source must define 'src'")
        if not isinstance(display_config, dict):
            raise ValueError("Pipeline display must be an object")
        if not isinstance(raw_steps, list):
            raise ValueError("Pipeline 'steps' must be an array")
        steps_config = list(iter_pipeline_steps(raw_steps))
        structure_errors = validate_step_structure(steps_config, report_unsupported=False)
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

        model_plans = []
        for step in steps_config:
            if not step.is_active or step.type != "model":
                continue
            model_config = step.get("model")
            model_spec = ModelSpec.from_config(model_config)
            model_spec.require_runtime_valid()
            model_plans.append((model_config, model_spec))

        from ..inference.registry import _create_model_from_spec, get_model_run_kwargs
        from ..inference.session import get_model_accelerator
        from ..inference.tracker import Tracker

        root = Path(base_dir or os.getcwd())
        source, video_kwargs = _prepare_source(source_config, display_config, root)
        video = None
        steps = []
        created_models = []
        components = {}
        try:
            def create_models():
                created = []
                try:
                    for model_config, model_spec in model_plans:
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
                    _close_resources([model for model, _kwargs, _field in created])
                    raise

            # Camera startup and model/session initialization are independent.
            # Starting them together avoids making users wait for both operations
            # serially while keeping each resource owned by this build operation.
            with ThreadPoolExecutor(max_workers=2) as executor:
                source_future = executor.submit(
                    create_source, source, video_kwargs
                )
                models_future = executor.submit(create_models)
                try:
                    video = source_future.result()
                    created_models = models_future.result()
                except Exception:
                    try:
                        video = source_future.result()
                    except Exception:
                        pass
                    try:
                        created_models = models_future.result()
                    except Exception:
                        pass
                    raise

            model_iter = iter(created_models)
            model_index = 0
            for step in steps_config:
                step_id = step.id
                if not step.is_active:
                    continue
                step_type = step.type
                if step_type == "model":
                    model, model_kwargs, model_spec = next(model_iter)
                    steps.append(create_model_step(
                        step_id, model, model_spec, model_kwargs,
                        primary=model_index == 0,
                    ))
                    model_index += 1
                elif step_type == "filter":
                    steps.append(FilterStep(
                        step_id,
                        labels=step.get("labels"),
                        min_confidence=step.get("min_confidence"),
                    ))
                elif step_type == "crop":
                    steps.append(CropStep(
                        step_id,
                        padding=step.get("padding", 0.0),
                    ))
                elif step_type == "tracker":
                    tracker = Tracker(
                        track_thresh=step.get("track_thresh", 0.25),
                        track_buffer=step.get("track_buffer", 30),
                        match_thresh=step.get("match_thresh", 0.8),
                        frame_rate=video.frame_rate,
                    )
                    steps.append(ProcessorStep(step_id, tracker, step_type))
                    components["tracker"] = tracker
                elif step_type == "line_counter":
                    counter = LineCounter(step.get("line"))
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
                    steps.append(ProcessorStep(step_id, region_timer, step_type))
                    components["region_timer"] = region_timer
            primary_model = created_models[0][0] if created_models else None
            video.accelerator = (
                get_model_accelerator(primary_model) if primary_model is not None else "CPU"
            )
            renderer = _make_pipeline_renderer(
                components,
                render_results=display_config.get("render_results", True),
                render_metrics=display_config.get("render_metrics", True),
            )
            show_display = bool(display_config.get("show", True))
            if not show_display:
                video.set_display_enabled(False)
            display = video if show_display or video_kwargs.get("dest") else None
            if renderer_factory is not None:
                renderer = renderer_factory(renderer, config)
            return cls(
                source=video,
                model=primary_model,
                steps=steps,
                display=display,
                render=renderer,
                on_frame=on_frame,
                components=components,
            )
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
            _close_resources([*steps, *unwrapped_models, video])
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
        resources = [self.display, self.source]
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
]
