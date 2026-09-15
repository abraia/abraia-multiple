"""Small synchronous pipelines for frame-by-frame inference workflows."""

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


def _close_resources(resources):
    """Close unique resources without masking the original pipeline error."""
    seen = set()
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.warning("Failed to close pipeline resource", exc_info=True)


def _build_stages(stages_config, source_config, source, video, tracker_cls,
                  line_counter_cls, region_filter_cls, region_timer_cls):
    """Build stages and retain their stateful components for cleanup."""
    stages = []
    components = {}
    for stage_config in stages_config:
        if not isinstance(stage_config, dict):
            raise ValueError("Each pipeline stage must be an object")
        stage_type = stage_config.get("type")
        if stage_config.get("enabled") is False and stage_type in (
            "tracker", "line_counter", "counter", "region_filter", "region", "region_timer"
        ):
            continue

        if stage_type == "tracker":
            enabled = stage_config.get("enabled", True)
            if enabled == "auto":
                enabled = _is_temporal_source(source_config, source)
            if not enabled:
                continue
            tracker = tracker_cls(
                track_thresh=stage_config.get("track_thresh", 0.25),
                track_buffer=stage_config.get("track_buffer", 30),
                match_thresh=stage_config.get("match_thresh", 0.8),
                frame_rate=video.frame_rate,
            )
            stage = TrackerStage(tracker)
            components["tracker"] = tracker
        elif stage_type in ("line_counter", "counter"):
            line = stage_config.get("line")
            if not line or len(line) != 2:
                raise ValueError("line_counter requires a two-point 'line'")
            counter = line_counter_cls(line)
            stage = LineCounterStage(counter)
            components["line_counter"] = counter
        elif stage_type in ("region_filter", "region"):
            polygon = stage_config.get("polygon", stage_config.get("region"))
            if not polygon:
                raise ValueError("region_filter requires a 'polygon'")
            region_filter = region_filter_cls(polygon)
            stage = RegionFilterStage(region_filter)
            components["region_filter"] = region_filter
        elif stage_type == "region_timer":
            polygon = stage_config.get("polygon", stage_config.get("region"))
            if not polygon:
                raise ValueError("region_timer requires a 'polygon'")
            region_timer = region_timer_cls(polygon)
            stage = RegionTimerStage(region_timer)
            components["region_timer"] = region_timer
        else:
            raise ValueError(f"Unknown pipeline stage type: {stage_type}")
        stages.append(stage)
    return stages, components


def _is_remote_source(source: str) -> bool:
    source = source.lower()
    return source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")


def _is_temporal_source(source_config: Dict[str, Any], source: Any) -> bool:
    source_type = source_config.get("type")
    if source_type in ("video", "stream", "camera"):
        return True
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        source_lower = source.lower()
        return _is_remote_source(source) or source_lower.endswith((".mp4", ".avi", ".mov", ".mkv"))
    return False


@dataclass
class FrameContext:
    """State passed between pipeline stages for one frame."""

    frame: Any
    frame_index: int
    frame_time: float
    results: List[dict] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    views: Dict[str, List[dict]] = field(default_factory=dict)


class Pipeline:
    """Run a model and a sequence of processors over a frame source.

    Stages are called in the order supplied. Each stage receives a
    :class:`FrameContext` and may mutate it or return a replacement context.
    The source is expected to be iterable and to expose ``frame_rate`` when
    timestamps based on a video frame rate are useful.
    """

    def __init__(
        self,
        source: Iterable,
        model: Any,
        stages: Optional[Iterable[Callable[[FrameContext], Optional[FrameContext]]]] = None,
        display: Any = None,
        render: Optional[Callable[[FrameContext], Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        components: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.model = model
        self.stages = list(stages or [])
        self.display = display
        self.render = render
        self.model_kwargs = model_kwargs or {}
        self.on_frame = on_frame
        self.components = components or {}
        self._closed = False

        frame_rate = getattr(source, "frame_rate", None)
        self.frame_rate = float(frame_rate or 0)

    def run(self) -> Optional[FrameContext]:
        """Process all source frames and return the last frame context."""
        if self._closed:
            raise RuntimeError("Pipeline has already been closed")
        last_context = None
        try:
            for frame_index, frame in enumerate(self.source):
                started = time.time()
                frame_time = (
                    frame_index / self.frame_rate
                    if self.frame_rate > 0
                    else float(frame_index)
                )
                context = FrameContext(frame, frame_index, frame_time)
                context.results = self.model.run(frame, **self.model_kwargs)

                for stage in self.stages:
                    context = stage(context) or context

                if self.on_frame is not None:
                    elapsed_ms = (time.time() - started) * 1000
                    self.on_frame(context, elapsed_ms)

                output = self.render(context) if self.render else context.frame
                if self.display is not None:
                    self.display.show(output)
                last_context = context

            return last_context
        finally:
            self.close()

    @classmethod
    def from_file(
        cls,
        path: str,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
    ) -> "Pipeline":
        """Build a pipeline from a JSON configuration file."""
        config_path = Path(path)
        with config_path.open("r") as config_file:
            config = json.load(config_file)
        return cls.from_dict(config, base_dir=config_path.parent, on_frame=on_frame)

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
    ) -> "Pipeline":
        """Build a pipeline from the small version-1 JSON schema.

        The loader intentionally creates the existing SDK components rather
        than importing arbitrary classes named by a configuration file.
        """
        if not isinstance(config, dict):
            raise ValueError("Pipeline configuration must be a JSON object")
        if config.get("version", 1) != 1:
            raise ValueError("Unsupported pipeline configuration version")

        source_config = config.get("source") or {}
        model_config = config.get("model") or {}
        display_config = config.get("display") or {}
        stages_config = config.get("stages", []) or []
        if not isinstance(source_config, dict) or "src" not in source_config:
            raise ValueError("Pipeline source must define 'src'")
        if not isinstance(model_config, dict):
            raise ValueError("Pipeline model must be an object")
        model_kind = str(model_config.get("kind", "onnx")).strip().lower()
        if model_kind == "onnx" and not model_config.get("uri"):
            raise ValueError("An ONNX pipeline model must define 'uri'")
        if not isinstance(display_config, dict):
            raise ValueError("Pipeline display must be an object")
        if not isinstance(stages_config, list):
            raise ValueError("Pipeline stages must be an array")

        from ..inference import Tracker
        from ..inference.registry import create_model
        from ..inference.tools import LineCounter, RegionFilter, RegionTimer
        from .video import Video

        root = Path(base_dir or os.getcwd())
        source = source_config["src"]
        source_type = source_config.get("type")
        camera_source = (
            source_type in ("camera", "usb_camera")
            or isinstance(source, int)
            or (isinstance(source, str) and source.strip().isdigit() and source_type != "image")
        )
        if camera_source and isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        if isinstance(source, str) and not camera_source and not _is_remote_source(source):
            source_path = Path(source)
            if not source_path.is_absolute():
                source = str(root / source_path)

        resolution = source_config.get("resolution", (1920, 1080))
        if isinstance(resolution, list):
            resolution = tuple(resolution)

        destination = display_config.get("dest")
        if isinstance(destination, str) and not os.path.isabs(destination):
            destination = str(root / destination)

        model_kwargs = {
            key: model_config[key]
            for key in ("labels", "conf_threshold", "iou_threshold", "approx")
            if key in model_config
        }
        if model_kind not in ("onnx", "object_detection", "instance_segmentation"):
            model_kwargs = {}
        model = create_model(model_config, base_dir=root)
        video = None
        components = {}
        try:
            video = Video(
                source,
                resolution=resolution,
                fps=source_config.get("fps", 30),
                dest=destination,
                source_type=source_type,
            )
            stages, components = _build_stages(
                stages_config,
                source_config,
                source,
                video,
                Tracker,
                LineCounter,
                RegionFilter,
                RegionTimer,
            )
        except Exception:
            _close_resources([model, video, *components.values()])
            raise

        render_results_enabled = display_config.get("render_results", True)
        render_metrics_enabled = display_config.get("render_metrics", True)

        def render(context):
            out = context.frame.copy()
            if render_metrics_enabled:
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
            if render_results_enabled:
                from ..utils.draw import render_results
                out = render_results(out, context.results)
            return out

        show_display = bool(display_config.get("show", True))
        if not show_display:
            # Keep the sink alive when a destination was configured, but do
            # not open an OpenCV preview window in headless runs.
            video._display_enabled = False
        display = video if show_display or destination else None
        return cls(
            source=video,
            model=model,
            stages=stages,
            display=display,
            render=render,
            model_kwargs=model_kwargs,
            on_frame=on_frame,
            components=components,
        )

    def close(self):
        """Close the model and any closeable pipeline components."""
        if self._closed:
            return
        self._closed = True
        _close_resources([
            self.display,
            self.source,
            self.model,
            *self.stages,
            *self.components.values(),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TrackerStage:
    """Attach tracking IDs to model results."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def __call__(self, context: FrameContext) -> FrameContext:
        context.results = self.tracker.update(context.results)
        return context


class LineCounterStage:
    """Update a line counter and expose its values as frame metrics."""

    def __init__(self, counter: Any):
        self.counter = counter

    def __call__(self, context: FrameContext) -> FrameContext:
        in_count, out_count = self.counter.update(context.results)
        context.metrics["line_counter"] = {
            "in": in_count,
            "out": out_count,
        }
        return context


class RegionFilterStage:
    """Split detections into objects inside and outside a region."""

    def __init__(self, region_filter: Any):
        self.region_filter = region_filter

    def __call__(self, context: FrameContext) -> FrameContext:
        in_objects, out_objects = self.region_filter.update(context.results)
        context.views["in_region"] = in_objects
        context.views["out_region"] = out_objects
        context.results = in_objects
        return context


class RegionTimerStage:
    """Track how long detections remain inside a region."""

    def __init__(self, region_timer: Any):
        self.region_timer = region_timer

    def __call__(self, context: FrameContext) -> FrameContext:
        in_objects, out_objects = self.region_timer.update(
            context.results, context.frame_time
        )
        context.views["in_region"] = in_objects
        context.views["out_region"] = out_objects
        context.metrics["region"] = {
            "count": len(in_objects),
            "in_objects": in_objects,
            "out_objects": out_objects,
        }
        context.results = in_objects
        return context


__all__ = [
    "FrameContext",
    "Pipeline",
    "TrackerStage",
    "LineCounterStage",
    "RegionFilterStage",
    "RegionTimerStage",
]
