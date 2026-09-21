"""Factories used to assemble runtime inference pipelines."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .lifecycle import close_resources


def _is_remote_source(source: str) -> bool:
    source = source.lower()
    return source.startswith(("http://", "https://", "rtsp://"))


def _is_temporal_source(source_config: Dict[str, Any], source: Any) -> bool:
    source_type = source_config.get("type")
    if source_type in ("video", "stream", "camera"):
        return True
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        source_lower = source.lower()
        return _is_remote_source(source) or source_lower.endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    return False


@dataclass(frozen=True)
class SourcePlan:
    """Normalized source and video constructor arguments."""

    source: Any
    source_type: Optional[str]
    video_kwargs: Dict[str, Any]


class SourceFactory:
    """Normalize pipeline source configuration and create video sources."""

    @staticmethod
    def prepare(source_config, display_config, root: Path) -> SourcePlan:
        source = source_config["src"]
        source_type = source_config.get("type")
        camera_source = (
            source_type in ("camera", "usb_camera")
            or isinstance(source, int)
            or (
                isinstance(source, str)
                and source.strip().isdigit()
                and source_type != "image"
            )
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

        video_kwargs = {
            "resolution": resolution,
            "fps": source_config.get("fps", 30),
            "dest": destination,
            "source_type": source_type,
        }
        if "video_unpaced" in source_config:
            video_kwargs["video_unpaced"] = source_config["video_unpaced"]
        return SourcePlan(source, source_type, video_kwargs)

    @staticmethod
    def create(plan: SourcePlan):
        from .video import Video

        return Video(plan.source, **plan.video_kwargs)


class TrackerStage:
    """Attach tracking IDs to model results."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def __call__(self, context):
        context.results = self.tracker.update(context.results)
        return context


class LineCounterStage:
    """Update a line counter and expose its values as frame metrics."""

    def __init__(self, counter: Any):
        self.counter = counter

    def __call__(self, context):
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

    def __call__(self, context):
        inside, outside = self.region_filter.update(context.results)
        context.views["in_region"] = inside
        context.views["out_region"] = outside
        context.results = inside
        return context


class RegionTimerStage:
    """Track time spent by detections inside a configured region."""

    def __init__(self, region_timer: Any):
        self.region_timer = region_timer

    def __call__(self, context):
        inside, outside = self.region_timer.update(
            context.results,
            context.frame_time,
        )
        context.views["in_region"] = inside
        context.views["out_region"] = outside
        context.metrics["region"] = {
            "count": len(inside),
            "in_objects": inside,
            "out_objects": outside,
        }
        context.results = inside
        return context


class StageFactory:
    """Build configured stateful pipeline stages."""

    def __init__(
        self,
        tracker_cls,
        line_counter_cls,
        region_filter_cls,
        region_timer_cls,
    ):
        self.tracker_cls = tracker_cls
        self.line_counter_cls = line_counter_cls
        self.region_filter_cls = region_filter_cls
        self.region_timer_cls = region_timer_cls

    def build(self, stages_config, source_config, source, video):
        stages = []
        components = {}
        for stage_config in stages_config:
            if not isinstance(stage_config, dict):
                raise ValueError("Each pipeline stage must be an object")
            stage_type = stage_config.get("type")
            if stage_config.get("enabled") is False and stage_type in (
                "tracker",
                "line_counter",
                "region_filter",
                "region_timer",
            ):
                continue

            if stage_type == "tracker":
                enabled = stage_config.get("enabled", True)
                if enabled == "auto":
                    enabled = _is_temporal_source(source_config, source)
                if not enabled:
                    continue
                tracker = self.tracker_cls(
                    track_thresh=stage_config.get("track_thresh", 0.25),
                    track_buffer=stage_config.get("track_buffer", 30),
                    match_thresh=stage_config.get("match_thresh", 0.8),
                    frame_rate=video.frame_rate,
                )
                stage = TrackerStage(tracker)
                components["tracker"] = tracker
            elif stage_type == "line_counter":
                line = stage_config.get("line")
                if not line or len(line) != 2:
                    raise ValueError("line_counter requires a two-point 'line'")
                counter = self.line_counter_cls(line)
                stage = LineCounterStage(counter)
                components["line_counter"] = counter
            elif stage_type == "region_filter":
                polygon = stage_config.get("polygon")
                if not polygon:
                    raise ValueError("region_filter requires a 'polygon'")
                region_filter = self.region_filter_cls(polygon)
                stage = RegionFilterStage(region_filter)
                components["region_filter"] = region_filter
            elif stage_type == "region_timer":
                polygon = stage_config.get("polygon")
                if not polygon:
                    raise ValueError("region_timer requires a 'polygon'")
                region_timer = self.region_timer_cls(polygon)
                stage = RegionTimerStage(region_timer)
                components["region_timer"] = region_timer
            else:
                raise ValueError(f"Unknown pipeline stage type: {stage_type}")
            stages.append(stage)
        return stages, components


class PipelineRenderer:
    """Render inference results and stage metrics for a pipeline frame."""

    def __init__(self, components, render_results=True, render_metrics=True):
        self.components = components
        self.render_results = render_results
        self.render_metrics = render_metrics

    def __call__(self, context):
        out = context.frame.copy()
        if self.render_metrics:
            counter = self.components.get("line_counter")
            if counter:
                from ..utils.draw import render_counter

                out = render_counter(
                    out,
                    counter.line,
                    f"In: {counter.in_count} | Out: {counter.out_count}",
                )
            region_timer = self.components.get("region_timer")
            if region_timer:
                from ..utils.draw import render_region

                out = render_region(
                    out,
                    region_timer.region,
                    f"Count: {context.metrics['region']['count']}",
                    color=(255, 255, 0),
                )
        if self.render_results:
            from ..utils.draw import render_results

            out = render_results(out, context.results)
        return out


class PipelineBuilder:
    """Assemble a :class:`Pipeline` from the version-one configuration."""

    def __init__(self, pipeline_cls):
        self.pipeline_cls = pipeline_cls

    def build(
        self,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable] = None,
        accelerator: Optional[str] = "auto",
    ):
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
        if not isinstance(display_config, dict):
            raise ValueError("Pipeline display must be an object")
        if not isinstance(stages_config, list):
            raise ValueError("Pipeline stages must be an array")

        from ..inference import Tracker
        from ..inference.registry import create_model, get_model_run_kwargs
        from ..inference.session import get_model_accelerator
        from .stages import LineCounter, RegionFilter, RegionTimer

        root = Path(base_dir or os.getcwd())
        source_plan = SourceFactory.prepare(source_config, display_config, root)
        model_kwargs = get_model_run_kwargs(model_config)
        model = create_model(
            model_config,
            base_dir=root,
            accelerator=accelerator,
        )
        video = None
        components = {}
        try:
            video = SourceFactory.create(source_plan)
            video.accelerator = get_model_accelerator(model)
            stage_factory = StageFactory(
                Tracker,
                LineCounter,
                RegionFilter,
                RegionTimer,
            )
            stages, components = stage_factory.build(
                stages_config,
                source_config,
                source_plan.source,
                video,
            )
        except Exception:
            close_resources([model, video, *components.values()])
            raise

        renderer = PipelineRenderer(
            components,
            render_results=display_config.get("render_results", True),
            render_metrics=display_config.get("render_metrics", True),
        )
        show_display = bool(display_config.get("show", True))
        if not show_display:
            # Keep the sink alive when a destination was configured, but do
            # not open an OpenCV preview window in headless runs.
            video.set_display_enabled(False)
        display = video if show_display or source_plan.video_kwargs.get("dest") else None
        return self.pipeline_cls(
            source=video,
            model=model,
            stages=stages,
            display=display,
            render=renderer,
            model_kwargs=model_kwargs,
            on_frame=on_frame,
            components=components,
        )


__all__ = [
    "LineCounterStage",
    "PipelineBuilder",
    "PipelineRenderer",
    "RegionFilterStage",
    "RegionTimerStage",
    "SourceFactory",
    "SourcePlan",
    "StageFactory",
    "TrackerStage",
    "_is_remote_source",
    "_is_temporal_source",
]
