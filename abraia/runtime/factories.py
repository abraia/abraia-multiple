"""Factories used to assemble runtime inference pipelines."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .lifecycle import close_resources
from .config import (
    ModelSpec,
)
from .composition import (
    LineCounterStep,
    RegionFilterStep,
    RegionTimerStep,
    TrackerStep,
)


def _is_remote_source(source: str) -> bool:
    source = source.lower()
    return source.startswith(("http://", "https://", "rtsp://"))


def _is_spectral_source(source: Any) -> bool:
    """Return whether a source path identifies a spectral capture."""
    if not isinstance(source, str):
        return False
    return source.lower().endswith(
        (".tif", ".tiff", ".hdr", ".raw", ".img")
    )


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


_SPECTRAL_SOURCE_FACTORY = None


def register_spectral_source(factory):
    """Register the application supplied source for spectral data."""
    if not callable(factory):
        raise TypeError("Spectral source factory must be callable")
    global _SPECTRAL_SOURCE_FACTORY
    _SPECTRAL_SOURCE_FACTORY = factory


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
    def create(plan: SourcePlan, spectral=False):
        if spectral:
            if _SPECTRAL_SOURCE_FACTORY is None:
                raise ValueError(
                    "Spectral pipeline support has not been registered"
                )
            return _SPECTRAL_SOURCE_FACTORY(plan.source, **plan.video_kwargs)
        from .video import Video

        return Video(plan.source, **plan.video_kwargs)


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


def build_pipeline_output(video, source_plan, display_config, components):
    """Create shared rendering and display configuration for a pipeline."""
    renderer = PipelineRenderer(
        components,
        render_results=display_config.get("render_results", True),
        render_metrics=display_config.get("render_metrics", True),
    )
    show_display = bool(display_config.get("show", True))
    if not show_display:
        video.set_display_enabled(False)
    display = video if show_display or source_plan.video_kwargs.get("dest") else None
    return renderer, display


class PipelineBuilder:
    """Assemble a :class:`Pipeline` from a versioned configuration."""

    def __init__(self, pipeline_cls):
        self.pipeline_cls = pipeline_cls

    def build(
        self,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable] = None,
        accelerator: Optional[str] = "auto",
        renderer_factory: Optional[Callable] = None,
    ):
        if not isinstance(config, dict):
            raise ValueError("Pipeline configuration must be a JSON object")
        if config.get("version") != 2:
            raise ValueError("Only version 2 pipeline configurations are supported")
        from .composed_builder import build_composed_pipeline

        return build_composed_pipeline(
            self.pipeline_cls,
            config,
            base_dir=base_dir,
            on_frame=on_frame,
            accelerator=accelerator,
            renderer_factory=renderer_factory,
        )

__all__ = [
    "PipelineBuilder",
    "PipelineRenderer",
    "SourceFactory",
    "register_spectral_source",
    "SourcePlan",
    "_is_remote_source",
    "_is_temporal_source",
]
