"""Runtime components for media input and inference pipelines."""

from .factories import register_spectral_source
from .pipeline import (
    FrameContext,
    CancellableSource,
    Pipeline,
)
from .composition import (
    BoundModelStep,
    CropStep,
    FilterStep,
    LineCounterStep,
    ModelStep,
    RegionFilterStep,
    RegionInput,
    RegionTimerStep,
    TrackerStep,
    crop_region,
)
from .inference import (
    AsyncInferenceRunner,
    FrameBatch,
    FrameRecord,
    FrameResult,
)
from .output import VideoOutput
from .video import FrameSource, Video
from .stages import LineCounter, RegionFilter, RegionTimer, count_objects
from .config import (
    PipelineDraft,
    SUPPORTED_HAILO_TASKS,
    SUPPORTED_MODEL_DEFAULT_URIS,
    SUPPORTED_MODEL_KINDS,
    SUPPORTED_MODEL_OPTIONS,
    SUPPORTED_SECOND_STAGE_MODEL_OPTIONS,
    SUPPORTED_MODEL_TASKS,
    SUPPORTED_STAGE_TYPES,
    default_primary_stage,
    default_stage,
    format_points,
    model_control_visibility,
    model_defaults,
    parse_points,
)
from .runner import prepare_pipeline_display_frame, run_pipeline
from .pipeline_schema import (
    PIPELINE_STAGE_DEFINITIONS,
    PipelineStep,
    iter_pipeline_steps,
    TRACKER_DEPENDENT_STAGE_TYPES,
    normalize_step_ids,
    strip_internal_step_ids,
)

__all__ = [
    "FrameContext",
    "CancellableSource",
    "FrameSource",
    "AsyncInferenceRunner",
    "FrameBatch",
    "FrameRecord",
    "FrameResult",
    "LineCounter",
    "Pipeline",
    "RegionTimer",
    "RegionFilter",
    "count_objects",
    "BoundModelStep",
    "CropStep",
    "FilterStep",
    "LineCounterStep",
    "RegionFilterStep",
    "RegionTimerStep",
    "TrackerStep",
    "ModelStep",
    "RegionInput",
    "crop_region",
    "Video",
    "VideoOutput",
    "PipelineDraft",
    "SUPPORTED_HAILO_TASKS",
    "SUPPORTED_MODEL_DEFAULT_URIS",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_MODEL_OPTIONS",
    "SUPPORTED_SECOND_STAGE_MODEL_OPTIONS",
    "SUPPORTED_MODEL_TASKS",
    "SUPPORTED_STAGE_TYPES",
    "PIPELINE_STAGE_DEFINITIONS",
    "TRACKER_DEPENDENT_STAGE_TYPES",
    "PipelineStep",
    "iter_pipeline_steps",
    "normalize_step_ids",
    "strip_internal_step_ids",
    "default_primary_stage",
    "default_stage",
    "format_points",
    "model_control_visibility",
    "model_defaults",
    "parse_points",
    "prepare_pipeline_display_frame",
    "run_pipeline",
    "register_spectral_source",
]
