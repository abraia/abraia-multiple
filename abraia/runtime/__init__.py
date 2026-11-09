"""Public pipeline execution, configuration and extension interfaces."""

from .pipeline import FrameContext, Pipeline, prepare_source
from .execution import run_pipeline
from .discovery import (
    discover_cameras, discover_genicam_devices, discover_streams,
)
from .composition import register_model_step_factory
from .sources import create_source, infer_source_type, register_source_factory
from ..media.sources import (
    CameraSource,
    FrameSource,
    GenICamSource,
    ImageSequence,
    ImageSequenceSource,
    ImageSource,
    StreamSource,
    VideoSource,
)
from .config import (
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    PIPELINE_STEP_DEFINITIONS,
    PipelineDraft,
    SUPPORTED_MODEL_DEFAULT_URIS,
    SUPPORTED_MODEL_OPTIONS,
    SUPPORTED_SECOND_STEP_MODEL_OPTIONS,
    SUPPORTED_STEP_TYPES,
    default_primary_step,
    default_step,
    format_points,
    is_primary_model_step,
    model_control_visibility,
    model_defaults,
    model_editor_values,
    parse_points,
    steps_missing_tracker,
)

__all__ = [
    "FrameContext", "FrameSource", "GenICamSource", "ImageSequence",
    "Pipeline", "PipelineDraft", "create_source", "prepare_source",
    "run_pipeline", "discover_cameras", "discover_genicam_devices",
    "discover_streams",
    "ImageSource", "ImageSequenceSource", "VideoSource", "CameraSource",
    "StreamSource",
    "MAX_PIPELINE_MODELS", "MAX_PIPELINE_TRACKERS", "PIPELINE_STEP_DEFINITIONS",
    "SUPPORTED_MODEL_DEFAULT_URIS", "SUPPORTED_MODEL_OPTIONS",
    "SUPPORTED_SECOND_STEP_MODEL_OPTIONS", "SUPPORTED_STEP_TYPES",
    "default_primary_step", "default_step", "format_points", "infer_source_type",
    "is_primary_model_step", "model_control_visibility", "model_defaults",
    "model_editor_values", "parse_points", "steps_missing_tracker",
    "register_model_step_factory", "register_source_factory",
]
