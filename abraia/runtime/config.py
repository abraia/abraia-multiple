"""Versioned pipeline configuration and validation for runtime clients."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from abraia.sources import infer_source_type, normalize_source_type
from .pipeline_schema import (
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    normalize_points,
    PIPELINE_STAGE_DEFINITIONS,
    SUPPORTED_STAGE_TYPES,
    normalize_step_ids,
    strip_internal_step_ids,
    is_primary_model_step,
)
from .validation import validate_pipeline_draft
from abraia.inference.model_config import (
    DEFAULT_MODEL_URIS as SUPPORTED_MODEL_DEFAULT_URIS,
    ModelSpec,
    PIPELINE_HAILO_TASKS as SUPPORTED_HAILO_TASKS,
    PIPELINE_MODEL_KINDS as SUPPORTED_MODEL_KINDS,
    PIPELINE_MODEL_OPTIONS as SUPPORTED_MODEL_OPTIONS,
    PIPELINE_SECOND_STAGE_MODEL_OPTIONS as SUPPORTED_SECOND_STAGE_MODEL_OPTIONS,
    PIPELINE_MODEL_TASKS as SUPPORTED_MODEL_TASKS,
    model_control_visibility,
    model_defaults,
)


def default_primary_stage():
    model = ModelSpec.from_config({
        "kind": "yolov8",
        "task": "detection",
    }).to_pipeline_config()
    return {
        "type": "model",
        "model": model,
    }


@dataclass(init=False)
class PipelineDraft:
    """Editable, JSON-compatible representation of a pipeline.

    ``validation_errors`` is intentionally independent from SDK imports so it
    can be used by the Qt view and by offline tests. Runtime component
    validation remains the responsibility of :class:`abraia.runtime.Pipeline`.
    """

    source_type: str = "camera"
    source: str = "0"
    resolution: List[int] = field(default_factory=lambda: [1920, 1080])
    fps: int = 30
    steps: List[Dict[str, Any]] = field(default_factory=list)
    show: bool = True
    render_results: bool = True
    render_metrics: bool = True

    def __init__(
        self,
        source_type="camera",
        source="0",
        resolution=None,
        fps=30,
        show=True,
        render_results=True,
        render_metrics=True,
        *,
        steps=None,
    ):
        """Create a draft with ordered version-two steps."""
        self.source_type = source_type
        self.source = source
        self.resolution = [1920, 1080] if resolution is None else resolution
        self.fps = fps
        self.steps = (
            [default_primary_stage()]
            if steps is None
            else list(steps)
        )
        self.show = show
        self.render_results = render_results
        self.render_metrics = render_metrics

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PipelineDraft":
        return parse_pipeline_draft(config, draft_class=cls)

    def to_dict(self) -> Dict[str, Any]:
        return serialize_pipeline_draft(self)

    def validation_errors(self) -> List[str]:
        return validate_pipeline_draft(self)

    def is_valid(self):
        return not self.validation_errors()


def default_stage(stage_type):
    """Return a fresh stage config suitable for an editor card."""
    definition = PIPELINE_STAGE_DEFINITIONS.get(stage_type)
    if definition is None:
        raise ValueError(f"Unsupported stage type: {stage_type}")
    return deepcopy(definition["default"])


def parse_points(text):
    """Parse ``x, y; x, y`` text into JSON-compatible coordinate pairs."""
    return normalize_points(str(text or ""))


def format_points(points):
    """Format coordinate pairs for the pipeline editor's text fields."""
    return "; ".join(
        f"{point[0]:g}, {point[1]:g}"
        for point in points or []
        if isinstance(point, (list, tuple)) and len(point) == 2
    )


def _number(value, default, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def parse_pipeline_draft(config: Dict[str, Any], draft_class=None):
    """Build a :class:`PipelineDraft` from a versioned pipeline document."""
    draft_class = draft_class or PipelineDraft

    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    if config.get("version") != 2:
        raise ValueError("Only version 2 pipeline documents are supported")
    source = config.get("source") or {}
    display = config.get("display") or {}
    if not isinstance(source, dict):
        source = {}
    if not isinstance(display, dict):
        display = {}
    resolution = source.get("resolution", [1920, 1080])
    if isinstance(resolution, tuple):
        resolution = list(resolution)
    if not isinstance(resolution, list):
        resolution = [1920, 1080]
    resolution = resolution[:2]
    while len(resolution) < 2:
        resolution.append([1920, 1080][len(resolution)])
    steps_config = config.get("steps", []) or []
    if not isinstance(steps_config, list):
        raise ValueError("Pipeline steps must be an array")
    if any(not isinstance(step, dict) for step in steps_config):
        raise ValueError("Pipeline steps must contain objects")
    if steps_config and not is_primary_model_step(steps_config[0]):
        raise ValueError(
            "A version 2 pipeline with stages must start with a frame model step"
        )
    normalized_steps = normalize_step_ids(steps_config)
    source_default = "0" if source.get("type") in (None, "camera") else ""
    source_value = str(source.get("src", source_default)).strip()
    configured_source_type = source.get("type")
    if configured_source_type is None:
        source_type = infer_source_type(
            source_value,
            fallback="camera" if not source_value else "image",
        )
    else:
        source_type = normalize_source_type(configured_source_type)
    return draft_class(
        source_type=source_type or "camera",
        source=source_value,
        resolution=[_number(v, 0, int) for v in resolution],
        fps=_number(source.get("fps", 30), 30, int),
        steps=normalized_steps,
        show=bool(display.get("show", True)),
        render_results=bool(display.get("render_results", True)),
        render_metrics=bool(display.get("render_metrics", True)),
    )


def serialize_pipeline_draft(draft) -> Dict[str, Any]:
    """Serialize a pipeline draft as a version-two document."""
    source_value = "" if draft.source is None else str(draft.source).strip()
    source_type = infer_source_type(
        source_value,
        fallback=normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    source = {
        "type": source_type,
        "src": source_value,
        "resolution": [int(v) for v in (draft.resolution or [1920, 1080])[:2]],
        "fps": int(draft.fps),
    }
    steps = [dict(step) for step in draft.steps]
    canonical_steps = is_primary_model_step(steps[0]) if steps else True
    display = {
        "show": bool(draft.show),
        "render_results": bool(draft.render_results),
        "render_metrics": bool(draft.render_metrics),
    }
    if any(
        step.get("type") not in SUPPORTED_STAGE_TYPES
        for step in steps
        if isinstance(step, dict)
    ):
        raise ValueError("Pipeline contains an unsupported step type.")

    if not canonical_steps:
        raise ValueError(
            "A version 2 pipeline with stages must start with a model step."
        )
    pipeline_steps = strip_internal_step_ids(steps)

    return {
        "version": 2,
        "source": source,
        "steps": pipeline_steps,
        "display": display,
    }


__all__ = [
    "PipelineDraft",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_MODEL_DEFAULT_URIS",
    "SUPPORTED_MODEL_OPTIONS",
    "SUPPORTED_SECOND_STAGE_MODEL_OPTIONS",
    "SUPPORTED_MODEL_TASKS",
    "SUPPORTED_STAGE_TYPES",
    "MAX_PIPELINE_TRACKERS",
    "MAX_PIPELINE_MODELS",
    "default_primary_stage",
    "default_stage",
    "format_points",
    "infer_source_type",
    "model_control_visibility",
    "model_defaults",
    "parse_points",
    "parse_pipeline_draft",
    "serialize_pipeline_draft",
    "validate_pipeline_draft",
]
