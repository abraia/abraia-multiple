"""Versioned pipeline configuration and validation for runtime clients."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from abraia.sources import infer_source_type, normalize_source_type
from .pipeline_schema import (
    COMPOSITION_STAGE_TYPES,
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    SUPPORTED_STAGE_TYPES,
)
from .validation import validate_pipeline_draft
from abraia.inference.model_config import (
    DEFAULT_MODEL_URIS as SUPPORTED_MODEL_DEFAULT_URIS,
    ModelSpec,
    PIPELINE_HAILO_TASKS as SUPPORTED_HAILO_TASKS,
    PIPELINE_MODEL_KINDS as SUPPORTED_MODEL_KINDS,
    PIPELINE_MODEL_OPTIONS as SUPPORTED_MODEL_OPTIONS,
    PIPELINE_SECOND_STAGE_MODEL_OPTIONS as SUPPORTED_SECOND_STAGE_MODEL_OPTIONS,
    PIPELINE_SECOND_STAGE_MODEL_PAIRS,
    PIPELINE_MODEL_TASKS as SUPPORTED_MODEL_TASKS,
    model_control_visibility,
    model_defaults,
)


@dataclass
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
    model_uri: str = ""
    model_kind: str = "yolov8"
    model_task: str = "detection"
    model_index: str = ""
    labels: List[str] = field(default_factory=list)
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    approx: Optional[bool] = None
    stages: List[Dict[str, Any]] = field(default_factory=list)
    show: bool = True
    render_results: bool = True
    render_metrics: bool = True

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
    defaults = {
        "tracker": {"type": "tracker", "track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8},
        "line_counter": {"type": "line_counter", "line": [[0, 0], [100, 100]]},
        "region_filter": {"type": "region_filter", "polygon": [[0, 0], [100, 0], [100, 100]]},
        "region_timer": {"type": "region_timer", "polygon": [[0, 0], [100, 0], [100, 100]]},
        "filter": {"type": "filter", "id": "filter", "input": "results", "labels": [], "min_confidence": 0.0},
        "crop": {"type": "crop", "id": "crop", "input": "results", "padding": 0.0},
        "model": {
            "type": "model",
            "id": "read_text",
            "model": {"kind": "ocr", "task": "recognition"},
        },
    }
    if stage_type not in defaults:
        raise ValueError(f"Unsupported stage type: {stage_type}")
    return deepcopy(defaults[stage_type])


def parse_points(text):
    """Parse ``x, y; x, y`` text into JSON-compatible coordinate pairs."""
    points = []
    for raw_point in str(text or "").split(";"):
        values = [value.strip() for value in raw_point.split(",")]
        if len(values) != 2:
            continue
        try:
            points.append([float(values[0]), float(values[1])])
        except (TypeError, ValueError):
            continue
    return points


def format_points(points):
    """Format coordinate pairs for the pipeline editor's text fields."""
    return "; ".join(
        f"{point[0]:g}, {point[1]:g}"
        for point in points or []
        if isinstance(point, (list, tuple)) and len(point) == 2
    )


def load_pipeline_document(filename):
    """Load a version-one pipeline document from a JSON file."""
    with Path(filename).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_pipeline_document(filename, configuration):
    """Save a pipeline document as indented JSON with a trailing newline."""
    with Path(filename).open("w", encoding="utf-8") as stream:
        json.dump(configuration, stream, indent=2)
        stream.write("\n")


def _number(value, default, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def parse_pipeline_draft(config: Dict[str, Any], draft_class=None):
    """Build a :class:`PipelineDraft` from a version-one or version-two document."""
    draft_class = draft_class or PipelineDraft

    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    if config.get("version", 1) not in (1, 2):
        raise ValueError("Unsupported pipeline configuration version")
    source = config.get("source") or {}
    model = config.get("model") or {}
    display = config.get("display") or {}
    if not isinstance(source, dict):
        source = {}
    if not isinstance(model, dict):
        model = {}
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
    stages_config = config.get("stages", []) or []
    if config.get("version", 1) == 2:
        steps = config.get("steps", []) or []
        if not isinstance(steps, list):
            raise ValueError("Pipeline steps must be an array")
        primary = next(
            (
                step for step in steps
                if isinstance(step, dict)
                and step.get("type") == "model"
                and step.get("input", "frame") == "frame"
            ),
            None,
        )
        if primary is None:
            raise ValueError("A composed pipeline must define a frame model step")
        model = primary.get("model") or {}
        stages_config = [dict(step) for step in steps]
    stages = []
    for stage in stages_config:
        if isinstance(stage, dict):
            normalized_stage = dict(stage)
            stages.append(normalized_stage)
    labels_value = model.get("labels", [])
    if isinstance(labels_value, str):
        labels_value = labels_value.split(",")
    model_spec = ModelSpec.from_config(model)
    model_kind = model_spec.kind
    model_task = model_spec.task
    editor_values = model_spec.editor_values(model)
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
        model_uri=str(
            model.get("uri", "")
            if "uri" in model
            else model_spec.resolved_uri
        ),
        model_kind=model_kind,
        model_task=model_task,
        model_index=editor_values["model_index"],
        labels=[str(label).strip() for label in labels_value or [] if str(label).strip()],
        conf_threshold=editor_values["conf_threshold"],
        iou_threshold=editor_values["iou_threshold"],
        approx=model.get("approx"),
        stages=stages,
        show=bool(display.get("show", True)),
        render_results=bool(display.get("render_results", True)),
        render_metrics=bool(display.get("render_metrics", True)),
    )


def serialize_pipeline_draft(draft) -> Dict[str, Any]:
    """Serialize a pipeline draft into the appropriate versioned document."""
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
    model_spec = ModelSpec.from_config({
        "kind": draft.model_kind,
        "task": draft.model_task,
        "uri": draft.model_uri,
    })
    model_task = model_spec.task
    if model_task not in SUPPORTED_MODEL_TASKS:
        raise ValueError(f"Unsupported pipeline model task: {model_task}")
    model = model_spec.to_pipeline_config(
        labels=draft.labels,
        conf_threshold=draft.conf_threshold,
        iou_threshold=draft.iou_threshold,
        approx=draft.approx,
        model_index=draft.model_index,
    )
    stages = [dict(stage) for stage in draft.stages]
    canonical_stages = bool(
        stages
        and isinstance(stages[0], dict)
        and stages[0].get("type") == "model"
        and stages[0].get("input", "frame") == "frame"
    )
    composed = canonical_stages or any(
        stage.get("type") in COMPOSITION_STAGE_TYPES
        for stage in stages
        if isinstance(stage, dict)
    )
    display = {
        "show": bool(draft.show),
        "render_results": bool(draft.render_results),
        "render_metrics": bool(draft.render_metrics),
    }
    if not composed:
        return {
            "version": 1,
            "source": source,
            "model": model,
            "stages": stages,
            "display": display,
        }

    if canonical_stages:
        steps = []
        for index, stage in enumerate(stages, 1):
            stage.setdefault("id", "model" if index == 1 else f"step-{index - 1}")
            steps.append(stage)
        return {
            "version": 2,
            "source": source,
            "steps": steps,
            "display": display,
        }

    if any(
        stage.get("type") not in COMPOSITION_STAGE_TYPES
        for stage in stages
        if isinstance(stage, dict)
    ):
        raise ValueError(
            "Composed pipelines currently support model, filter, crop, and attach steps only."
        )
    steps = [{"id": "model", "type": "model", "input": "frame", "model": model}]
    for index, stage in enumerate(stages, 1):
        stage.setdefault("id", f"step-{index}")
        steps.append(stage)
    return {
        "version": 2,
        "source": source,
        "steps": steps,
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
    "COMPOSITION_STAGE_TYPES",
    "MAX_PIPELINE_TRACKERS",
    "MAX_PIPELINE_MODELS",
    "default_stage",
    "format_points",
    "infer_source_type",
    "load_pipeline_document",
    "model_control_visibility",
    "model_defaults",
    "parse_points",
    "parse_pipeline_draft",
    "save_pipeline_document",
    "serialize_pipeline_draft",
    "validate_pipeline_draft",
]
