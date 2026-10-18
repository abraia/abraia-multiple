"""Versioned pipeline configuration and validation for runtime clients."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from abraia.sources import infer_source_type, normalize_source_type
from .pipeline_schema import (
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    normalize_points,
    PIPELINE_STEP_DEFINITIONS,
    SUPPORTED_STEP_TYPES,
    normalize_step_ids,
    strip_internal_step_ids,
    starts_with_model_step,
    iter_pipeline_steps,
    is_valid_step_geometry,
    validate_step_structure,
    PIPELINE_SECOND_STEP_MODEL_PAIRS,
)
from abraia.inference.model_config import (
    DEFAULT_MODEL_URIS as SUPPORTED_MODEL_DEFAULT_URIS,
    MODEL_ARCHITECTURES,
    MODEL_DESCRIPTORS,
    MODEL_SIZE_URIS,
    ModelSpec,
    PIPELINE_HAILO_TASKS as SUPPORTED_HAILO_TASKS,
    PIPELINE_MODEL_KINDS as SUPPORTED_MODEL_KINDS,
    PIPELINE_MODEL_TASKS as SUPPORTED_MODEL_TASKS,
)
from abraia.tasks import normalize_task


MODEL_DEFAULT_THRESHOLDS = {
    ("yolov8", "detection"): (0.25, 0.45),
    ("yolo9", "detection"): (0.25, 0.45),
    ("yolov8", "segmentation"): (0.25, 0.7),
    ("yolov8", "pose"): (0.25, 0.7),
    ("rtmdet", "detection"): (0.25, 0.7),
    ("rtmdet", "segmentation"): (0.25, 0.7),
    ("resnet", "classification"): (0.25, 0.0),
    ("grounding_dino", "detection"): (0.25, 0.45),
    ("face", "detection"): (0.75, 0.5),
    ("face", "recognition"): (0.45, 0.5),
    ("license_plate", "detection"): (0.5, 0.1),
    ("license_plate", "recognition"): (0.85, 0.15),
    ("ocr", "recognition"): (0.5, 0.0),
}


def _sized_model_options(kind, label, task):
    size_labels = {"small": "Small", "medium": "Medium", "large": "Large"}
    return tuple(
        (f"{kind}_{task}_{size}", f"{label} ({size_labels[size]})", kind, task, uri)
        for size, uri in MODEL_SIZE_URIS[kind][task].items()
    )


SUPPORTED_MODEL_OPTIONS = (
    *_sized_model_options("yolov8", "Object detection", "detection"),
    *_sized_model_options("yolo9", "YOLOv9 object detection", "detection"),
    *_sized_model_options("yolov8", "Instance segmentation", "segmentation"),
    *_sized_model_options("yolov8", "Pose estimation", "pose"),
    *_sized_model_options("rtmdet", "RTMDet object detection", "detection"),
    *_sized_model_options("rtmdet", "RTMDet instance segmentation", "segmentation"),
    *_sized_model_options("resnet", "ResNet classification", "classification"),
    ("face_detection", "Face detection", "face", "detection", ""),
    ("face_recognition", "Face recognition", "face", "recognition", ""),
    ("license_plate_detection", "License plate detection", "license_plate", "detection", ""),
    ("license_plate_recognition", "License plate recognition", "license_plate", "recognition", ""),
    ("ocr_recognition", "OCR recognition", "ocr", "recognition", ""),
)
SUPPORTED_SECOND_STEP_MODEL_OPTIONS = tuple(
    option for option in SUPPORTED_MODEL_OPTIONS
    if (option[2], option[3]) in PIPELINE_SECOND_STEP_MODEL_PAIRS
) + ((
    "multispectral_classification",
    "Multispectral classification",
    "multispectral",
    "classification",
    "",
),)


def model_defaults(kind, task):
    kind = str(kind or "").strip().lower()
    task = normalize_task(
        task, default="classification" if kind == "resnet" else "detection"
    )
    if kind in MODEL_ARCHITECTURES and (kind, task) not in MODEL_DEFAULT_THRESHOLDS:
        kind = "yolov8"
    return MODEL_DEFAULT_THRESHOLDS.get((kind, task), (0.25, 0.45))


def model_control_visibility(kind, task):
    kind = str(kind or "").strip().lower()
    task = normalize_task(
        task, default="classification" if kind == "resnet" else "detection"
    )
    descriptor = MODEL_DESCRIPTORS.get(kind)
    uses_uri = bool(descriptor and descriptor.uri_based)
    return {
        "uri": uses_uri,
        "index": kind == "face" and task == "recognition",
        "labels": uses_uri,
        "confidence": True,
        "iou": kind not in ("ocr", "resnet", "multispectral")
        and not (kind == "face" and task == "recognition"),
        "approx": kind in MODEL_ARCHITECTURES and task == "segmentation",
    }


def model_pipeline_errors(model):
    spec = model if isinstance(model, ModelSpec) else ModelSpec.from_config(model)
    if spec.kind not in SUPPORTED_MODEL_KINDS:
        return ("Choose a supported detector kind.",)
    if spec.task not in SUPPORTED_MODEL_TASKS:
        return ("Choose a supported detector task.",)
    descriptor = spec.descriptor
    if descriptor and spec.task not in descriptor.tasks:
        messages = {
            "resnet": "Classification models require the classification task.",
            "grounding_dino": "Grounding DINO models require the detection task.",
            "ocr": "OCR models only support the recognition task.",
            "multispectral": "Multispectral models require the classification task.",
        }
        if spec.kind in MODEL_ARCHITECTURES:
            return (f"{spec.kind} models support detection, segmentation, or pose.",)
        return (
            messages.get(
                spec.kind,
                f"{spec.kind} models do not support the {spec.task} task.",
            ),
        )
    if descriptor and descriptor.explicit_uri and not spec.uri:
        return ("Enter a model URI or local model path.",)
    if (
        spec.kind == "face"
        and spec.task == "recognition"
        and not spec.params.get("index")
    ):
        return ("Face recognition requires an index JSON file.",)
    return ()


def model_editor_values(model, config=None):
    spec = model if isinstance(model, ModelSpec) else ModelSpec.from_config(model)
    if not isinstance(config, Mapping):
        config = {}
    params = spec.params

    def value(name, fallback=None):
        return params.get(name, config.get(name, fallback))

    if spec.kind == "face":
        confidence = value("prob_threshold", config.get("conf_threshold"))
        if spec.task == "recognition":
            confidence = value("threshold", confidence)
        iou = value("iou_threshold", config.get("iou_threshold"))
    elif spec.kind == "license_plate":
        confidence = value("threshold", config.get("conf_threshold"))
        iou = value("iou_threshold", config.get("iou_threshold"))
    elif spec.kind == "ocr":
        confidence, iou = value("drop_score", config.get("conf_threshold")), None
    else:
        confidence, iou = config.get("conf_threshold"), config.get("iou_threshold")
    return {
        "conf_threshold": confidence,
        "iou_threshold": iou,
        "model_index": str(params.get("index", "")),
    }


def default_primary_step():
    return {
        "type": "model",
        "model": {
            "task": "detection",
            "kind": "yolov8",
            "uri": SUPPORTED_MODEL_DEFAULT_URIS["yolov8"]["detection"],
        },
    }


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
    show: bool = True
    render_results: bool = True
    render_metrics: bool = True
    steps: List[Dict[str, Any]] = field(
        default_factory=lambda: [default_primary_step()]
    )

    def __post_init__(self):
        if self.resolution is None:
            self.resolution = [1920, 1080]
        self.steps = (
            [default_primary_step()] if self.steps is None else list(self.steps)
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PipelineDraft":
        return _parse_pipeline_draft(config)

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_pipeline_draft(self)

    def validation_errors(self) -> List[str]:
        return _validate_pipeline_draft(self)


def default_step(step_type):
    """Return a fresh step config suitable for an editor card."""
    definition = PIPELINE_STEP_DEFINITIONS.get(step_type)
    if definition is None:
        raise ValueError(f"Unsupported step type: {step_type}")
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


def _parse_pipeline_draft(config: Dict[str, Any]):
    """Build a :class:`PipelineDraft` from a versioned pipeline document."""
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
    if not starts_with_model_step(steps_config):
        raise ValueError(
            "A version 2 pipeline with steps must start with a frame model step"
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
    return PipelineDraft(
        source_type=source_type or "camera",
        source=source_value,
        resolution=[_number(v, 0, int) for v in resolution],
        fps=_number(source.get("fps", 30), 30, int),
        steps=normalized_steps,
        show=bool(display.get("show", True)),
        render_results=bool(display.get("render_results", True)),
        render_metrics=bool(display.get("render_metrics", True)),
    )


def _serialize_pipeline_draft(draft) -> Dict[str, Any]:
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
    canonical_steps = starts_with_model_step(steps)
    display = {
        "show": bool(draft.show),
        "render_results": bool(draft.render_results),
        "render_metrics": bool(draft.render_metrics),
    }
    if any(
        step.get("type") not in SUPPORTED_STEP_TYPES
        for step in steps
        if isinstance(step, dict)
    ):
        raise ValueError("Pipeline contains an unsupported step type.")

    if not canonical_steps:
        raise ValueError(
            "A version 2 pipeline with steps must start with a model step."
        )
    pipeline_steps = strip_internal_step_ids(steps)

    return {
        "version": 2,
        "source": source,
        "steps": pipeline_steps,
        "display": display,
    }


def _validate_pipeline_draft(draft):
    """Return user-facing validation messages for a pipeline draft."""
    errors = []
    source_value = "" if draft.source is None else str(draft.source).strip()
    source_type = infer_source_type(
        source_value,
        fallback=normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    if not source_value:
        errors.append("Choose a source file, camera, or stream.")
    elif source_type == "camera" and infer_source_type(source_value) != "camera":
        errors.append("Camera source must be a numeric camera index.")
    try:
        resolution_valid = len(draft.resolution) == 2 and all(
            int(value) > 0 for value in draft.resolution
        )
    except (TypeError, ValueError):
        resolution_valid = False
    if not resolution_valid:
        errors.append("Resolution must contain two positive values.")
    try:
        fps_valid = int(draft.fps) > 0
    except (TypeError, ValueError):
        fps_valid = False
    if not fps_valid:
        errors.append("FPS must be greater than zero.")
    steps = draft.steps
    if not starts_with_model_step(steps):
        errors.append(
            "A version 2 pipeline with steps must start with a frame model step."
        )
    errors.extend(validate_step_structure(steps))
    for index, step in enumerate(iter_pipeline_steps(steps), 1):
        if not step.is_mapping or not step.is_active:
            continue
        step_type = step.type
        model = step.get("model") if step_type == "model" else None
        if isinstance(model, dict):
            spec = ModelSpec.from_config(model)
            errors.extend(model_pipeline_errors(spec))
            editor_values = model_editor_values(spec, model)
            for name, label in (
                ("conf_threshold", "Confidence threshold"),
                ("iou_threshold", "IoU threshold"),
            ):
                value = editor_values[name]
                if value is None:
                    continue
                try:
                    valid = 0 <= float(value) <= 1
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(f"{label} must be between 0 and 1.")
        if step_type == "line_counter" and not is_valid_step_geometry(
            step_type, step
        ):
            errors.append(f"Step {index}: line counter needs two points.")
        if step_type in ("region_filter", "region_timer") and not is_valid_step_geometry(
            step_type, step
        ):
            errors.append(f"Step {index}: region needs at least three points.")
    return errors


__all__ = [
    "PipelineDraft",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_MODEL_DEFAULT_URIS",
    "SUPPORTED_MODEL_OPTIONS",
    "SUPPORTED_SECOND_STEP_MODEL_OPTIONS",
    "SUPPORTED_MODEL_TASKS",
    "SUPPORTED_STEP_TYPES",
    "MAX_PIPELINE_TRACKERS",
    "MAX_PIPELINE_MODELS",
    "default_primary_step",
    "default_step",
    "format_points",
    "infer_source_type",
    "model_control_visibility",
    "model_defaults",
    "parse_points",
]
