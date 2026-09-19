"""Version-one pipeline configuration and validation for runtime clients."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from abraia.sources import infer_source_type
from abraia.tasks import normalize_task
from abraia.inference.model_config import (
    DEFAULT_MODEL_URIS as SUPPORTED_MODEL_DEFAULT_URIS,
    PIPELINE_HAILO_TASKS as SUPPORTED_HAILO_TASKS,
    PIPELINE_MODEL_KINDS as SUPPORTED_MODEL_KINDS,
    PIPELINE_MODEL_OPTIONS as SUPPORTED_MODEL_OPTIONS,
    PIPELINE_MODEL_TASKS as SUPPORTED_MODEL_TASKS,
    model_control_visibility,
    model_defaults,
)


SUPPORTED_STAGE_TYPES = ("tracker", "line_counter", "region_filter", "region_timer")


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
    model_kind: str = "object_detection"
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

    def validate(self):
        """Return validation messages, or an empty list when valid."""
        return self.validation_errors()


def default_stage(stage_type):
    """Return a fresh stage config suitable for an editor card."""
    defaults = {
        "tracker": {"type": "tracker", "enabled": True, "track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8},
        "line_counter": {"type": "line_counter", "line": [[0, 0], [100, 100]]},
        "region_filter": {"type": "region_filter", "polygon": [[0, 0], [100, 0], [100, 100]]},
        "region_timer": {"type": "region_timer", "polygon": [[0, 0], [100, 0], [100, 100]]},
    }
    if stage_type not in defaults:
        raise ValueError(f"Unsupported stage type: {stage_type}")
    return deepcopy(defaults[stage_type])


def _normalize_source_type(source_type: Any) -> str:
    aliases = {
        "images": "image",
        "usb_camera": "camera",
        "rpi_camera": "camera",
    }
    normalized = str(source_type or "").strip().lower()
    return aliases.get(normalized, normalized)


def _number(value, default, cast=float):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def parse_pipeline_draft(config: Dict[str, Any], draft_class=None):
    """Build a :class:`PipelineDraft` from a version-one document."""
    draft_class = draft_class or PipelineDraft

    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    if config.get("version", 1) != 1:
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
    stages = []
    for stage in config.get("stages", []) or []:
        if isinstance(stage, dict):
            normalized_stage = dict(stage)
            if normalized_stage.get("type") == "counter":
                normalized_stage["type"] = "line_counter"
            elif normalized_stage.get("type") == "region":
                normalized_stage["type"] = "region_filter"
            stages.append(normalized_stage)
    labels_value = model.get("labels", [])
    if isinstance(labels_value, str):
        labels_value = labels_value.split(",")
    model_kind = str(model.get("kind", "object_detection")).strip().lower()
    model_kind = {
        "classification": "resnet",
        "face_detector": "face",
        "license_plate_detector": "license_plate",
        "plate": "license_plate",
    }.get(model_kind, model_kind)
    model_task = normalize_task(model.get("task"), default="detection")
    params = model.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    if model_kind == "face":
        conf_threshold = params.get("prob_threshold", model.get("conf_threshold"))
        iou_threshold = params.get("iou_threshold", model.get("iou_threshold"))
        if model_task == "recognition":
            conf_threshold = params.get("threshold", conf_threshold)
    elif model_kind == "license_plate":
        conf_threshold = params.get("threshold", model.get("conf_threshold"))
        iou_threshold = params.get("iou_threshold", model.get("iou_threshold"))
    elif model_kind == "ocr":
        conf_threshold = params.get("drop_score", model.get("conf_threshold"))
        iou_threshold = None
    else:
        conf_threshold = model.get("conf_threshold")
        iou_threshold = model.get("iou_threshold")
    source_default = "0" if source.get("type") in (None, "camera") else ""
    source_value = str(source.get("src", source_default)).strip()
    configured_source_type = source.get("type")
    if configured_source_type is None:
        source_type = infer_source_type(
            source_value,
            fallback="camera" if not source_value else "image",
        )
    else:
        source_type = _normalize_source_type(configured_source_type)
    return draft_class(
        source_type=source_type or "camera",
        source=source_value,
        resolution=[_number(v, 0, int) for v in resolution],
        fps=_number(source.get("fps", 30), 30, int),
        model_uri=str(
            model.get("uri", "") or SUPPORTED_MODEL_DEFAULT_URIS.get(model_kind, "")
        ),
        model_kind=model_kind,
        model_task=model_task,
        model_index=str(params.get("index", "")),
        labels=[str(label).strip() for label in labels_value or [] if str(label).strip()],
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        approx=model.get("approx"),
        stages=stages,
        show=bool(display.get("show", True)),
        render_results=bool(display.get("render_results", True)),
        render_metrics=bool(display.get("render_metrics", True)),
    )


def serialize_pipeline_draft(draft) -> Dict[str, Any]:
    """Serialize a pipeline draft into the stable version-one document."""
    source_value = "" if draft.source is None else str(draft.source).strip()
    source_type = infer_source_type(
        source_value,
        fallback=_normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    source = {
        "type": source_type,
        "src": source_value,
        "resolution": [int(v) for v in (draft.resolution or [1920, 1080])[:2]],
        "fps": int(draft.fps),
    }
    model_kind = (draft.model_kind or "object_detection").strip().lower()
    model_task = normalize_task(draft.model_task, default="detection")
    if model_task not in SUPPORTED_MODEL_TASKS:
        raise ValueError(f"Unsupported pipeline model task: {model_task}")
    model = {"task": model_task, "kind": model_kind}
    if model_kind in (
        "onnx", "object_detection", "instance_segmentation", "pose",
        "classification", "resnet", "hailo",
    ) and (model_kind == "hailo" or model_task in ("detection", "pose", "classification")):
        model["uri"] = draft.model_uri.strip() or SUPPORTED_MODEL_DEFAULT_URIS.get(model_kind, "")
        if draft.labels:
            model["labels"] = list(draft.labels)
        if draft.conf_threshold is not None:
            model["conf_threshold"] = float(draft.conf_threshold)
        if draft.iou_threshold is not None:
            model["iou_threshold"] = float(draft.iou_threshold)
        if draft.approx is not None and model_kind == "instance_segmentation":
            model["approx"] = bool(draft.approx)
    else:
        model["params"] = {}
        if draft.conf_threshold is not None:
            if model_task == "recognition":
                parameter = "drop_score" if model_kind == "ocr" else "threshold"
            else:
                parameter = "prob_threshold" if model_kind == "face" else "threshold"
            model["params"][parameter] = float(draft.conf_threshold)
        if draft.iou_threshold is not None:
            model["params"]["iou_threshold"] = float(draft.iou_threshold)
        if model_kind == "face" and model_task == "recognition" and draft.model_index.strip():
            model["params"]["index"] = draft.model_index.strip()
    return {
        "version": 1,
        "source": source,
        "model": model,
        "stages": [dict(stage) for stage in draft.stages],
        "display": {
            "show": bool(draft.show),
            "render_results": bool(draft.render_results),
            "render_metrics": bool(draft.render_metrics),
        },
    }


def _points(value):
    """Normalize a sequence of coordinate pairs without mutating input."""
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return []
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return []
    return points


def validate_pipeline_draft(draft):
    """Return user-facing validation messages for a pipeline draft."""
    errors = []
    source_value = "" if draft.source is None else str(draft.source).strip()
    source_type = infer_source_type(
        source_value,
        fallback=_normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    if not source_value:
        errors.append("Choose a source file, camera, or stream.")
    elif source_type == "camera" and not source_value.isdigit():
        errors.append("Camera source must be a numeric camera index.")
    model_kind = (draft.model_kind or "object_detection").strip().lower()
    model_task = normalize_task(draft.model_task, default="detection")
    if model_kind not in SUPPORTED_MODEL_KINDS:
        errors.append("Choose a supported detector kind.")
    elif model_kind == "hailo" and model_task not in SUPPORTED_HAILO_TASKS:
        errors.append("Hailo models support detection, segmentation, or pose tasks.")
    elif model_kind != "hailo" and model_task not in (
        "detection", "pose", "classification", "recognition"
    ):
        errors.append("Choose a supported detector task.")
    elif model_kind in ("object_detection", "instance_segmentation") and model_task != "detection":
        errors.append("This ONNX model kind only supports the detection task.")
    elif model_kind == "onnx" and model_task not in ("detection", "pose", "classification"):
        errors.append("Generic ONNX models support detection, pose, or classification.")
    elif model_kind == "pose" and model_task != "pose":
        errors.append("Pose models require the pose task.")
    elif model_kind in ("classification", "resnet") and model_task != "classification":
        errors.append("Classification models require the classification task.")
    elif model_kind in ("onnx", "classification", "resnet", "hailo") and not draft.model_uri.strip():
        errors.append("Enter a model URI or local model path.")
    elif model_kind == "face" and model_task == "recognition" and not draft.model_index.strip():
        errors.append("Face recognition requires an index JSON file.")
    try:
        resolution_valid = len(draft.resolution) == 2 and all(int(value) > 0 for value in draft.resolution)
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
    for name, value, label in (
        ("conf_threshold", draft.conf_threshold, "Confidence threshold"),
        ("iou_threshold", draft.iou_threshold, "IoU threshold"),
    ):
        if value is None:
            continue
        try:
            valid = 0 <= float(value) <= 1
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{label} must be between 0 and 1.")
    for index, stage in enumerate(draft.stages, 1):
        stage_type = stage.get("type") if isinstance(stage, dict) else None
        if isinstance(stage, dict) and stage.get("enabled") is False:
            continue
        if stage_type not in SUPPORTED_STAGE_TYPES:
            errors.append(f"Stage {index}: unsupported stage type.")
            continue
        if stage_type == "line_counter" and len(_points(stage.get("line"))) != 2:
            errors.append(f"Stage {index}: line counter needs two points.")
        if stage_type in ("region_filter", "region_timer"):
            polygon = stage.get("polygon", stage.get("region"))
            if len(_points(polygon)) < 3:
                errors.append(f"Stage {index}: region needs at least three points.")
    return errors


__all__ = [
    "PipelineDraft",
    "SUPPORTED_MODEL_KINDS",
    "SUPPORTED_MODEL_DEFAULT_URIS",
    "SUPPORTED_MODEL_OPTIONS",
    "SUPPORTED_MODEL_TASKS",
    "SUPPORTED_STAGE_TYPES",
    "default_stage",
    "infer_source_type",
    "model_control_visibility",
    "model_defaults",
    "parse_pipeline_draft",
    "serialize_pipeline_draft",
    "validate_pipeline_draft",
]
