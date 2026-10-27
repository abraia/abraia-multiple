"""Versioned pipeline configuration and validation for runtime clients."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from abraia.inference.model_config import (
    DEFAULT_MODEL_URIS as SUPPORTED_MODEL_DEFAULT_URIS,
    MODEL_ARCHITECTURES,
    MODEL_DESCRIPTORS,
    MODEL_SIZE_URIS,
    ModelSpec,
    PIPELINE_MODEL_KINDS as SUPPORTED_MODEL_KINDS,
    PIPELINE_MODEL_TASKS as SUPPORTED_MODEL_TASKS,
    normalize_task,
)

from .sources import infer_source_type, normalize_source_type


PIPELINE_SECOND_STEP_MODEL_PAIRS = frozenset({
    ("resnet", "classification"),
    ("face", "recognition"),
    ("license_plate", "recognition"),
    ("ocr", "recognition"),
})

PIPELINE_STEP_DEFINITIONS = {
    "tracker": {
        "category": "geometry",
        "default": {
            "type": "tracker",
            "track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.8,
        },
    },
    "line_counter": {
        "category": "geometry",
        "requires_tracker": True,
        "default": {
            "type": "line_counter",
            "line": [[960, 0], [960, 720]],
        },
    },
    "region_filter": {
        "category": "geometry",
        "default": {
            "type": "region_filter",
            "polygon": [[0, 0], [100, 0], [100, 100]],
        },
    },
    "region_timer": {
        "category": "geometry",
        "requires_tracker": True,
        "default": {
            "type": "region_timer",
            "polygon": [[0, 0], [100, 0], [100, 100]],
        },
    },
    "filter": {
        "category": "fields",
        "default": {
            "type": "filter",
            "labels": [],
            "min_confidence": 0.0,
        },
    },
    "crop": {
        "category": "fields",
        "default": {
            "type": "crop",
            "padding": 0.0,
        },
    },
    "model": {
        "category": "model",
        "default": {
            "type": "model",
            "model": {"kind": "ocr", "task": "recognition"},
        },
    },
}

SUPPORTED_STEP_TYPES = tuple(PIPELINE_STEP_DEFINITIONS)
TRACKER_DEPENDENT_STEP_TYPES = frozenset(
    step_type
    for step_type, definition in PIPELINE_STEP_DEFINITIONS.items()
    if definition.get("requires_tracker")
)


@dataclass(frozen=True)
class PipelineStep:
    """Typed view of one pipeline step while retaining its JSON mapping."""

    raw: Optional[Mapping[str, Any]]
    type: Optional[str]
    id: str
    enabled: bool

    @classmethod
    def from_config(cls, config):
        """Build a typed step view without mutating the source mapping."""
        if not isinstance(config, Mapping):
            return cls(None, None, "", True)
        step_type = config.get("type")
        step_id = str(config.get("id") or "").strip()
        return cls(
            config,
            str(step_type).strip() if step_type is not None else None,
            step_id,
            config.get("enabled") is not False,
        )

    @property
    def is_mapping(self):
        """Return whether the source value was a JSON object."""
        return self.raw is not None

    @property
    def is_active(self):
        """Return whether the step participates in the composed pipeline."""
        return self.enabled or self.type == "tracker"

    def get(self, name, default=None):
        """Read a value from the original step mapping."""
        if self.raw is None:
            return default
        return self.raw.get(name, default)


def default_step_id(index):
    """Return the generated internal ID for an ordered step."""
    return "model" if index == 1 else f"step-{index - 1}"


def _normalize_step_id(step, index):
    if not isinstance(step, Mapping):
        return step
    item = dict(step)
    item["id"] = str(item.get("id") or default_step_id(index)).strip()
    return item


def normalize_step_ids(steps):
    """Copy ordered steps and add internal IDs where they are absent."""
    return [
        _normalize_step_id(step, index)
        for index, step in enumerate(steps or (), 1)
    ]


def strip_internal_step_ids(steps):
    """Copy ordered steps while removing runtime-only IDs from public JSON."""
    public_steps = []
    for step in steps or ():
        if not isinstance(step, Mapping):
            public_steps.append(step)
            continue
        public_step = dict(step)
        public_step.pop("id", None)
        public_steps.append(public_step)
    return public_steps


def iter_pipeline_steps(steps):
    """Yield typed, internally normalized views of ordered pipeline steps."""
    for index, step in enumerate(steps or (), 1):
        if isinstance(step, PipelineStep):
            yield step
        else:
            yield PipelineStep.from_config(_normalize_step_id(step, index))


def normalize_points(value):
    """Normalize text or coordinate pairs into JSON-compatible points."""
    if isinstance(value, str):
        raw_points = value.split(";")
        points = []
        for raw_point in raw_points:
            values = [item.strip() for item in raw_point.split(",")]
            if len(values) != 2:
                continue
            try:
                points.append([float(values[0]), float(values[1])])
            except (TypeError, ValueError):
                continue
        return points
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


def is_primary_model_step(step):
    """Return whether a step is the frame-consuming model step."""
    return (
        isinstance(step, dict)
        and step.get("type") == "model"
    )


def starts_with_model_step(steps):
    """Return whether an ordered pipeline is empty or starts with a model."""
    return not steps or is_primary_model_step(steps[0])


MAX_PIPELINE_TRACKERS = 1
MAX_PIPELINE_MODELS = 2


def is_valid_step_geometry(step_type, step):
    """Return whether a geometry step contains valid coordinates."""
    step = step if isinstance(step, PipelineStep) else PipelineStep.from_config(step)
    if step_type == "line_counter":
        return len(normalize_points(step.get("line"))) == 2
    if step_type in ("region_filter", "region_timer"):
        return len(normalize_points(step.get("polygon"))) >= 3
    return True


def steps_missing_tracker(steps):
    """Return ordered steps that need an earlier tracker step."""
    missing = []
    tracker_seen = False
    for index, step in enumerate(iter_pipeline_steps(steps), 1):
        if not step.is_mapping or not step.is_active:
            continue
        if step.type == "tracker":
            tracker_seen = True
        elif step.type in TRACKER_DEPENDENT_STEP_TYPES and not tracker_seen:
            missing.append((index, step.type))
    return missing


def validate_step_structure(steps, *, report_unsupported=True):
    """Return structural errors shared by draft and runtime validation."""
    errors = []
    known_ids = set()
    model_count = 0
    tracker_count = 0
    model_seen = False

    for index, step in enumerate(iter_pipeline_steps(steps), 1):
        step_type = step.type
        if not step.is_mapping:
            errors.append(f"Step {index}: step must be an object.")
            continue
        if not step.is_active:
            continue
        if step_type not in SUPPORTED_STEP_TYPES:
            if report_unsupported:
                errors.append(f"Step {index}: unsupported step type.")
            continue

        if step_type != "model" and not model_seen:
            errors.append(
                f"Step {index}: {step_type} requires an earlier model step."
            )

        if step.id in known_ids:
            errors.append(f"Step {index}: duplicate step ID '{step.id}'.")
        else:
            known_ids.add(step.id)

        if step_type == "model":
            model_seen = True
            model_count += 1
            if model_count > MAX_PIPELINE_MODELS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_MODELS} models."
                )
            model = step.get("model")
            if not isinstance(model, dict):
                errors.append(f"Step {index}: model step requires a model object.")
            elif model_count > 1:
                model_spec = ModelSpec.from_config(model)
                if (
                    model_spec.kind,
                    model_spec.task,
                ) not in PIPELINE_SECOND_STEP_MODEL_PAIRS and not (
                    model_spec.descriptor and model_spec.descriptor.follow_up
                ):
                    errors.append(
                        f"Step {index}: model '{model_spec.kind}' with task "
                        f"'{model_spec.task}' cannot run as a second-step model."
                    )
        elif step_type == "tracker":
            tracker_count += 1
            if tracker_count > MAX_PIPELINE_TRACKERS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_TRACKERS} tracker."
                )
    return errors


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
)


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
        "iou": kind not in ("ocr", "resnet")
        and not (kind == "face" and task == "recognition"),
        "approx": kind in MODEL_ARCHITECTURES and task == "segmentation",
    }


def model_pipeline_errors(model):
    spec = model if isinstance(model, ModelSpec) else ModelSpec.from_config(model)
    error = spec.validation_error()
    if error:
        field, message = error
        if field == "kind":
            message = "Choose a supported detector kind."
        elif field == "task" and spec.task not in SUPPORTED_MODEL_TASKS:
            message = "Choose a supported detector task."
        elif field == "uri":
            message = "Enter a model URI or local model path."
        return (message,)
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

    Runtime component validation belongs to :class:`abraia.runtime.Pipeline`.
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
    source_options: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.resolution is None:
            self.resolution = [1920, 1080]
        self.steps = (
            [default_primary_step()] if self.steps is None else list(self.steps)
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PipelineDraft":
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
        return cls(
            source_type=source_type or "camera",
            source=source_value,
            source_options=source.get("options", {}),
            resolution=[_number(v, 0, int) for v in resolution],
            fps=_number(source.get("fps", 30), 30, int),
            steps=normalized_steps,
            show=bool(display.get("show", True)),
            render_results=bool(display.get("render_results", True)),
            render_metrics=bool(display.get("render_metrics", True)),
            output=deepcopy(config.get("output")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a pipeline draft as a version-two document."""
        source_value = "" if self.source is None else str(self.source).strip()
        source_type = normalize_source_type(self.source_type) or "camera"
        if source_type != "genicam":
            source_type = infer_source_type(source_value, fallback=source_type)
        source = {
            "type": source_type,
            "src": source_value,
            "resolution": [int(v) for v in (self.resolution or [1920, 1080])[:2]],
            "fps": int(self.fps),
        }
        if not isinstance(self.source_options, dict):
            raise ValueError("Pipeline source options must be an object.")
        if self.source_options:
            source["options"] = dict(self.source_options)
        steps = [dict(step) for step in self.steps]
        display = {
            "show": bool(self.show),
            "render_results": bool(self.render_results),
            "render_metrics": bool(self.render_metrics),
        }
        if any(
            step.get("type") not in SUPPORTED_STEP_TYPES
            for step in steps
            if isinstance(step, dict)
        ):
            raise ValueError("Pipeline contains an unsupported step type.")

        if not starts_with_model_step(steps):
            raise ValueError(
                "A version 2 pipeline with steps must start with a model step."
            )
        pipeline_steps = strip_internal_step_ids(steps)

        config = {
            "version": 2,
            "source": source,
            "steps": pipeline_steps,
            "display": display,
        }
        if self.output is not None:
            if not isinstance(self.output, dict):
                raise ValueError("Pipeline output must be an object.")
            if self.output.get("type") != "rtsp":
                raise ValueError("Output type must be 'rtsp'.")
            if not str(self.output.get("url", "")).startswith("rtsp://"):
                raise ValueError("RTSP output must have an RTSP URL.")
            config["output"] = deepcopy(self.output)
        return config

    def validation_errors(self) -> List[str]:
        """Return user-facing validation messages for this pipeline draft."""
        errors = []
        source_value = "" if self.source is None else str(self.source).strip()
        source_type = normalize_source_type(self.source_type) or "camera"
        if source_type != "genicam":
            source_type = infer_source_type(source_value, fallback=source_type)
        if not source_value:
            errors.append("Choose a source file, camera, stream, or GenICam device.")
        elif source_type == "camera" and infer_source_type(source_value) != "camera":
            errors.append("Camera source must be a numeric camera index.")
        if not isinstance(self.source_options, dict):
            errors.append("Source options must be an object.")
        if self.output is not None:
            if not isinstance(self.output, dict):
                errors.append("Pipeline output must be an object.")
            elif self.output.get("type") != "rtsp":
                errors.append("Output type must be 'rtsp'.")
            elif not str(self.output.get("url", "")).startswith("rtsp://"):
                errors.append("RTSP output must have an RTSP URL.")
            else:
                try:
                    if int(self.output.get("bitrate", 4_000_000)) <= 0:
                        errors.append("RTSP output bitrate must be greater than zero.")
                except (TypeError, ValueError):
                    errors.append("RTSP output bitrate must be a positive integer.")
        try:
            resolution_valid = len(self.resolution) == 2 and all(
                int(value) > 0 for value in self.resolution
            )
        except (TypeError, ValueError):
            resolution_valid = False
        if not resolution_valid:
            errors.append("Resolution must contain two positive values.")
        try:
            fps_valid = int(self.fps) > 0
        except (TypeError, ValueError):
            fps_valid = False
        if not fps_valid:
            errors.append("FPS must be greater than zero.")
        steps = self.steps
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


__all__ = [
    "PIPELINE_SECOND_STEP_MODEL_PAIRS",
    "PIPELINE_STEP_DEFINITIONS",
    "TRACKER_DEPENDENT_STEP_TYPES",
    "PipelineStep",
    "default_step_id",
    "normalize_step_ids",
    "strip_internal_step_ids",
    "iter_pipeline_steps",
    "normalize_points",
    "is_primary_model_step",
    "starts_with_model_step",
    "is_valid_step_geometry",
    "steps_missing_tracker",
    "validate_step_structure",
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
