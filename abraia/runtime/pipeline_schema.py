"""Shared limits and step definitions for pipeline configuration."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from abraia.inference.model_config import ModelSpec

PIPELINE_SECOND_STEP_MODEL_PAIRS = frozenset({
    ("resnet", "classification"),
    ("face", "recognition"),
    ("license_plate", "recognition"),
    ("ocr", "recognition"),
    ("multispectral", "classification"),
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
                ) not in PIPELINE_SECOND_STEP_MODEL_PAIRS:
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
