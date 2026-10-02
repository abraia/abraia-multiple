"""Shared limits and stage definitions for pipeline configuration."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

PIPELINE_STAGE_DEFINITIONS = {
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
            "line": [[0, 0], [100, 100]],
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

SUPPORTED_STAGE_TYPES = tuple(PIPELINE_STAGE_DEFINITIONS)
TRACKER_DEPENDENT_STAGE_TYPES = frozenset(
    stage_type
    for stage_type, definition in PIPELINE_STAGE_DEFINITIONS.items()
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
        stage_type = config.get("type")
        step_id = str(config.get("id") or "").strip()
        return cls(
            config,
            str(stage_type).strip() if stage_type is not None else None,
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


def normalize_step_ids(steps):
    """Copy ordered steps and add internal IDs where they are absent."""
    normalized = []
    for index, step in enumerate(steps or (), 1):
        if not isinstance(step, Mapping):
            normalized.append(step)
            continue
        item = dict(step)
        item["id"] = str(item.get("id") or default_step_id(index)).strip()
        normalized.append(item)
    return normalized


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
    for step in normalize_step_ids(steps):
        yield PipelineStep.from_config(step)


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


MAX_PIPELINE_TRACKERS = 1
MAX_PIPELINE_MODELS = 2
