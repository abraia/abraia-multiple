"""Shared limits and stage types for pipeline configuration."""

SUPPORTED_STAGE_TYPES = (
    "tracker",
    "line_counter",
    "region_filter",
    "region_timer",
    "filter",
    "crop",
    "model",
    "attach",
)
COMPOSITION_STAGE_TYPES = frozenset({"filter", "crop", "model", "attach"})
MAX_PIPELINE_TRACKERS = 1
MAX_PIPELINE_MODELS = 2

