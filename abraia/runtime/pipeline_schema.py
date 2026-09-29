"""Shared limits and supported step types for pipeline configuration."""

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
# Retained as a public compatibility alias; implementation uses the canonical
# SUPPORTED_STAGE_TYPES tuple directly.
COMPOSITION_STEP_TYPES = frozenset(SUPPORTED_STAGE_TYPES)
MAX_PIPELINE_TRACKERS = 1
MAX_PIPELINE_MODELS = 2
