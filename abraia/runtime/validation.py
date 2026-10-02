"""Validation rules for editable pipeline drafts."""

from abraia.sources import infer_source_type, normalize_source_type
from abraia.inference.model_config import ModelSpec

from .pipeline_schema import (
    PipelineStep,
    iter_pipeline_steps,
    is_primary_model_step,
)
from .pipeline_rules import is_valid_stage_geometry, validate_step_structure


def validate_pipeline_draft(draft):
    """Return user-facing validation messages for a pipeline draft."""
    errors = []
    source_value = "" if draft.source is None else str(draft.source).strip()
    source_type = infer_source_type(
        source_value,
        fallback=normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    if not source_value:
        errors.append("Choose a source file, camera, or stream.")
    elif source_type == "camera" and not source_value.isdigit():
        errors.append("Camera source must be a numeric camera index.")
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
    steps = draft.steps
    canonical_steps = is_primary_model_step(steps[0]) if steps else True
    if not canonical_steps:
        errors.append(
            "A version 2 pipeline with stages must start with a frame model step."
        )
    errors.extend(validate_step_structure(steps))
    for index, stage in enumerate(iter_pipeline_steps(steps), 1):
        if not stage.is_mapping:
            continue
        stage_type = stage.type
        if not stage.is_active:
            continue
        if stage_type == "model" and isinstance(stage.get("model"), dict):
            model = stage.get("model")
            model_spec = ModelSpec.from_config(model)
            errors.extend(model_spec.pipeline_errors())
            editor_values = model_spec.editor_values(model)
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
        if stage_type == "line_counter" and not is_valid_stage_geometry(
            stage_type, stage
        ):
            errors.append(f"Stage {index}: line counter needs two points.")
        if stage_type in ("region_filter", "region_timer") and not is_valid_stage_geometry(
            stage_type, stage
        ):
            errors.append(f"Stage {index}: region needs at least three points.")
    return errors
