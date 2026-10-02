"""Shared structural rules for version-two pipeline steps."""

from abraia.inference.model_config import (
    PIPELINE_SECOND_STAGE_MODEL_PAIRS,
    ModelSpec,
)

from .pipeline_schema import (
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    PipelineStep,
    iter_pipeline_steps,
    normalize_points,
    TRACKER_DEPENDENT_STAGE_TYPES,
    SUPPORTED_STAGE_TYPES,
)


def is_valid_stage_geometry(stage_type, stage):
    """Return whether a geometry stage contains valid coordinates."""
    stage = stage if isinstance(stage, PipelineStep) else PipelineStep.from_config(stage)
    if stage_type == "line_counter":
        return len(normalize_points(stage.get("line"))) == 2
    if stage_type in ("region_filter", "region_timer"):
        return len(normalize_points(stage.get("polygon"))) >= 3
    return True


def stages_missing_tracker(steps):
    """Return ordered stages that need an earlier tracker step."""
    missing = []
    tracker_seen = False
    for index, stage in enumerate(iter_pipeline_steps(steps), 1):
        if not stage.is_mapping:
            continue
        stage_type = stage.type
        if not stage.is_active:
            continue
        if stage_type == "tracker":
            tracker_seen = True
        elif stage_type in TRACKER_DEPENDENT_STAGE_TYPES and not tracker_seen:
            missing.append((index, stage_type))
    return missing


def validate_step_structure(steps, *, report_unsupported=True):
    """Return errors shared by draft validation and runtime construction."""
    errors = []
    known_ids = set()
    model_count = 0
    tracker_count = 0
    model_seen = False

    for index, stage in enumerate(iter_pipeline_steps(steps), 1):
        stage_type = stage.type
        if not stage.is_mapping:
            errors.append(f"Stage {index}: step must be an object.")
            continue
        if not stage.is_active:
            continue
        if stage_type not in SUPPORTED_STAGE_TYPES:
            if report_unsupported:
                errors.append(f"Stage {index}: unsupported stage type.")
            continue

        if stage_type != "model" and not model_seen:
            errors.append(
                f"Stage {index}: {stage_type} requires an earlier model step."
            )

        step_id = stage.id
        if step_id in known_ids:
            errors.append(f"Stage {index}: duplicate step ID '{step_id}'.")
        else:
            known_ids.add(step_id)

        if stage_type == "model":
            model_seen = True
            model_count += 1
            if model_count > MAX_PIPELINE_MODELS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_MODELS} models."
                )
            model = stage.get("model")
            if not isinstance(model, dict):
                errors.append(f"Stage {index}: model step requires a model object.")
            elif model_count > 1:
                model_spec = ModelSpec.from_config(model)
                if (
                    model_spec.kind,
                    model_spec.task,
                ) not in PIPELINE_SECOND_STAGE_MODEL_PAIRS:
                    errors.append(
                        f"Stage {index}: model '{model_spec.kind}' with task "
                        f"'{model_spec.task}' cannot run as a second-stage model."
                    )
        elif stage_type == "tracker":
            tracker_count += 1
            if tracker_count > MAX_PIPELINE_TRACKERS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_TRACKERS} tracker."
                )
    return errors


__all__ = [
    "is_valid_stage_geometry",
    "stages_missing_tracker",
    "validate_step_structure",
]
