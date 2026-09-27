"""Validation rules for editable pipeline drafts."""

from abraia.sources import infer_source_type, normalize_source_type
from abraia.inference.model_config import ModelSpec, PIPELINE_SECOND_STAGE_MODEL_PAIRS

from .pipeline_schema import (
    COMPOSITION_STAGE_TYPES,
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    SUPPORTED_STAGE_TYPES,
)


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
        fallback=normalize_source_type(draft.source_type) or "camera",
    ) or "camera"
    if not source_value:
        errors.append("Choose a source file, camera, or stream.")
    elif source_type == "camera" and not source_value.isdigit():
        errors.append("Camera source must be a numeric camera index.")
    model_spec = ModelSpec.from_config({
        "kind": draft.model_kind,
        "task": draft.model_task,
        "uri": draft.model_uri,
        "params": {"index": draft.model_index},
    })
    errors.extend(model_spec.pipeline_errors())
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
    canonical_stages = bool(
        draft.stages
        and isinstance(draft.stages[0], dict)
        and draft.stages[0].get("type") == "model"
        and draft.stages[0].get("input", "frame") == "frame"
    )
    has_composition = canonical_stages or any(
        isinstance(stage, dict) and stage.get("type") in COMPOSITION_STAGE_TYPES
        for stage in draft.stages
    )
    known_outputs = {"frame", "results"}
    known_ids = set()
    model_count = 0
    tracker_count = 0
    if not canonical_stages:
        known_outputs.add("model.results")
        known_ids.add("model")
    for index, stage in enumerate(draft.stages, 1):
        stage_type = stage.get("type") if isinstance(stage, dict) else None
        if (
            isinstance(stage, dict)
            and stage.get("enabled") is False
            and stage.get("type") != "tracker"
        ):
            continue
        if stage_type not in SUPPORTED_STAGE_TYPES:
            errors.append(f"Stage {index}: unsupported stage type.")
            continue
        if stage_type == "model":
            model_count += 1
            if model_count > MAX_PIPELINE_MODELS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_MODELS} models."
                )
        elif stage_type == "tracker":
            tracker_count += 1
            if tracker_count > MAX_PIPELINE_TRACKERS:
                errors.append(
                    f"A pipeline can contain at most {MAX_PIPELINE_TRACKERS} tracker."
                )
        if stage_type in COMPOSITION_STAGE_TYPES:
            step_id = str(stage.get("id", "")).strip()
            if has_composition and not step_id:
                errors.append(f"Stage {index}: composed steps require a step ID.")
            elif step_id in known_ids:
                errors.append(f"Stage {index}: duplicate step ID '{step_id}'.")
            elif step_id:
                known_ids.add(step_id)
            raw_input = stage.get("input", "")
            input_options = raw_input if isinstance(raw_input, dict) else {}
            input_ref = str(
                input_options.get("from", "") if input_options else raw_input
            ).strip()
            if input_ref and input_ref not in known_outputs:
                errors.append(f"Stage {index}: unknown input reference '{input_ref}'.")
            if stage_type == "model":
                model = stage.get("model")
                if not isinstance(model, dict):
                    errors.append(f"Stage {index}: model step requires a model object.")
                else:
                    legacy_fields = {"crop", "output"}
                    if model_count > 1:
                        legacy_fields.add("input")
                    configured_legacy_fields = sorted(
                        legacy_fields.intersection(stage)
                    )
                    if configured_legacy_fields:
                        errors.append(
                            f"Stage {index}: removed explicit model field(s): "
                            f"{', '.join(configured_legacy_fields)}."
                        )
                    model_spec = ModelSpec.from_config(model)
                    errors.extend(model_spec.pipeline_errors())
                    if model_count > 1 and (
                        model_spec.kind,
                        model_spec.task,
                    ) not in PIPELINE_SECOND_STAGE_MODEL_PAIRS:
                        errors.append(
                            f"Stage {index}: model '{model_spec.kind}' with task "
                            f"'{model_spec.task}' cannot run as a second-stage model."
                        )
            elif not str(stage.get("input", "")).strip():
                errors.append(f"Stage {index}: {stage_type} step requires an input reference.")
            if stage_type == "attach" and not str(stage.get("target", "")).strip():
                errors.append(f"Stage {index}: attach step requires a target reference.")
            if stage_type == "attach":
                target_ref = str(stage.get("target", "")).strip()
                if target_ref and target_ref not in known_outputs:
                    errors.append(f"Stage {index}: unknown target reference '{target_ref}'.")
                output = "results"
            elif stage_type == "crop":
                output = "items"
            else:
                output = "results"
            if step_id:
                known_outputs.add(f"{step_id}.{output}")
            continue
        if stage_type == "line_counter" and len(_points(stage.get("line"))) != 2:
            errors.append(f"Stage {index}: line counter needs two points.")
        if stage_type in ("region_filter", "region_timer"):
            polygon = stage.get("polygon")
            if len(_points(polygon)) < 3:
                errors.append(f"Stage {index}: region needs at least three points.")
    return errors

