"""Construction of ordered, multi-model runtime pipelines."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .composition import (
    AttachStep,
    BoundModelStep,
    CropStep,
    FilterStep,
    LineCounterStep,
    ModelStep,
    RegionFilterStep,
    RegionTimerStep,
    TrackerStep,
)
from .config import (
    MAX_PIPELINE_MODELS,
    MAX_PIPELINE_TRACKERS,
    ModelSpec,
    PIPELINE_SECOND_STAGE_MODEL_PAIRS,
)
from .factories import (
    SourceFactory,
    build_pipeline_output,
    _is_spectral_source,
)
from .lifecycle import close_resources


def build_composed_pipeline(
    pipeline_cls,
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
    on_frame: Optional[Callable] = None,
    accelerator: Optional[str] = "auto",
):
    """Build the version-two ordered multi-model pipeline."""
    source_config = config.get("source") or {}
    display_config = config.get("display") or {}
    steps_config = config.get("steps", []) or []
    if not isinstance(source_config, dict) or "src" not in source_config:
        raise ValueError("Pipeline source must define 'src'")
    if not isinstance(display_config, dict):
        raise ValueError("Pipeline display must be an object")
    if not isinstance(steps_config, list) or not steps_config:
        raise ValueError("A composed pipeline must define a non-empty 'steps' array")

    from ..inference.registry import create_model, get_model_run_kwargs
    from ..inference.session import get_model_accelerator
    from ..inference import Tracker
    from .stages import LineCounter, RegionFilter, RegionTimer

    root = Path(base_dir or os.getcwd())
    source_plan = SourceFactory.prepare(source_config, display_config, root)
    spectral_source = _is_spectral_source(source_plan.source) or any(
        isinstance(step, dict)
        and ModelSpec.from_config(step.get("model") or {}).kind == "multispectral"
        for step in steps_config
        if isinstance(step, dict) and step.get("type") == "model"
    )
    video = None
    steps = []
    models = []
    components = {}
    ids = set()
    outputs = {"frame", "results"}
    model_count = 0
    tracker_count = 0
    previous_results_ref = "results"
    try:
        video = SourceFactory.create(source_plan, spectral=spectral_source)
        for index, step_config in enumerate(steps_config, 1):
            if not isinstance(step_config, dict):
                raise ValueError(f"Pipeline step {index} must be an object")
            step_id = str(step_config.get("id", "")).strip()
            if not step_id:
                raise ValueError(f"Pipeline step {index} must define an 'id'")
            if step_id in ids:
                raise ValueError(f"Duplicate pipeline step id: {step_id}")
            ids.add(step_id)
            if (
                step_config.get("enabled") is False
                and step_config.get("type") != "tracker"
            ):
                continue
            step_type = step_config.get("type")
            if step_type == "model":
                model_config = step_config.get("model") or {}
                if not isinstance(model_config, dict):
                    raise ValueError(
                        f"Model step '{step_id}' must define an object 'model'"
                    )
                model_count += 1
                if model_count > MAX_PIPELINE_MODELS:
                    raise ValueError(
                        f"A pipeline can contain at most {MAX_PIPELINE_MODELS} models"
                    )
                if model_count > 1:
                    model_spec = ModelSpec.from_config(model_config)
                    model_pair = (model_spec.kind, model_spec.task)
                    if model_pair not in PIPELINE_SECOND_STAGE_MODEL_PAIRS:
                        raise ValueError(
                            "Second-stage models must support region input: "
                            f"{model_pair[0]} ({model_pair[1]})"
                        )
            elif step_type == "tracker":
                tracker_count += 1
                if tracker_count > MAX_PIPELINE_TRACKERS:
                    raise ValueError(
                        f"A pipeline can contain at most {MAX_PIPELINE_TRACKERS} tracker"
                    )
            default_input = (
                "frame"
                if step_type == "model" and model_count == 1
                else previous_results_ref
            )
            if step_type == "model" and model_count > 1:
                raw_input = default_input
                input_options = {}
            else:
                raw_input = step_config.get("input", default_input)
                input_options = raw_input if isinstance(raw_input, dict) else {}
            input_ref = str(
                input_options.get("from", default_input)
                if input_options
                else raw_input
            ).strip()
            if index == 1 and (step_type != "model" or input_ref != "frame"):
                raise ValueError(
                    "A composed pipeline must start with a model step consuming 'frame'"
                )
            if input_ref not in outputs:
                raise ValueError(
                    f"Pipeline step '{step_id}' references unknown input '{input_ref}'"
                )
            if step_type == "model":
                legacy_fields = {"crop", "output"}
                if model_count > 1:
                    legacy_fields.add("input")
                configured_legacy_fields = sorted(
                    legacy_fields.intersection(step_config)
                )
                if configured_legacy_fields:
                    fields = ", ".join(configured_legacy_fields)
                    raise ValueError(
                        f"Model step '{step_id}' uses removed explicit field(s): {fields}"
                    )
                model = create_model(
                    model_config,
                    base_dir=root,
                    accelerator=accelerator,
                )
                models.append(model)
                if model_count > 1:
                    default_field = (
                        "ocr" if model_config.get("kind") == "ocr"
                        else "classification"
                        if model_config.get("task") == "classification"
                        else "recognition"
                        if model_config.get("task") == "recognition"
                        else "result"
                    )
                    steps.append(BoundModelStep(
                        step_id,
                        model,
                        input_ref=previous_results_ref,
                        attach_to=previous_results_ref,
                        field=default_field,
                        model_kwargs=get_model_run_kwargs(model_config),
                    ))
                else:
                    steps.append(ModelStep(
                        step_id,
                        model,
                        input_ref=input_ref,
                        model_kwargs=get_model_run_kwargs(model_config),
                    ))
            elif step_type == "filter":
                steps.append(FilterStep(
                    step_id,
                    input_ref,
                    labels=step_config.get("labels"),
                    min_confidence=step_config.get("min_confidence"),
                ))
            elif step_type == "crop":
                steps.append(CropStep(
                    step_id,
                    input_ref,
                    padding=step_config.get("padding", 0.0),
                ))
            elif step_type == "attach":
                target = step_config.get("target")
                if not target:
                    raise ValueError(f"Attach step '{step_id}' requires 'target'")
                if target not in outputs:
                    raise ValueError(
                        f"Attach step '{step_id}' references unknown target '{target}'"
                    )
                steps.append(AttachStep(
                    step_id,
                    input_ref,
                    target,
                    field=step_config.get("field", "result"),
                ))
            elif step_type == "tracker":
                tracker = Tracker(
                    track_thresh=step_config.get("track_thresh", 0.25),
                    track_buffer=step_config.get("track_buffer", 30),
                    match_thresh=step_config.get("match_thresh", 0.8),
                    frame_rate=video.frame_rate,
                )
                steps.append(TrackerStep(step_id, input_ref, tracker))
                components["tracker"] = tracker
            elif step_type == "line_counter":
                line = step_config.get("line")
                if not line or len(line) != 2:
                    raise ValueError("line_counter requires a two-point 'line'")
                counter = LineCounter(line)
                steps.append(LineCounterStep(step_id, input_ref, counter))
                components["line_counter"] = counter
            elif step_type == "region_filter":
                polygon = step_config.get("polygon")
                if not polygon:
                    raise ValueError("region_filter requires a 'polygon'")
                region_filter = RegionFilter(polygon)
                steps.append(RegionFilterStep(step_id, input_ref, region_filter))
                components["region_filter"] = region_filter
            elif step_type == "region_timer":
                polygon = step_config.get("polygon")
                if not polygon:
                    raise ValueError("region_timer requires a 'polygon'")
                region_timer = RegionTimer(polygon)
                steps.append(RegionTimerStep(step_id, input_ref, region_timer))
                components["region_timer"] = region_timer
            else:
                raise ValueError(f"Unknown composed pipeline step type: {step_type}")
            output_name = "items" if step_type == "crop" else "results"
            outputs.add(f"{step_id}.{output_name}")
            if output_name == "results" and step_type in (
                "model",
                "filter",
                "attach",
                "tracker",
                "line_counter",
                "region_filter",
                "region_timer",
            ):
                previous_results_ref = f"{step_id}.results"

        video.accelerator = (
            get_model_accelerator(models[0]) if models else "CPU"
        )
    except Exception:
        close_resources([*steps, *models, video])
        raise

    renderer, display = build_pipeline_output(
        video, source_plan, display_config, components
    )
    return pipeline_cls(
        source=video,
        model=models[0] if models else None,
        steps=steps,
        display=display,
        render=renderer,
        on_frame=on_frame,
        components={},
    )
