"""Construction of ordered, multi-model runtime pipelines."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .composition import (
    BoundModelStep,
    CropStep,
    FilterStep,
    LineCounterStep,
    ModelStep,
    RegionFilterStep,
    RegionTimerStep,
    TrackerStep,
)
from .config import ModelSpec
from .factories import (
    SourceFactory,
    build_pipeline_output,
    _is_spectral_source,
)
from .lifecycle import close_resources
from .pipeline_schema import (
    PipelineStep,
    SUPPORTED_STAGE_TYPES,
    normalize_step_ids,
)
from .pipeline_rules import (
    is_valid_stage_geometry,
    stages_missing_tracker,
    validate_step_structure,
)


def build_composed_pipeline(
    pipeline_cls,
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
    on_frame: Optional[Callable] = None,
    accelerator: Optional[str] = "auto",
    renderer_factory: Optional[Callable] = None,
):
    """Build the version-two ordered multi-model pipeline."""
    source_config = config.get("source") or {}
    display_config = config.get("display") or {}
    steps_config = normalize_step_ids(config.get("steps", []) or [])
    if not isinstance(source_config, dict) or "src" not in source_config:
        raise ValueError("Pipeline source must define 'src'")
    if not isinstance(display_config, dict):
        raise ValueError("Pipeline display must be an object")
    if not isinstance(steps_config, list):
        raise ValueError("Pipeline 'steps' must be an array")
    structure_errors = validate_step_structure(
        steps_config,
        report_unsupported=False,
    )
    if structure_errors:
        raise ValueError(structure_errors[0])
    dependency_errors = stages_missing_tracker(steps_config)
    if dependency_errors:
        index, stage_type = dependency_errors[0]
        raise ValueError(
            f"Stage {index}: {stage_type} requires an earlier tracker."
        )

    pipeline_steps = [PipelineStep.from_config(step) for step in steps_config]
    for step in pipeline_steps:
        if not step.is_mapping or not step.is_active:
            continue
        if step.type not in SUPPORTED_STAGE_TYPES:
            raise ValueError(f"Unknown composed pipeline step type: {step.type}")
        if not is_valid_stage_geometry(step.type, step):
            if step.type == "line_counter":
                raise ValueError("line_counter requires a two-point 'line'")
            raise ValueError(f"{step.type} requires a 'polygon'")

    model_plans = []
    for step in pipeline_steps:
        if not step.is_mapping or not step.is_active or step.type != "model":
            continue
        model_config = step.get("model") or {}
        if not isinstance(model_config, dict):
            raise ValueError(
                f"Model step '{step.id}' must define an object 'model'"
            )
        model_spec = ModelSpec.from_config(model_config)
        model_spec.require_runtime_valid()
        model_plans.append((step, model_config, model_spec))

    from ..inference.registry import create_model, get_model_run_kwargs
    from ..inference.session import get_model_accelerator
    from ..inference import Tracker
    from .stages import LineCounter, RegionFilter, RegionTimer

    root = Path(base_dir or os.getcwd())
    source_plan = SourceFactory.prepare(source_config, display_config, root)
    spectral_source = _is_spectral_source(source_plan.source) or any(
        model_spec.kind == "multispectral"
        for _step, _model_config, model_spec in model_plans
    )
    video = None
    steps = []
    models = []
    created_models = []
    components = {}
    try:
        def create_models():
            created = []
            try:
                for _step, model_config, model_spec in model_plans:
                    model_kwargs = get_model_run_kwargs(model_config)
                    result_field = model_spec.result_field
                    model = create_model(
                        model_config,
                        base_dir=root,
                        accelerator=accelerator,
                    )
                    created.append((model, model_kwargs, result_field))
                return created
            except Exception:
                close_resources([model for model, _kwargs, _field in created])
                raise

        # Camera startup and model/session initialization are independent.
        # Starting them together avoids making users wait for both operations
        # serially while keeping each resource owned by this build operation.
        with ThreadPoolExecutor(max_workers=2) as executor:
            source_future = executor.submit(
                SourceFactory.create, source_plan, spectral=spectral_source
            )
            models_future = executor.submit(create_models)
            try:
                video = source_future.result()
                created_models = models_future.result()
                models = [item[0] for item in created_models]
            except Exception:
                try:
                    video = source_future.result()
                except Exception:
                    pass
                try:
                    created_models = models_future.result()
                    models = [item[0] for item in created_models]
                except Exception:
                    pass
                raise

        model_iter = iter(created_models)
        model_index = 0
        for index, step in enumerate(pipeline_steps, 1):
            if not step.is_mapping:
                raise ValueError(f"Pipeline step {index} must be an object")
            step_id = step.id
            if not step.is_active:
                continue
            step_type = step.type
            if step_type == "model":
                model, model_kwargs, result_field = next(model_iter)
                if model_index > 0:
                    steps.append(BoundModelStep(
                        step_id,
                        model,
                        field=result_field,
                        model_kwargs=model_kwargs,
                    ))
                else:
                    steps.append(ModelStep(
                        step_id,
                        model,
                        model_kwargs=model_kwargs,
                    ))
                model_index += 1
            elif step_type == "filter":
                steps.append(FilterStep(
                    step_id,
                    labels=step.get("labels"),
                    min_confidence=step.get("min_confidence"),
                ))
            elif step_type == "crop":
                steps.append(CropStep(
                    step_id,
                    padding=step.get("padding", 0.0),
                ))
            elif step_type == "tracker":
                tracker = Tracker(
                    track_thresh=step.get("track_thresh", 0.25),
                    track_buffer=step.get("track_buffer", 30),
                    match_thresh=step.get("match_thresh", 0.8),
                    frame_rate=video.frame_rate,
                )
                steps.append(TrackerStep(step_id, tracker))
                components["tracker"] = tracker
            elif step_type == "line_counter":
                line = step.get("line")
                counter = LineCounter(line)
                steps.append(LineCounterStep(step_id, counter))
                components["line_counter"] = counter
            elif step_type == "region_filter":
                polygon = step.get("polygon")
                region_filter = RegionFilter(polygon)
                steps.append(RegionFilterStep(step_id, region_filter))
                components["region_filter"] = region_filter
            elif step_type == "region_timer":
                polygon = step.get("polygon")
                region_timer = RegionTimer(polygon)
                steps.append(RegionTimerStep(step_id, region_timer))
                components["region_timer"] = region_timer
            else:
                raise ValueError(f"Unknown composed pipeline step type: {step_type}")
        video.accelerator = (
            get_model_accelerator(models[0]) if models else "CPU"
        )
        renderer, display = build_pipeline_output(
            video, source_plan, display_config, components
        )
        if renderer_factory is not None:
            renderer = renderer_factory(renderer, config)
        return pipeline_cls(
            source=video,
            model=models[0] if models else None,
            steps=steps,
            display=display,
            render=renderer,
            on_frame=on_frame,
            components=components,
        )
    except Exception:
        wrapped_model_ids = {
            id(step.model)
            for step in steps
            if isinstance(step, ModelStep)
        }
        unwrapped_models = [
            model for model in models if id(model) not in wrapped_model_ids
        ]
        close_resources([*steps, *unwrapped_models, video])
        raise
