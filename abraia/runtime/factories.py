"""Factories used to assemble runtime inference pipelines."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from abraia.inference.model_config import ModelSpec
from abraia.sources import SPECTRAL_SUFFIXES, infer_source_type

from .composition import (
    BoundModelStep,
    CropStep,
    FilterStep,
    LineCounter,
    ModelStep,
    ProcessorStep,
    RegionTimer,
    filter_region,
)
from .lifecycle import close_resources
from .pipeline_schema import (
    SUPPORTED_STEP_TYPES,
    iter_pipeline_steps,
    is_valid_step_geometry,
    steps_missing_tracker,
    validate_step_structure,
)


def _is_spectral_source(source: Any) -> bool:
    """Return whether a source path identifies a spectral capture."""
    if not isinstance(source, str):
        return False
    return source.lower().endswith(SPECTRAL_SUFFIXES)


_SPECTRAL_SOURCE_FACTORY = None


def register_spectral_source(factory):
    """Register the application supplied source for spectral data."""
    if not callable(factory):
        raise TypeError("Spectral source factory must be callable")
    global _SPECTRAL_SOURCE_FACTORY
    _SPECTRAL_SOURCE_FACTORY = factory


def _prepare_source(source_config, display_config, root: Path):
    """Normalize source configuration into a source and constructor kwargs."""
    source = source_config["src"]
    source_type = source_config.get("type")
    numeric_source = infer_source_type(source) == "camera"
    camera_source = (
        source_type in ("camera", "usb_camera")
        or isinstance(source, int)
        or (numeric_source and source_type != "image")
    )
    if camera_source and isinstance(source, str) and numeric_source:
        source = int(source.strip())
    if (
        isinstance(source, str)
        and not camera_source
        and infer_source_type(source) != "stream"
    ):
        source_path = Path(source)
        if not source_path.is_absolute():
            source = str(root / source_path)

    resolution = source_config.get("resolution", (1920, 1080))
    if isinstance(resolution, list):
        resolution = tuple(resolution)

    destination = display_config.get("dest")
    if isinstance(destination, str) and not os.path.isabs(destination):
        destination = str(root / destination)

    video_kwargs = {
        "resolution": resolution,
        "fps": source_config.get("fps", 30),
        "dest": destination,
        "source_type": source_type,
    }
    if "video_unpaced" in source_config:
        video_kwargs["video_unpaced"] = source_config["video_unpaced"]
    return source, video_kwargs


def _create_source(source, video_kwargs, spectral=False):
    """Create a standard or application supplied spectral frame source."""
    if spectral:
        if _SPECTRAL_SOURCE_FACTORY is None:
            raise ValueError("Spectral pipeline support has not been registered")
        return _SPECTRAL_SOURCE_FACTORY(source, **video_kwargs)
    from .video import Video

    return Video(source, **video_kwargs)


def _make_pipeline_renderer(components, render_results=True, render_metrics=True):
    """Create a frame renderer from pipeline components and display options."""
    def render(context):
        out = context.frame.copy()
        if render_metrics:
            counter = components.get("line_counter")
            if counter:
                from ..utils.draw import render_counter

                out = render_counter(
                    out,
                    counter.line,
                    f"In: {counter.in_count} | Out: {counter.out_count}",
                )
            region_timer = components.get("region_timer")
            if region_timer:
                from ..utils.draw import render_region

                out = render_region(
                    out,
                    region_timer.region,
                    f"Count: {context.metrics['region']['count']}",
                    color=(255, 255, 0),
                )
        if render_results:
            from ..utils.draw import render_results as draw_results

            out = draw_results(out, context.results)
        return out

    return render


def build_pipeline_output(video, video_kwargs, display_config, components):
    """Create shared rendering and display configuration for a pipeline."""
    renderer = _make_pipeline_renderer(
        components,
        render_results=display_config.get("render_results", True),
        render_metrics=display_config.get("render_metrics", True),
    )
    show_display = bool(display_config.get("show", True))
    if not show_display:
        video.set_display_enabled(False)
    display = video if show_display or video_kwargs.get("dest") else None
    return renderer, display


def _build_pipeline(
    pipeline_cls,
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
    on_frame: Optional[Callable] = None,
    accelerator: Optional[str] = "auto",
    renderer_factory: Optional[Callable] = None,
):
    """Assemble a pipeline from a supported versioned configuration."""
    if not isinstance(config, dict):
        raise ValueError("Pipeline configuration must be a JSON object")
    if config.get("version") != 2:
        raise ValueError("Only version 2 pipeline configurations are supported")
    source_config = config.get("source") or {}
    display_config = config.get("display") or {}
    raw_steps = config.get("steps", []) or []
    if not isinstance(source_config, dict) or "src" not in source_config:
        raise ValueError("Pipeline source must define 'src'")
    if not isinstance(display_config, dict):
        raise ValueError("Pipeline display must be an object")
    if not isinstance(raw_steps, list):
        raise ValueError("Pipeline 'steps' must be an array")
    steps_config = list(iter_pipeline_steps(raw_steps))
    structure_errors = validate_step_structure(steps_config, report_unsupported=False)
    if structure_errors:
        raise ValueError(structure_errors[0])
    dependency_errors = steps_missing_tracker(steps_config)
    if dependency_errors:
        index, step_type = dependency_errors[0]
        raise ValueError(
            f"Step {index}: {step_type} requires an earlier tracker."
        )

    for step in steps_config:
        if not step.is_active:
            continue
        if step.type not in SUPPORTED_STEP_TYPES:
            raise ValueError(f"Unknown composed pipeline step type: {step.type}")
        if not is_valid_step_geometry(step.type, step):
            if step.type == "line_counter":
                raise ValueError("line_counter requires a two-point 'line'")
            raise ValueError(f"{step.type} requires a 'polygon'")

    model_plans = []
    for step in steps_config:
        if not step.is_active or step.type != "model":
            continue
        model_config = step.get("model")
        model_spec = ModelSpec.from_config(model_config)
        model_spec.require_runtime_valid()
        model_plans.append((step, model_config, model_spec))

    from ..inference.registry import _create_model_from_spec, get_model_run_kwargs
    from ..inference.session import get_model_accelerator
    from ..inference.tracker import Tracker

    root = Path(base_dir or os.getcwd())
    source, video_kwargs = _prepare_source(source_config, display_config, root)
    spectral_source = _is_spectral_source(source) or any(
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
                    model_kwargs = get_model_run_kwargs(model_config, model_spec)
                    result_field = model_spec.result_field
                    model = _create_model_from_spec(
                        model_spec,
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
                _create_source, source, video_kwargs, spectral=spectral_source
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
        for step in steps_config:
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
                steps.append(ProcessorStep(step_id, tracker, step_type))
                components["tracker"] = tracker
            elif step_type == "line_counter":
                counter = LineCounter(step.get("line"))
                steps.append(ProcessorStep(step_id, counter, step_type))
                components["line_counter"] = counter
            elif step_type == "region_filter":
                region_filter = partial(
                    filter_region, polygon=step.get("polygon")
                )
                steps.append(ProcessorStep(step_id, region_filter, step_type))
                components["region_filter"] = region_filter
            elif step_type == "region_timer":
                region_timer = RegionTimer(step.get("polygon"))
                steps.append(ProcessorStep(step_id, region_timer, step_type))
                components["region_timer"] = region_timer
        video.accelerator = (
            get_model_accelerator(models[0]) if models else "CPU"
        )
        renderer, display = build_pipeline_output(
            video, video_kwargs, display_config, components
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
            if getattr(step, "model", None) is not None
        }
        unwrapped_models = [
            model for model in models if id(model) not in wrapped_model_ids
        ]
        close_resources([*steps, *unwrapped_models, video])
        raise


__all__ = [
    "register_spectral_source",
]
