"""Callback-oriented pipeline execution helpers."""

from collections import deque
import time

from .pipeline import Pipeline


def run_pipeline(
    configuration,
    *,
    frame_callback=None,
    is_cancelled=None,
    max_events=0,
    status_callback=None,
    input_source=None,
    renderer_factory=None,
    base_dir=None,
    accelerator="auto",
):
    """Run a configured pipeline and report bounded frame events."""
    if max_events is None:
        max_events = 0
    max_events = int(max_events)
    if max_events < 0:
        raise ValueError("max_events must be a non-negative integer")
    is_cancelled = is_cancelled or (lambda: False)

    runtime_config = dict(configuration)
    display_config = dict(runtime_config.get("display") or {})
    runtime_config["display"] = display_config
    show_display = bool(display_config.get("show", True))
    display_config["show"] = False

    events = deque(maxlen=max_events)
    frame_count = 0
    display_clock = None
    accelerator_name = "CPU"

    def on_frame(context, elapsed_ms):
        nonlocal display_clock, frame_count
        display_frame = None
        now = time.monotonic()
        if show_display and (
            display_clock is None or now - display_clock >= 0.08
        ):
            renderer = pipeline.render
            display_frame = (
                renderer(context) if renderer else context.frame.copy()
            )
            fps = (
                1 / (now - display_clock)
                if display_clock is not None and now > display_clock
                else 0
            )
            display_clock = now
            from ..utils.draw import render_resolution, render_status

            render_status(
                display_frame,
                fps=fps,
                accelerator=accelerator_name,
            )
            render_resolution(display_frame)
            height, width = display_frame.shape[:2]
            scale = min(1280 / width, 720 / height, 1.0)
            if scale < 1.0:
                import cv2

                display_frame = cv2.resize(
                    display_frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
        event = {
            "frame": context.frame_index,
            "detections": len(context.results or []),
            "metrics": context.metrics,
            "elapsed_ms": round(elapsed_ms, 2),
            "accelerator": accelerator_name,
            "display_frame": display_frame,
        }
        frame_count += 1
        events.append(event)
        if frame_callback is not None:
            frame_callback(event)

    build_started = time.monotonic()
    pipeline = Pipeline.from_dict(
        runtime_config,
        base_dir=base_dir,
        on_frame=on_frame,
        renderer_factory=renderer_factory,
        accelerator=accelerator,
        input_source=input_source,
    )
    try:
        from ..inference import get_model_accelerator

        accelerator_name = get_model_accelerator(pipeline.model)
        if status_callback is not None:
            status_callback(
                "Model and source ready in "
                f"{time.monotonic() - build_started:.1f}s; waiting for first frame…"
            )
        pipeline.run(is_cancelled=is_cancelled)
        return {
            "events": list(events),
            "frame_count": frame_count,
            "stopped": bool(is_cancelled()),
        }
    finally:
        pipeline.close()


__all__ = ["run_pipeline"]
