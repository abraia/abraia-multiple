"""Reusable background execution adapter for Abraia pipelines."""

import copy
import inspect
import time
from collections import deque

from .pipeline import CancellableSource


def run_pipeline(
    config,
    frame_callback=None,
    is_cancelled=None,
    max_events=100,
):
    """Run a configured pipeline and collect bounded backend-neutral events.

    The runner deliberately has no Qt dependency. Applications can consume
    ``frame_callback`` events and decide how to display ``display_frame``.
    Every frame is sent to the callback, while the returned ``events`` list
    retains only the most recent ``max_events`` events. Set ``max_events=0``
    when the caller only needs streaming callbacks.
    """
    from . import Pipeline

    if max_events is None:
        raise ValueError("max_events must be a non-negative integer")
    max_events = int(max_events)
    if max_events < 0:
        raise ValueError("max_events must be a non-negative integer")
    is_cancelled = is_cancelled or (lambda: False)
    runtime_config = copy.deepcopy(config)
    display_config = runtime_config.setdefault("display", {})
    show_display = bool(display_config.get("show", True))
    display_config["show"] = False
    events = deque(maxlen=max_events)
    display_clock = [time.monotonic()]

    def on_frame(context, elapsed_ms):
        from abraia.inference.session import get_model_accelerator

        accelerator = get_model_accelerator(getattr(pipeline, "model", None))
        renderer = getattr(pipeline, "render", None)
        display_frame = None
        if show_display:
            display_frame = renderer(context) if renderer else (
                context.frame.copy() if hasattr(context.frame, "copy") else context.frame
            )
            if hasattr(display_frame, "shape") and len(display_frame.shape) >= 2:
                from abraia.utils.draw import render_resolution, render_status

                now = time.monotonic()
                fps = 1 / (now - display_clock[0]) if now > display_clock[0] else 0
                display_clock[0] = now
                render_status(display_frame, fps=fps, accelerator=accelerator)
                render_resolution(display_frame)
        event = {
            "frame": context.frame_index,
            "detections": len(context.results or []),
            "metrics": context.metrics,
            "elapsed_ms": round(elapsed_ms, 2),
            "accelerator": accelerator,
            "display_frame": display_frame,
        }
        events.append(event)
        if frame_callback is not None:
            frame_callback(event)

    pipeline = Pipeline.from_dict(runtime_config, on_frame=on_frame)
    run = pipeline.run
    if "is_cancelled" in inspect.signature(run).parameters:
        run(is_cancelled=is_cancelled)
    else:
        # Keep compatibility with lightweight test/dummy pipeline adapters.
        run()
    return {"events": list(events), "stopped": bool(is_cancelled())}


__all__ = ["CancellableSource", "run_pipeline"]
