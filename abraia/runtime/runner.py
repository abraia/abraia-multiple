"""Reusable background execution adapter for Abraia pipelines."""

import copy
import time
from collections import deque

from .pipeline import CancellableSource


def prepare_pipeline_display_frame(frame, max_size=(1280, 720)):
    """Bound a rendered frame's size before sending it to an application UI."""
    import numpy as np

    frame = np.asarray(frame)
    if frame.ndim < 2:
        return frame
    height, width = frame.shape[:2]
    max_width, max_height = (int(value) for value in max_size)
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    import cv2

    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def run_pipeline(
    config,
    frame_callback=None,
    is_cancelled=None,
    max_events=100,
    renderer_factory=None,
    status_callback=None,
):
    """Run a configured pipeline and collect bounded backend-neutral events.

    Applications can consume ``frame_callback`` events and decide how to
    display ``display_frame``. Every frame is sent to the callback, while the
    returned ``events`` list retains only the most recent ``max_events``.
    Set ``max_events=0`` when the caller only needs streaming callbacks.
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
    frame_count = [0]
    accelerator = ["CPU"]

    def on_frame(context, elapsed_ms):
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
                render_status(
                    display_frame,
                    fps=fps,
                    accelerator=accelerator[0],
                )
                render_resolution(display_frame)
        event = {
            "frame": context.frame_index,
            "detections": len(context.results or []),
            "metrics": context.metrics,
            "elapsed_ms": round(elapsed_ms, 2),
            "accelerator": accelerator[0],
            "display_frame": display_frame,
        }
        frame_count[0] += 1
        events.append(event)
        if frame_callback is not None:
            frame_callback(event)

    build_started = time.monotonic()
    pipeline = Pipeline.from_dict(
        runtime_config,
        on_frame=on_frame,
        renderer_factory=renderer_factory,
    )
    from abraia.inference.session import get_model_accelerator

    accelerator[0] = get_model_accelerator(getattr(pipeline, "model", None))
    if status_callback is not None:
        build_seconds = time.monotonic() - build_started
        status_callback(
            f"Model and source ready in {build_seconds:.1f}s; waiting for first frame…"
        )
    pipeline.run(is_cancelled=is_cancelled)
    return {
        "events": list(events),
        "frame_count": frame_count[0],
        "stopped": bool(is_cancelled()),
    }


__all__ = [
    "CancellableSource",
    "prepare_pipeline_display_frame",
    "run_pipeline",
]
