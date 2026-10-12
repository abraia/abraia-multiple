"""Pipelines for frame-by-frame inference workflows."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .factories import _build_pipeline
from .lifecycle import CancellationWatcher, close_resources, stop_resource


@dataclass
class FrameContext:
    """State passed between pipeline steps for one frame."""

    frame: Any
    frame_index: int
    frame_time: float
    spectral_cube: Any = None
    spectral_metadata: Any = None
    results: List[dict] = field(default_factory=list)
    current: Any = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    views: Dict[str, List[dict]] = field(default_factory=dict)

    def __post_init__(self):
        if self.current is None:
            self.current = self.results


def _source_frame_parts(source_frame):
    """Return the RGB frame and optional backing spectral data."""
    if hasattr(source_frame, "rgb") and hasattr(source_frame, "cube"):
        return (
            source_frame.rgb,
            source_frame.cube,
            getattr(source_frame, "metadata", None),
        )
    return source_frame, None, None


class Pipeline:
    """Run a model and a sequence of processors over a frame source.

    Steps are called in the order supplied. Each step receives a
    :class:`FrameContext` and may mutate it or return a replacement context.
    The source is expected to be iterable and to expose ``frame_rate`` when
    timestamps based on a video frame rate are useful.
    """

    def __init__(
        self,
        source: Iterable,
        model: Any,
        steps: Optional[Iterable[Callable[[FrameContext], Optional[FrameContext]]]] = None,
        display: Any = None,
        render: Optional[Callable[[FrameContext], Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        components: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.model = model
        self.steps = list(steps or [])
        self.display = display
        self.render = render
        self.model_kwargs = model_kwargs or {}
        self.on_frame = on_frame
        self.components = components or {}
        self._closed = False
        self._stop_requested = threading.Event()
        self._is_cancelled = lambda: False

        frame_rate = getattr(source, "frame_rate", None)
        self.frame_rate = float(frame_rate or 0)

    def stop(self):
        """Request cancellation and release a blocking source if possible."""
        self._stop_requested.set()
        stop_resource(self.source)

    def run(self, is_cancelled=None) -> Optional[FrameContext]:
        """Process all source frames and return the last frame context.

        Models with an ``iter_inference(source)`` method may process frames
        asynchronously.  Results are still consumed in source order so
        stateful steps such as tracking and counting remain deterministic.
        """
        if self._closed:
            raise RuntimeError("Pipeline has already been closed")
        watcher = None
        if is_cancelled is None:
            self._is_cancelled = lambda: False
        else:
            self._is_cancelled = is_cancelled
            watcher = CancellationWatcher(
                is_cancelled,
                self.stop,
                stop_event=self._stop_requested,
            )
            watcher.start()
        try:
            return self._run_frames(self._is_cancelled)
        finally:
            self._stop_requested.set()
            if watcher is not None:
                watcher.close()
            self.close()

    def _run_frames(self, is_cancelled):
        """Run inference and ordered steps over each source frame."""
        last_context = None
        first_step = self.steps[0] if self.steps else None
        inference_model = (
            getattr(first_step, "model", None)
            if first_step is not None
            else self.model
        )
        stream_inference = getattr(inference_model, "iter_inference", None)
        streaming = callable(stream_inference)
        if streaming:
            from .inference import FrameResult

            model_kwargs = (
                first_step.model_kwargs
                if first_step is not None
                else self.model_kwargs
            )
            frame_results = stream_inference(self.source, **model_kwargs)
            frames = enumerate(frame_results)
        else:
            frames = enumerate(self.source)

        single_model = not self.steps and self.model is not None
        for source_index, item in frames:
            if self._stop_requested.is_set() or is_cancelled():
                self.stop()
                break
            if streaming:
                if not isinstance(item, FrameResult):
                    raise TypeError("Async models must yield FrameResult instances")
                frame_index = item.record.index
                source_frame = item.record.frame
                inference_ms = item.elapsed_ms
            else:
                frame_index = source_index
                source_frame = item
                inference_ms = None

            frame, spectral_cube, spectral_metadata = _source_frame_parts(
                source_frame
            )
            frame_time = (
                frame_index / self.frame_rate
                if self.frame_rate > 0
                else float(frame_index)
            )
            context = FrameContext(
                frame,
                frame_index,
                frame_time,
                spectral_cube=spectral_cube,
                spectral_metadata=spectral_metadata,
                current=frame,
            )

            started = time.perf_counter()
            if streaming and first_step is not None:
                first_step.apply_results(context, item.results)
                steps = self.steps[1:]
            elif single_model:
                if streaming:
                    results = item.results
                else:
                    inference_started = time.perf_counter()
                    results = self.model.run(source_frame, **self.model_kwargs)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                context.results = results
                context.current = results
                steps = ()
            else:
                steps = self.steps

            for step in steps:
                context = step(context) or context
            step_elapsed_ms = (time.perf_counter() - started) * 1000
            if single_model:
                elapsed_ms = inference_ms
            elif streaming:
                elapsed_ms = (inference_ms or 0) + step_elapsed_ms
            else:
                elapsed_ms = step_elapsed_ms
            if self.on_frame is not None and elapsed_ms is None:
                elapsed_ms = 0
            last_context = context
            if not self._finish_frame(context, elapsed_ms):
                break
        return last_context

    def _finish_frame(self, context, elapsed_ms):
        """Report, render, and display one completed frame context."""
        if self.on_frame is not None:
            self.on_frame(context, elapsed_ms)
        if self.display is None:
            return True
        output = self.render(context) if self.render else context.frame
        display_result = self.display.show(output)
        if display_result is False or getattr(self.display, "quit", False):
            self.stop()
            return False
        return True

    @classmethod
    def from_file(
        cls,
        path: str,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
        renderer_factory: Optional[Callable] = None,
    ) -> "Pipeline":
        """Build a pipeline from a JSON configuration file."""
        config_path = Path(path)
        with config_path.open("r") as config_file:
            config = json.load(config_file)
        return cls.from_dict(
            config,
            base_dir=config_path.parent,
            on_frame=on_frame,
            accelerator=accelerator,
            renderer_factory=renderer_factory,
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
        renderer_factory: Optional[Callable] = None,
    ) -> "Pipeline":
        """Build a pipeline from a supported versioned JSON configuration.

        The loader intentionally creates the existing SDK components rather
        than importing arbitrary classes named by a configuration file.
        ``accelerator`` selects a paired accelerator implementation when
        available and falls back to ONNX/CPU when it is not.
        """
        return _build_pipeline(
            cls,
            config,
            base_dir=base_dir,
            on_frame=on_frame,
            accelerator=accelerator,
            renderer_factory=renderer_factory,
        )

    def close(self):
        """Close the model and any closeable pipeline components."""
        if self._closed:
            return
        self._closed = True
        step_models = {
            id(step.model)
            for step in self.steps
            if getattr(step, "model", None) is not None
            and callable(getattr(step, "close", None))
        }
        resources = [self.display, self.source]
        if self.model is not None and id(self.model) not in step_models:
            resources.append(self.model)
        close_resources([
            *resources,
            *self.steps,
            *self.components.values(),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
__all__ = [
    "FrameContext",
    "Pipeline",
]
