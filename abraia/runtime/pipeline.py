"""Pipelines for frame-by-frame inference workflows."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from .factories import (
    PipelineBuilder,
    RegionFilterStage,
    RegionTimerStage,
    LineCounterStage,
    TrackerStage,
)
from .lifecycle import CancellationWatcher, close_resources, stop_resource


class CancellableSource:
    """Wrap a blocking source with cooperative cancellation."""

    def __init__(self, source, is_cancelled):
        self.source = source
        self.frame_rate = getattr(source, "frame_rate", 0)
        self._is_cancelled = is_cancelled
        self._watcher = CancellationWatcher(
            is_cancelled,
            lambda: stop_resource(self.source),
        )

    def stop(self):
        self._watcher.stop_event.set()
        stop_resource(self.source)

    def __iter__(self):
        self._watcher.start()
        try:
            for frame in self.source:
                if self._is_cancelled():
                    break
                yield frame
        finally:
            self.stop()
            self._watcher.close()


@dataclass
class FrameContext:
    """State passed between pipeline stages for one frame."""

    frame: Any
    frame_index: int
    frame_time: float
    results: List[dict] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    views: Dict[str, List[dict]] = field(default_factory=dict)


class Pipeline:
    """Run a model and a sequence of processors over a frame source.

    Stages are called in the order supplied. Each stage receives a
    :class:`FrameContext` and may mutate it or return a replacement context.
    The source is expected to be iterable and to expose ``frame_rate`` when
    timestamps based on a video frame rate are useful.
    """

    def __init__(
        self,
        source: Iterable,
        model: Any,
        stages: Optional[Iterable[Callable[[FrameContext], Optional[FrameContext]]]] = None,
        display: Any = None,
        render: Optional[Callable[[FrameContext], Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        components: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.model = model
        self.stages = list(stages or [])
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

    def _iter_inference(self):
        """Yield :class:`FrameResult` objects from the model.

        Most models expose the original synchronous ``run`` method.  Hardware
        backends may instead expose ``iter_inference(source)`` and overlap
        capture, preprocessing, and inference internally.  Keeping this small
        protocol here lets the rest of the pipeline (including stateful
        stages) remain backend-independent.
        """
        from .inference import FrameRecord, FrameResult

        async_iterator = getattr(self.model, "iter_inference", None)
        if callable(async_iterator):
            for result in async_iterator(self.source, **self.model_kwargs):
                if not isinstance(result, FrameResult):
                    raise TypeError(
                        "Async models must yield FrameResult instances"
                    )
                yield result
            return

        for frame_index, frame in enumerate(self.source):
            if self._stop_requested.is_set() or self._is_cancelled():
                self.stop()
                break
            started = time.perf_counter()
            yield FrameResult(
                FrameRecord(frame_index, frame, started),
                self.model.run(frame, **self.model_kwargs),
                (time.perf_counter() - started) * 1000,
            )

    def stop(self):
        """Request cancellation and release a blocking source if possible."""
        self._stop_requested.set()
        stop_resource(self.source)

    def run(self, is_cancelled=None) -> Optional[FrameContext]:
        """Process all source frames and return the last frame context.

        Models with an ``iter_inference(source)`` method may process frames
        asynchronously.  Results are still consumed in source order so
        stateful stages such as tracking and counting remain deterministic.
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
        last_context = None
        try:
            for result in self._iter_inference():
                if self._stop_requested.is_set() or self._is_cancelled():
                    self.stop()
                    break
                frame_index = result.record.index
                frame = result.record.frame
                results = result.results
                elapsed_ms = result.elapsed_ms
                frame_time = (
                    frame_index / self.frame_rate
                    if self.frame_rate > 0
                    else float(frame_index)
                )
                context = FrameContext(frame, frame_index, frame_time)
                context.results = results

                for stage in self.stages:
                    context = stage(context) or context

                if self.on_frame is not None:
                    if elapsed_ms is None:
                        elapsed_ms = 0
                    self.on_frame(context, elapsed_ms)

                output = self.render(context) if self.render else context.frame
                last_context = context
                if self.display is not None:
                    display_result = self.display.show(output)
                    # Display backends may report that their window was
                    # closed either through the return value or a persistent
                    # ``quit`` flag.  Propagate that request to the source
                    # and asynchronous model immediately; otherwise a
                    # preview key such as ``q`` is ignored until the input
                    # stream ends.
                    if (
                        display_result is False
                        or getattr(self.display, "quit", False)
                    ):
                        self.stop()
                        break

            return last_context
        finally:
            self._stop_requested.set()
            if watcher is not None:
                watcher.close()
            self.close()

    @classmethod
    def from_file(
        cls,
        path: str,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
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
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        base_dir: Optional[str] = None,
        on_frame: Optional[Callable[[FrameContext, float], None]] = None,
        accelerator: Optional[str] = "auto",
    ) -> "Pipeline":
        """Build a pipeline from the small version-1 JSON schema.

        The loader intentionally creates the existing SDK components rather
        than importing arbitrary classes named by a configuration file.
        ``accelerator`` selects a paired accelerator implementation when
        available and falls back to ONNX/CPU when it is not.
        """
        return PipelineBuilder(cls).build(
            config,
            base_dir=base_dir,
            on_frame=on_frame,
            accelerator=accelerator,
        )

    def close(self):
        """Close the model and any closeable pipeline components."""
        if self._closed:
            return
        self._closed = True
        close_resources([
            self.display,
            self.source,
            self.model,
            *self.stages,
            *self.components.values(),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
__all__ = [
    "FrameContext",
    "Pipeline",
    "TrackerStage",
    "LineCounterStage",
    "RegionFilterStage",
    "RegionTimerStage",
]
