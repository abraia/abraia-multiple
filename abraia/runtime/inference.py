"""Shared producer/consumer orchestration for asynchronous inference backends."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRecord:
    """A source frame together with its runtime identity and start time."""

    index: int
    frame: Any
    started: float


@dataclass(frozen=True)
class FrameBatch:
    """A batch passed from the runtime producer to an inference backend."""

    records: Sequence[FrameRecord]
    processed_frames: Sequence[Any]

    def __post_init__(self):
        if len(self.records) != len(self.processed_frames):
            raise ValueError("A frame batch must have matching raw and processed frames")


@dataclass(frozen=True)
class FrameResult:
    """Inference output associated with one source frame."""

    record: FrameRecord
    results: Any
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None


ResultEmitter = Callable[[FrameResult], bool]
BatchInference = Callable[[FrameBatch, ResultEmitter, threading.Event], None]


class AsyncInferenceRunner:
    """Batch frames and consume asynchronous inference results in order.

    Backends receive a :class:`FrameBatch`, an emitter callback, and a
    cancellation event. They must call the emitter once for every frame they
    accept. A backend may submit work asynchronously, but must be paired with
    ``flush`` so all callbacks have completed before the runner signals the
    consumer.

    The runner owns queues, worker threads, ordering, cancellation, sentinels,
    and worker-error propagation. Backends do not need to know about queues or
    runtime sentinel values.
    """

    def __init__(
        self,
        source: Iterable,
        inference: BatchInference,
        preprocess: Callable[[Any], Any],
        batch_size: int = 1,
        stop_event: Optional[threading.Event] = None,
        input_queue_size: int = 60,
        output_queue_size: int = 60,
        join_timeout: Optional[float] = 2.0,
        flush: Optional[Callable[[], None]] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if input_queue_size < 1 or output_queue_size < 1:
            raise ValueError("queue sizes must be at least 1")

        self.source = source
        self.inference = inference
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.stop_event = stop_event or threading.Event()
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size
        self.join_timeout = join_timeout
        self.flush = flush

        self._closed = False
        self._started = False
        self._active_threads = ()
        self._input_queue = None
        self._output_queue = None
        self._produced_count = 0

    @staticmethod
    def _put_until_stopped(target, item, stop_event) -> bool:
        """Put an item without leaving a producer blocked during shutdown."""
        while not stop_event.is_set():
            try:
                target.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _put_sentinel(target, stop_event) -> None:
        """Release a worker even when cancellation left its queue full."""
        while True:
            try:
                target.put(None, timeout=0.1)
                return
            except queue.Full:
                if not stop_event.is_set():
                    continue
                try:
                    target.get_nowait()
                    target.task_done()
                except queue.Empty:
                    continue

    def _produce(self, input_queue, worker_errors):
        records = []
        processed_frames = []
        try:
            for frame_index, frame in enumerate(self.source):
                if self.stop_event.is_set():
                    break
                record = FrameRecord(frame_index, frame, time.perf_counter())
                self._produced_count = frame_index + 1
                records.append(record)
                processed_frames.append(self.preprocess(frame))
                if len(records) >= self.batch_size:
                    if not self._put_until_stopped(
                        input_queue,
                        FrameBatch(tuple(records), tuple(processed_frames)),
                        self.stop_event,
                    ):
                        break
                    records, processed_frames = [], []

            if records and not self.stop_event.is_set():
                self._put_until_stopped(
                    input_queue,
                    FrameBatch(tuple(records), tuple(processed_frames)),
                    self.stop_event,
                )
        except Exception as exc:
            worker_errors.put(exc)
            self.stop_event.set()
        finally:
            self._put_sentinel(input_queue, self.stop_event)

    def _emit(self, output_queue, result) -> bool:
        if not isinstance(result, FrameResult):
            raise TypeError("Inference backends must emit FrameResult instances")
        return self._put_until_stopped(output_queue, result, self.stop_event)

    def _infer(self, input_queue, output_queue, worker_errors):
        try:
            while True:
                batch = input_queue.get()
                try:
                    if batch is None:
                        break
                    if self.stop_event.is_set():
                        continue
                    self.inference(
                        batch,
                        lambda result: self._emit(output_queue, result),
                        self.stop_event,
                    )
                finally:
                    input_queue.task_done()
        except Exception as exc:
            worker_errors.put(exc)
            self.stop_event.set()
        finally:
            if self.flush is not None:
                try:
                    self.flush()
                except Exception as exc:
                    worker_errors.put(exc)
                    self.stop_event.set()
            # The backend must have flushed its asynchronous jobs before the
            # runner signals completion to the consumer.
            self._put_sentinel(output_queue, self.stop_event)

    def __iter__(self):
        return self.iter_inference()

    def iter_inference(self):
        """Yield :class:`FrameResult` objects in source order."""
        if self._closed:
            raise RuntimeError("Async inference runner has already been closed")
        if self._started:
            raise RuntimeError("Async inference runner can only be iterated once")
        self._started = True

        input_queue = queue.Queue(self.input_queue_size)
        output_queue = queue.Queue(self.output_queue_size)
        worker_errors = queue.Queue()
        self._input_queue = input_queue
        self._output_queue = output_queue

        producer_thread = threading.Thread(
            target=self._produce,
            args=(input_queue, worker_errors),
            name="async-inference-producer",
            daemon=True,
        )
        inference_thread = threading.Thread(
            target=self._infer,
            args=(input_queue, output_queue, worker_errors),
            name="async-inference-worker",
            daemon=True,
        )
        self._active_threads = (producer_thread, inference_thread)
        producer_thread.start()
        inference_thread.start()

        pending = {}
        seen_indices = set()
        next_frame = 0
        completed = False
        cancelled = False
        try:
            while True:
                item = output_queue.get()
                try:
                    if item is None:
                        completed = True
                        cancelled = self.stop_event.is_set()
                        break
                    if not isinstance(item, FrameResult):
                        raise TypeError(
                            "Inference runners only consume FrameResult instances"
                        )
                    record = item.record
                    if record.index in seen_indices:
                        raise RuntimeError(
                            f"Inference emitted duplicate frame index {record.index}"
                        )
                    if item.error is not None:
                        raise item.error
                    seen_indices.add(record.index)
                    pending[record.index] = FrameResult(
                        record,
                        item.results,
                        item.elapsed_ms
                        if item.elapsed_ms is not None
                        else (time.perf_counter() - record.started) * 1000,
                    )
                    while next_frame in pending:
                        yield pending.pop(next_frame)
                        next_frame += 1
                finally:
                    output_queue.task_done()
        finally:
            self.close()

        error = self._first_error(worker_errors)
        if error is not None:
            raise error
        if completed and not cancelled and next_frame != self._produced_count:
            raise RuntimeError(
                "Async inference ended before all frames completed: "
                f"received {next_frame} of {self._produced_count}"
            )
        if pending:
            raise RuntimeError("Async inference ended before all frames completed")

    @staticmethod
    def _first_error(worker_errors):
        try:
            return worker_errors.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        """Cancel workers and wait for them before releasing runner state."""
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        if self._input_queue is not None:
            self._put_sentinel(self._input_queue, self.stop_event)
        for thread in self._active_threads:
            thread.join(timeout=self.join_timeout)
            if thread.is_alive():
                logger.warning(
                    "Async inference worker exceeded shutdown timeout; waiting"
                )
                # Do not release backend-owned resources while a worker may
                # still be using them. The timeout is diagnostic only.
                thread.join()
        self._active_threads = ()
        self._input_queue = None
        self._output_queue = None


__all__ = [
    "AsyncInferenceRunner",
    "BatchInference",
    "FrameBatch",
    "FrameRecord",
    "FrameResult",
    "ResultEmitter",
]
