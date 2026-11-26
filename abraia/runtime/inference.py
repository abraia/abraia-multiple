"""Shared producer/consumer orchestration for asynchronous inference backends."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    capture_index: Optional[int] = None
    captured_at: Optional[float] = None


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
    """Inference output plus frame latency and backend inference time."""

    record: FrameRecord
    results: Any
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None
    inference_ms: Optional[float] = None


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
        input_queue_size: int = 4,
        output_queue_size: int = 4,
        join_timeout: Optional[float] = 2.0,
        flush: Optional[Callable[[], None]] = None,
        live_max_in_flight_frames: Optional[int] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if input_queue_size < 1 or output_queue_size < 1:
            raise ValueError("queue sizes must be at least 1")
        if live_max_in_flight_frames is not None and (
            not isinstance(live_max_in_flight_frames, int)
            or live_max_in_flight_frames < batch_size
        ):
            raise ValueError("Live inference frame budget must be an integer that accommodates one batch")

        self.source = source
        self.inference = inference
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.stop_event = stop_event or threading.Event()
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size
        self.join_timeout = join_timeout
        self.flush = flush
        self.live_max_in_flight_frames = live_max_in_flight_frames
        self._live_condition = (
            threading.Condition() if live_max_in_flight_frames is not None else None
        )

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

    def _iter_live(self):
        """Overlap inference with consumption, retaining the newest ready result."""
        condition = self._live_condition
        frame_budget = self.live_max_in_flight_frames
        in_flight = set()
        latest = None
        newest_index = -1
        error = None
        done = False

        def emit(result):
            nonlocal latest, newest_index, error
            with condition:
                if self.stop_event.is_set() or error is not None:
                    return False
                if not isinstance(result, FrameResult) or result.record.index not in in_flight:
                    error = RuntimeError("Live inference emitted an unexpected frame result")
                else:
                    index = result.record.index
                    in_flight.remove(index)
                    if result.error is not None:
                        error = result.error
                    elif index > newest_index:
                        newest_index = index
                        latest = result if result.elapsed_ms is not None else replace(
                            result, elapsed_ms=(time.perf_counter() - result.record.started) * 1000,
                        )
                condition.notify_all()
                return error is None

        def submit_frames():
            nonlocal error, done
            try:
                frames = iter(self.source)
                index = 0
                while True:
                    with condition:
                        condition.wait_for(
                            lambda: self.stop_event.is_set() or error is not None
                            or len(in_flight) + self.batch_size <= frame_budget,
                            timeout=0.1,
                        )
                        if self.stop_event.is_set() or error is not None:
                            break
                        if len(in_flight) + self.batch_size > frame_budget:
                            continue
                    records = []
                    for _ in range(self.batch_size):
                        if self.stop_event.is_set():
                            break
                        try:
                            frame = next(frames)
                        except StopIteration:
                            break
                        records.append(FrameRecord(
                            index, frame, time.perf_counter(),
                            getattr(self.source, "last_capture_index", None),
                            getattr(self.source, "last_captured_at", None),
                        ))
                        index += 1
                    if not records or self.stop_event.is_set():
                        break
                    processed = tuple(self.preprocess(record.frame) for record in records)
                    with condition:
                        if self.stop_event.is_set() or error is not None:
                            break
                        in_flight.update(record.index for record in records)
                    self.inference(FrameBatch(tuple(records), processed), emit, self.stop_event)
                    if len(records) < self.batch_size:
                        break
            except Exception as exc:
                with condition:
                    if error is None and not self.stop_event.is_set():
                        error = exc
                    condition.notify_all()
            finally:
                try:
                    if self.flush is not None:
                        self.flush()
                except Exception as exc:
                    with condition:
                        if error is None:
                            error = exc
                finally:
                    with condition:
                        if in_flight and error is None and not self.stop_event.is_set():
                            error = RuntimeError("Live inference ended before all frames completed")
                        done = True
                        condition.notify_all()

        worker = threading.Thread(
            target=submit_frames, name="abraia-live-inference", daemon=True,
        )
        self._active_threads = (worker,)
        worker.start()
        try:
            while True:
                with condition:
                    condition.wait_for(
                        lambda: latest is not None or done or error is not None
                        or self.stop_event.is_set(),
                        timeout=0.1,
                    )
                    if error is not None:
                        raise error
                    if self.stop_event.is_set():
                        break
                    result, latest = latest, None
                    if result is None and done:
                        break
                if result is not None:
                    yield result
        finally:
            self.close()

    def iter_inference(self):
        """Yield :class:`FrameResult` objects in source order."""
        if self._closed:
            raise RuntimeError("Async inference runner has already been closed")
        if self._started:
            raise RuntimeError("Async inference runner can only be iterated once")
        self._started = True
        if self._live_condition is not None:
            yield from self._iter_live()
            return

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
                    record = item.record
                    if record.index < next_frame or record.index in pending:
                        raise RuntimeError(
                            f"Inference emitted duplicate frame index {record.index}"
                        )
                    if item.error is not None:
                        raise item.error
                    pending[record.index] = FrameResult(
                        record,
                        item.results,
                        item.elapsed_ms
                        if item.elapsed_ms is not None
                        else (time.perf_counter() - record.started) * 1000,
                        inference_ms=item.inference_ms,
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
        if self._live_condition is not None:
            with self._live_condition:
                self._live_condition.notify_all()
            if any(thread.is_alive() for thread in self._active_threads):
                stop_source = getattr(self.source, "stop", None)
                if callable(stop_source):
                    try:
                        stop_source()
                    except Exception:
                        logger.warning("Failed to stop live inference source", exc_info=True)
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
