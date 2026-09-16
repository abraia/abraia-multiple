"""Hailo adapters for the generic Abraia runtime pipeline."""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterable, Optional

from .toolbox import (
    HAILO_AVAILABLE,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    ModelInference,
    default_preprocess,
)


class HailoPipelineModel:
    """Expose Hailo's producer/consumer inference as a pipeline model.

    The generic :class:`abraia.runtime.Pipeline` consumes the
    ``iter_inference`` protocol when it is present.  This adapter prepares
    batches in one thread, submits them to Hailo's asynchronous runtime in a
    second thread, and yields completed frames in source order.  The latter
    is important because tracking and counting stages are stateful.
    """

    def __init__(
        self,
        hef_path: str,
        task: str = "detect",
        labels: Optional[list] = None,
        batch_size: int = 1,
        score_threshold: float = 0.25,
        mask_threshold: float = 0.45,
        model_type: Optional[str] = None,
    ):
        if not HAILO_AVAILABLE or ModelInference is None:
            raise RuntimeError(
                "Hailo support is unavailable; install hailo_platform on a "
                "Hailo-enabled host"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        task = {"detection": "detect", "segmentation": "segment"}.get(
            str(task).strip().lower(), str(task).strip().lower()
        )
        if model_type is None:
            from .detect import resolve_model_type

            model_type = resolve_model_type(None, hef_path, task)
        self.inference = ModelInference(
            hef_path,
            task=task,
            labels=labels,
            batch_size=batch_size,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            model_type=model_type or "v8",
        )
        self.batch_size = batch_size
        self.input_height, self.input_width, _ = self.inference.get_input_shape()
        self._closed = False
        self._active_stop_event = None
        self._active_threads = ()

    def iter_inference(self, source: Iterable):
        """Yield ``(frame_index, frame, results, elapsed_ms)`` for ``source``."""
        if self._closed:
            raise RuntimeError("Hailo pipeline model has already been closed")

        input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
        output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)
        stop_event = threading.Event()
        worker_errors = queue.Queue()
        self._active_stop_event = stop_event

        def put_until_stopped(target, item):
            while not stop_event.is_set():
                try:
                    target.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def put_sentinel(target):
            """Ensure a worker can be released even after cancellation."""
            while True:
                try:
                    target.put(None, timeout=0.1)
                    return
                except queue.Full:
                    # Discarding queued work is correct during shutdown and
                    # prevents a failed consumer from leaving a worker stuck.
                    try:
                        target.get_nowait()
                    except queue.Empty:
                        continue

        def produce():
            raw_batch = []
            processed_batch = []
            frame_info_batch = []
            try:
                for frame_index, frame in enumerate(source):
                    if stop_event.is_set():
                        break
                    started = time.perf_counter()
                    processed = default_preprocess(
                        frame, self.input_width, self.input_height
                    )
                    raw_batch.append(frame)
                    processed_batch.append(processed)
                    frame_info_batch.append((frame_index, started))
                    if len(raw_batch) >= self.batch_size:
                        if not put_until_stopped(
                            input_queue,
                            (frame_info_batch, raw_batch, processed_batch),
                        ):
                            break
                        raw_batch, processed_batch, frame_info_batch = [], [], []
                if raw_batch and not stop_event.is_set():
                    put_until_stopped(
                        input_queue,
                        (frame_info_batch, raw_batch, processed_batch),
                    )
            except Exception as exc:
                worker_errors.put(exc)
                stop_event.set()
            finally:
                put_sentinel(input_queue)

        def infer():
            try:
                # ModelInference owns the Hailo async jobs and places each
                # completed frame into output_queue from its completion
                # callback.  It also waits for pending jobs during close.
                self.inference.infer(input_queue, output_queue, stop_event)
            except Exception as exc:
                worker_errors.put(exc)
                stop_event.set()
                put_sentinel(output_queue)

        producer_thread = threading.Thread(
            target=produce, name="hailo-pipeline-preprocess", daemon=True
        )
        inference_thread = threading.Thread(
            target=infer, name="hailo-pipeline-inference", daemon=True
        )
        self._active_threads = (producer_thread, inference_thread)
        producer_thread.start()
        inference_thread.start()

        pending = {}
        next_frame = 0
        try:
            while True:
                item = output_queue.get()
                try:
                    if item is None:
                        break
                    frame_info, frame, results = item
                    frame_index, started = frame_info
                    pending[frame_index] = (
                        frame,
                        results,
                        (time.perf_counter() - started) * 1000,
                    )
                    while next_frame in pending:
                        yield (next_frame, *pending.pop(next_frame))
                        next_frame += 1
                finally:
                    output_queue.task_done()
        finally:
            stop_event.set()
            # Unblock workers if the consumer or caller stops early.
            put_sentinel(input_queue)
            producer_thread.join(timeout=2)
            inference_thread.join(timeout=2)
            self._active_threads = ()
            self._active_stop_event = None

        if not worker_errors.empty():
            raise worker_errors.get()
        if pending:
            raise RuntimeError("Hailo pipeline ended before all frames completed")

    def close(self):
        """Stop active workers and release the Hailo runtime."""
        if self._closed:
            return
        self._closed = True
        if self._active_stop_event is not None:
            self._active_stop_event.set()
        for thread in self._active_threads:
            thread.join(timeout=2)
        self._active_threads = ()
        self._active_stop_event = None
        close = getattr(self.inference, "close", None)
        if callable(close):
            close()


__all__ = ["HailoPipelineModel"]
