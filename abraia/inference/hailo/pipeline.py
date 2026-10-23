"""Hailo adapters for the generic Abraia runtime pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, Optional

from ..model_config import normalize_task
from ...runtime.inference import (
    AsyncInferenceRunner,
    FrameBatch,
    FrameResult,
)
from .toolbox import (
    HAILO_AVAILABLE,
    ModelInference,
)
from .models import (
    labels_from_model_config,
    load_hailo_model_config,
    resolve_model_type,
)
from .postprocess import default_preprocess, rtmdet_preprocess


class HailoPipelineModel:
    """Expose Hailo's producer/consumer inference as a pipeline model.

    The generic :class:`abraia.runtime.Pipeline` consumes the
    ``iter_inference`` protocol when it is present.  This adapter prepares
    batches in one thread, submits them to Hailo's asynchronous runtime in a
    second thread, and yields completed frames in source order.  The latter
    is important because tracking and counting steps are stateful.
    """

    def __init__(
        self,
        hef_path: str,
        task: str = "detection",
        batch_size: int = 1,
        score_threshold: Optional[float] = None,
        mask_threshold: float = 0.45,
        iou_threshold: Optional[float] = None,
        kind: Optional[str] = None,
    ):
        if not HAILO_AVAILABLE or ModelInference is None:
            raise RuntimeError(
                "Hailo support is unavailable; install hailo_platform on a "
                "Hailo-enabled host"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        model_config = load_hailo_model_config(hef_path)
        self.kind = str(kind or model_config.get("kind", "yolov8")).strip().lower()
        score_threshold = (
            model_config.get("confidenceThreshold", 0.25)
            if score_threshold is None else score_threshold
        )
        iou_threshold = (
            model_config.get("iouThreshold", 0.7)
            if iou_threshold is None else iou_threshold
        )
        labels = labels_from_model_config(model_config)
        metadata_task = normalize_task(model_config.get("task"))
        task = normalize_task(task)
        if task == "detection" and metadata_task in ("detection", "segmentation", "pose"):
            task = metadata_task
        self.task = task
        self.accelerator = "HAILO"
        self.labels = labels
        self.model_type = resolve_model_type(self.kind, task)
        self.inference = ModelInference(
            hef_path,
            task=task,
            labels=labels,
            batch_size=batch_size,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
            model_type=self.model_type,
            model_config=model_config,
        )
        self.batch_size = batch_size
        self.input_height, self.input_width, _ = self.inference.get_input_shape()
        self._closed = False
        self._runner = None

    def _infer_batch(self, batch: FrameBatch, emit, stop_event):
        """Submit one runtime batch and adapt Hailo callbacks to FrameResult."""

        def emit_result(index, results, error=None, inference_ms=None):
            return emit(FrameResult(
                batch.records[index],
                results,
                error=error,
                inference_ms=inference_ms,
            ))

        self.inference.infer_batch(
            batch.processed_frames,
            emit_result,
            stop_event=stop_event,
            image_batch=[
                record.frame["frame"] if isinstance(record.frame, Mapping)
                else record.frame
                for record in batch.records
            ],
        )

    def _preprocess_frame(self, frame):
        if isinstance(frame, Mapping):
            frame = frame["frame"]
        if self.model_type == "rtmdet":
            return rtmdet_preprocess(
                frame,
                self.input_width,
                self.input_height,
            )
        return default_preprocess(frame, self.input_width, self.input_height)

    def iter_inference(self, source: Iterable, **_model_kwargs):
        """Yield :class:`FrameResult` objects for ``source``."""
        if self._closed:
            raise RuntimeError("Hailo pipeline model has already been closed")

        runner = AsyncInferenceRunner(
            source=source,
            inference=self._infer_batch,
            preprocess=self._preprocess_frame,
            batch_size=self.batch_size,
            flush=getattr(self.inference, "wait", None),
        )
        self._runner = runner
        try:
            yield from runner
        finally:
            runner.close()
            self._runner = None

    def close(self):
        """Stop active workers and release the Hailo runtime."""
        if self._closed:
            return
        self._closed = True
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        close = getattr(self.inference, "close", None)
        if callable(close):
            close()


__all__ = ["HailoPipelineModel"]
