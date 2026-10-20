"""Model-training orchestration."""

import gc
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from ..tasks import (
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_IMAGE_SIZE,
    TRAINING_TASKS,
    normalize_model_size,
    normalize_task,
    to_ultralytics_task,
)
from ..dataset import preparation
from ..dataset.splitting import DEFAULT_EVALUATION_SPLIT


class ModelTrainer:
    """High-level trainer orchestrator using task-specific model adapters."""

    def __init__(
        self,
        project: str,
        task: str,
        classes: list,
        imgsz: int = None,
        client=None,
        model_size="small",
        backend="ultralytics",
        checkpoint=None,
    ):
        task = normalize_task(task)
        if task not in TRAINING_TASKS:
            raise ValueError(f"Unsupported training task: {task}")
        self.project = project
        self.task = task
        self.classes = classes
        self.model_size = normalize_model_size(model_size)
        backend = str(backend).strip().lower()
        if backend not in ("ultralytics", "rtmdet", "libreyolo"):
            raise ValueError(
                "Unsupported training backend: "
                f"{backend}. Choose 'ultralytics', 'rtmdet', or 'libreyolo'."
            )
        if backend == "rtmdet" and task == "classification":
            raise ValueError(
                "RTMDet supports detection and segmentation, not classification."
            )
        if backend == "libreyolo" and task != "detection":
            raise ValueError("LibreYOLO YOLOv9 supports detection training only.")
        self.pbar = None
        imgsz = imgsz or DEFAULT_TRAINING_IMAGE_SIZE[task]
        if task == "classification":
            from . import classify

            self.model = classify.Model(
                client=client, model_size=self.model_size, imgsz=imgsz
            )
        else:
            if backend == "rtmdet":
                from . import rtmdet

                self.model = rtmdet.Model(
                    task,
                    imgsz=imgsz,
                    client=client,
                    model_size=self.model_size,
                    classes=classes,
                    checkpoint=checkpoint,
                )
            elif backend == "libreyolo":
                from . import libreyolo

                self.model = libreyolo.Model(
                    project=project,
                    imgsz=imgsz,
                    client=client,
                    model_size=self.model_size,
                    checkpoint=checkpoint,
                )
            else:
                from . import detect

                detect_kwargs = {
                    "imgsz": imgsz,
                    "client": client,
                    "model_size": self.model_size,
                    "checkpoint": checkpoint,
                }
                if checkpoint is not None:
                    detect_kwargs["model_type"] = (
                        detect.MODEL_SIZE_TYPES[self.model_size]
                    )
                self.model = detect.Model(
                    to_ultralytics_task(task),
                    **detect_kwargs,
                )
        self.backend = backend

    def _progress_callback(self, progress):
        if self.pbar is None:
            self.pbar = tqdm(total=progress["epochs"], initial=progress["epoch"])
        self.pbar.set_description(
            f"Loss: {progress['loss']:.4f} Acc: {progress['acc']:.4f}"
        )
        self.pbar.update(1)

    def train(self, epochs: int = None, batch: int = 32, callback=None,
              is_cancelled=None) -> None:
        epochs = epochs or DEFAULT_TRAINING_EPOCHS[self.task]
        callback = self._progress_callback if callback is None else callback
        try:
            self.model.train(
                self.project,
                epochs=epochs,
                batch=batch,
                callback=callback,
                is_cancelled=is_cancelled,
            )
        finally:
            if self.pbar:
                self.pbar.close()
                self.pbar = None

    def test(self, split: str = DEFAULT_EVALUATION_SPLIT, is_cancelled=None) -> Dict[str, Any]:
        return self.model.test(split=split, is_cancelled=is_cancelled)

    def save(self, device="cpu", publish_common=False) -> Dict[str, Any]:
        if publish_common and self.backend != "libreyolo":
            raise NotImplementedError(
                "Publishing to the common model repository is implemented "
                "for LibreYOLO models only."
            )
        if publish_common:
            return self.model.save(
                self.project,
                self.classes,
                device=device,
                publish_common=True,
            )
        return self.model.save(self.project, self.classes, device=device)

    def run(self, img):
        return self.model.run(img)

    def compile(
        self,
        device="hailo8",
        progress_callback=None,
        is_cancelled=None,
    ):
        if self.task not in ("detection", "segmentation"):
            raise NotImplementedError(
                "Hailo compilation is only implemented for detection and "
                "segmentation models."
            )
        return self.model.compile(
            self.project,
            self.classes,
            device=device,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

def _compile_checkpoint(
    project,
    task,
    classes,
    checkpoint,
    backend,
    model_size="small",
    imgsz=640,
    client=None,
    config_path=None,
    model_version=None,
    **options,
):
    """Compile local weights for the CLI without exposing paths on ModelTrainer."""
    task = normalize_task(task)
    model_size = normalize_model_size(model_size)
    if task not in ("detection", "segmentation"):
        raise ValueError("Hailo compilation supports detection and segmentation only.")
    if backend == "libreyolo":
        from .libreyolo import Model

        model = Model(
            project=project,
            imgsz=imgsz,
            client=client,
            model_size=model_size,
        )
        model.checkpoint_path = Path(checkpoint)
        model.model_version = model_version
    elif backend == "rtmdet":
        from .rtmdet import Model

        model = Model(
            task,
            imgsz=imgsz,
            client=client,
            checkpoint=checkpoint,
            model_size=model_size,
            classes=classes,
        )
        model.model_version = model_version
    else:
        from .detect import MODEL_SIZE_TYPES, Model

        model = Model(
            to_ultralytics_task(task),
            model_type=MODEL_SIZE_TYPES[model_size],
            client=client,
            checkpoint=checkpoint,
            model_size=model_size,
            imgsz=imgsz,
        )
        model.model_version = model_version
    if backend == "rtmdet":
        return model.compile(
            project, classes, config_path=config_path, **options
        )
    if backend == "libreyolo":
        return model.compile(project, classes, **options)
    return model.compile(project, classes, **options)


def train_dataset(
    project, dataset, epochs, training_callback, is_cancelled,
    model_size="small", imgsz=None,
    retain_trainer=False, analyze_thresholds=False, backend="ultralytics",
    checkpoint=None, publish_common=False,
):
    """Prepare, train, validate, and save a dataset model."""
    trainer = None
    try:
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Preparing dataset"})
        preparation.prepare_dataset(
            dataset,
            callback=lambda event: training_callback(
                {"stage": "Preparing dataset", **event}
            ),
        )
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Loading model"})
        trainer_kwargs = {"client": getattr(dataset, "client", None)}
        if backend != "ultralytics":
            trainer_kwargs["backend"] = backend
        if checkpoint is not None:
            trainer_kwargs["checkpoint"] = checkpoint
        if imgsz is not None:
            trainer_kwargs["imgsz"] = imgsz
        if model_size != "small":
            trainer_kwargs["model_size"] = model_size
        trainer = ModelTrainer(
            project, dataset.task, dataset.classes, **trainer_kwargs
        )
        task = trainer.task

        def on_epoch(metrics):
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Training", **metrics})

        training_callback({"stage": "Training"})
        trainer.train(
            epochs=epochs,
            callback=on_epoch,
            is_cancelled=is_cancelled,
        )
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Validating model"})
        stats = trainer.test(
            split=DEFAULT_EVALUATION_SPLIT,
            is_cancelled=is_cancelled,
        )
        result = {"stats": stats, "epochs": epochs, "task": task}
        if (
            analyze_thresholds
            and trainer.backend == "ultralytics"
            and task in ("detection", "segmentation")
        ):
            from .thresholds import detection_threshold_analysis

            training_callback({"stage": "Analyzing validation thresholds"})
            try:
                threshold_analysis = detection_threshold_analysis(
                    trainer,
                    is_cancelled=is_cancelled,
                )
                if is_cancelled():
                    raise RuntimeError("Training canceled")
            except Exception as error:
                if is_cancelled() or (
                    isinstance(error, RuntimeError)
                    and "cancel" in str(error).lower()
                ):
                    raise RuntimeError("Training canceled") from error
                result["threshold_analysis_error"] = str(error)
            else:
                if threshold_analysis is None:
                    result["threshold_analysis_error"] = (
                        "No labeled validation predictions were available."
                    )
                else:
                    stats["threshold_analysis"] = threshold_analysis
                    trainer.model.metrics["threshold_analysis"] = threshold_analysis
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Exporting model"})
        result["saved_model"] = trainer.save(publish_common=publish_common)
        if (
            retain_trainer
            and trainer.backend == "ultralytics"
            and task in ("detection", "segmentation")
        ):
            result["hailo_trainer"] = trainer
        return result
    finally:
        del trainer
        gc.collect()


__all__ = ["train_dataset"]
