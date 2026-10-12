"""Dataset preparation and model-training orchestration."""

import gc
import os
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from PIL import Image
from tqdm import tqdm

from ..tasks import (
    DEFAULT_TRAINING_EPOCHS,
    TRAINING_TASKS,
    normalize_model_size,
    normalize_task,
    to_ultralytics_task,
)
from ..utils import save_text
from ..utils.concurrency import (
    DEFAULT_MAX_REMOTE_WORKERS,
    bounded_as_completed,
)
from .artifacts import _resolve_client
from .splitting import (
    DATASET_SPLITS,
    DEFAULT_EVALUATION_SPLIT,
    DEFAULT_SPLIT_OPTIONS,
    SPLIT_RATIO_KEYS,
    split_annotation_records,
    summarize_split_records,
)


def download_file(path, folder, client=None):
    """Download one remote training file when it is not already local."""
    client = _resolve_client(client)
    destination = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(destination):
        client.download_file(path, destination)
    return destination


def save_annotation(annotation, folder, classes, task):
    """Write one annotation record in YOLO detection/segmentation format."""
    task = normalize_task(task)
    image = Image.open(os.path.join(folder, "images", annotation["filename"]))
    label_lines = []
    for obj in annotation.get("objects", []):
        label, box, polygon = obj.get("label"), obj.get("box"), obj.get("polygon")
        if task == "segmentation" and polygon:
            label_lines.append(
                f"{classes.index(label)} "
                + " ".join(
                    f"{point[0] / image.width} {point[1] / image.height}"
                    for point in polygon
                )
            )
        elif task == "detection":
            if polygon:
                x_values = [point[0] for point in polygon]
                y_values = [point[1] for point in polygon]
                x1, y1 = min(x_values), min(y_values)
                x2, y2 = max(x_values), max(y_values)
                box = [x1, y1, x2 - x1, y2 - y1]
            if box:
                label_lines.append(
                    f"{classes.index(label)} "
                    f"{(box[0] + box[2] / 2) / image.width} "
                    f"{(box[1] + box[3] / 2) / image.height} "
                    f"{box[2] / image.width} {box[3] / image.height}"
                )
    label_path = os.path.join(
        folder,
        "labels",
        f"{os.path.splitext(annotation['filename'])[0]}.txt",
    )
    save_text(label_path, "\n".join(label_lines))


def save_data(annotation, folder, classes, task, client=None, remote_path=None):
    """Download one annotation image and write its training label."""
    task = normalize_task(task)
    path = remote_path or annotation["path"]
    destination = folder if task == "classification" else os.path.join(folder, "images")
    if task == "classification":
        label = next(
            (obj.get("label", "") for obj in annotation.get("objects", [])),
            "",
        )
        destination = os.path.join(destination, label)
    download_file(path, destination, client=client)
    if task != "classification":
        save_annotation(annotation, folder, classes, task)


def save_config(dataset, classes):
    """Write the dataset configuration consumed by Ultralytics."""
    yaml_content = f"""
    train: {os.path.join(os.getcwd(), dataset, 'train/images')}
    val: {os.path.join(os.getcwd(), dataset, 'val/images')}
    test: {os.path.join(os.getcwd(), dataset, 'test/images')}
    names: {classes}
    """
    save_text(os.path.join(dataset, "data.yaml"), yaml_content)


def _normalize_split_options(split_options=None):
    return {**DEFAULT_SPLIT_OPTIONS, **(split_options or {})}


def split_dataset(annotations, split_options=None):
    """Split records with a deterministic, class-balanced policy."""
    options = _normalize_split_options(split_options)
    return split_annotation_records(annotations, **options)


def dataset_split_summary(dataset, split_options=None):
    """Return the split preview data used by Studio and training clients."""
    options = _normalize_split_options(split_options)
    splits = split_annotation_records(dataset.annotations, **options)
    return summarize_split_records(
        splits,
        classes=getattr(dataset, "classes", None),
        ratios={key: options[key] for key in SPLIT_RATIO_KEYS},
    )


def prepare_dataset(dataset, force=False, callback=None, split_options=None):
    """Download and split a dataset, optionally reporting each file."""
    if dataset.prune_orphaned_annotations():
        dataset.save()
    client = _resolve_client(getattr(dataset, "client", None))
    if force or not os.path.exists(dataset.project):
        if force:
            # These are generated training artifacts. Remove only their
            # explicit split directories so a changed UI split cannot retain
            # files from a previous partition.
            for split_name in DATASET_SPLITS:
                split_path = os.path.join(dataset.project, split_name)
                if os.path.isdir(split_path):
                    shutil.rmtree(split_path)
        annotations = dataset.annotations
        split_values = split_dataset(annotations, split_options)
        splits = list(zip(DATASET_SPLITS, split_values))
        total = sum(len(split_annotations) for _, split_annotations in splits)

        def work_items():
            for split_name, split_annotations in splits:
                folder = os.path.join(dataset.project, split_name)
                for annotation in split_annotations:
                    yield annotation, folder

        def report(current, annotation):
            if callback:
                callback({
                    "current": current,
                    "total": total,
                    "filename": annotation.get("filename", "image"),
                })

        if callback:
            callback({
                "current": 0,
                "total": total,
                "filename": "Starting download",
            })
        worker_count = min(DEFAULT_MAX_REMOTE_WORKERS, max(1, total))
        current = 0
        progress_bar = (
            tqdm(total=total, desc="Downloading images")
            if callback is None else None
        )
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                def save_item(item):
                    annotation, folder = item
                    return save_data(
                        annotation,
                        folder,
                        dataset.classes,
                        dataset.task,
                        client,
                        remote_path=posixpath.join(
                            str(dataset.project).strip("/"),
                            annotation.get("path") or annotation["filename"],
                        ),
                    )

                for (annotation, _folder), _result in bounded_as_completed(
                    executor,
                    save_item,
                    work_items(),
                    max_pending=worker_count * 2,
                ):
                    current += 1
                    report(current, annotation)
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        if normalize_task(dataset.task) != "classification":
            save_config(dataset.project, dataset.classes)


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
        checkpoint=None,
    ):
        task = normalize_task(task)
        if task not in TRAINING_TASKS:
            raise ValueError(f"Unsupported training task: {task}")
        self.project = project
        self.task = task
        self.classes = classes
        self.model_size = normalize_model_size(model_size)
        self.pbar = None
        imgsz = imgsz or (224 if task == "classification" else 640)
        if task == "classification":
            from . import classify

            self.model = classify.Model(client=client, model_size=self.model_size)
        else:
            from . import detect

            detect_kwargs = {
                "imgsz": imgsz,
                "client": client,
                "model_size": self.model_size,
            }
            if checkpoint is not None:
                detect_kwargs["checkpoint"] = checkpoint
            self.model = detect.Model(
                to_ultralytics_task(task),
                **detect_kwargs,
            )

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

    def save(self, device="cpu") -> Dict[str, Any]:
        return self.model.save(self.project, self.classes, device=device)

    def run(self, img):
        return self.model.run(img)

    def compile(
        self,
        device="hailo8",
        version=None,
        calibration_data=None,
        fraction=None,
        imgsz=None,
        conf=None,
        iou=None,
    ):
        if self.task not in ("detection", "segmentation"):
            raise NotImplementedError(
                "Hailo compilation is only implemented for Ultralytics "
                "detection and segmentation models."
            )
        return self.model.compile(
            self.project,
            self.classes,
            device=device,
            version=version,
            calibration_data=calibration_data,
            fraction=fraction,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

def train_dataset(
    project, dataset, epochs, training_callback, is_cancelled,
    model_size="small", split_options=None,
):
    """Prepare, train, validate, and save a dataset model."""
    from . import ModelTrainer, prepare_dataset

    task = normalize_task(dataset.task)
    trainer = None
    try:
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Preparing dataset"})
        prepare_dataset(
            dataset,
            force=split_options is not None,
            split_options=split_options,
            callback=lambda event: training_callback(
                {"stage": "Preparing dataset", **event}
            ),
        )
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Loading model"})
        trainer_kwargs = {"client": getattr(dataset, "client", None)}
        if model_size != "small":
            trainer_kwargs["model_size"] = model_size
        trainer = ModelTrainer(project, task, dataset.classes, **trainer_kwargs)

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
        if is_cancelled():
            raise RuntimeError("Training canceled")
        training_callback({"stage": "Exporting model"})
        saved_model = trainer.save()
        return {
            "stats": stats,
            "epochs": epochs,
            "task": task,
            "saved_model": saved_model,
        }
    finally:
        del trainer
        gc.collect()


__all__ = [
    "ModelTrainer",
    "dataset_split_summary",
    "download_file",
    "prepare_dataset",
    "save_annotation",
    "save_config",
    "save_data",
    "split_dataset",
    "train_dataset",
]
