"""Dataset preparation and model-training orchestration."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from PIL import Image
from tqdm import tqdm

from ..tasks import (
    TRAINING_TASKS,
    normalize_model_size,
    normalize_task,
    to_ultralytics_task,
)
from ..utils import save_text
from .core import _resolve_client
from .dataset import download_file
from .ops import train_test_split


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


def save_data(annotation, folder, classes, task, client=None):
    """Download one annotation image and write its training label."""
    task = normalize_task(task)
    path = annotation["path"]
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


def split_dataset(annotations):
    """Split annotated records while retaining background images in training."""
    backgrounds = [annotation for annotation in annotations if not annotation.get("objects")]
    annotated = [annotation for annotation in annotations if annotation.get("objects")]
    if not annotated:
        return backgrounds, [], []
    train, test = train_test_split(annotated, test_size=0.3, random_state=42)
    if test:
        validation, test = train_test_split(test, test_size=0.5, random_state=42)
    else:
        validation, test = [], []
    train.extend(backgrounds)
    return train, validation, test


def prepare_dataset(dataset, force=False, callback=None):
    """Download and split a dataset, optionally reporting each file."""
    client = _resolve_client(getattr(dataset, "client", None))
    if force or not os.path.exists(dataset.project):
        annotations = dataset.annotations
        dataset_path = f"{dataset.project}/dataset.json"
        if client.check_file(dataset_path):
            filenames = client.load_json(dataset_path)
            annotations = [
                annotation
                for annotation in annotations
                if annotation.get("filename") in filenames
            ]
        splits = list(zip(("train", "val", "test"), split_dataset(annotations)))
        all_annotations, all_folders = [], []
        for split_name, split_annotations in splits:
            folder = os.path.join(dataset.project, split_name)
            all_annotations.extend(split_annotations)
            all_folders.extend([folder] * len(split_annotations))
        total = len(all_annotations)

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
        work = list(zip(all_annotations, all_folders))
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    save_data,
                    annotation,
                    folder,
                    dataset.classes,
                    dataset.task,
                    client,
                ): annotation
                for annotation, folder in work
            }
            completed = as_completed(futures)
            if callback is None:
                completed = tqdm(completed, total=total, desc="Downloading images")
            for current, future in enumerate(completed, start=1):
                future.result()
                report(current, futures[future])
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

            self.model = detect.Model(
                to_ultralytics_task(task),
                imgsz=imgsz,
                client=client,
                model_size=self.model_size,
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
        epochs = epochs or (30 if self.task == "classification" else 300)
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

    def test(self, split: str = "val", is_cancelled=None) -> Dict[str, Any]:
        if is_cancelled is None:
            return self.model.test(split=split)
        return self.model.test(split=split, is_cancelled=is_cancelled)

    def save(self, device="cpu") -> None:
        self.model.save(self.project, self.classes, device=device)

    def run(self, img):
        return self.model.run(img)

    def compile(self, device="hailo8"):
        if self.task != "detection":
            raise NotImplementedError(
                "Model compilation is only implemented for detection models."
            )
        self.model.compile(self.project, self.classes, device=device)


__all__ = [
    "ModelTrainer",
    "prepare_dataset",
    "save_annotation",
    "save_config",
    "save_data",
    "split_dataset",
]
