"""Local dataset splitting, preparation, and training-file generation."""

import hashlib
import json
import os
import posixpath
import shutil
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm

from ..tasks import normalize_task
from ..utils.filesystem import save_text
from ..utils.concurrency import DEFAULT_MAX_REMOTE_WORKERS, bounded_as_completed
from .remote import _resolve_client
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


def _split_cache_key(dataset, annotations, split_options, task):
    """Return a stable key for the annotations and settings used in a split."""
    payload = {
        "annotations": annotations,
        "classes": dataset.classes,
        "task": task,
        "split_options": _normalize_split_options(split_options),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare_dataset(dataset, force=False, callback=None, split_options=None):
    """Download and split a dataset, reusing files while its inputs are unchanged."""
    if dataset.prune_orphaned_annotations():
        dataset.save()
    project = dataset.project
    task = normalize_task(dataset.task)
    cache_path = os.path.join(project, ".training-split.json")
    annotations = dataset.annotations
    cache_key = _split_cache_key(dataset, annotations, split_options, task)
    if not force and os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as cache_file:
            cached_key = json.load(cache_file).get("key")
        if cached_key == cache_key and all(
            os.path.isdir(os.path.join(project, split_name))
            for split_name in DATASET_SPLITS
        ):
            return

    client = _resolve_client(getattr(dataset, "client", None))
    # Split folders are generated artifacts; replace only these when inputs
    # changed, leaving unrelated files in the project untouched.
    for split_name in DATASET_SPLITS:
        split_path = os.path.join(project, split_name)
        if os.path.isdir(split_path):
            shutil.rmtree(split_path)
    if os.path.isfile(cache_path):
        os.remove(cache_path)
    os.makedirs(project, exist_ok=True)
    split_values = split_dataset(annotations, split_options)
    splits = list(zip(DATASET_SPLITS, split_values))
    total = sum(len(split_annotations) for _, split_annotations in splits)

    for split_name in DATASET_SPLITS:
        os.makedirs(os.path.join(project, split_name), exist_ok=True)

    def work_items():
        for split_name, split_annotations in splits:
            folder = os.path.join(project, split_name)
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
                    task,
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
    if task != "classification":
        save_config(project, dataset.classes)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump({"key": cache_key}, cache_file)
