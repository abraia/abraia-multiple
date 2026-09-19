"""Remote dataset loading with compatibility exports for training helpers."""

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ..utils import load_image, load_url
from ..tasks import normalize_task
from ..datasets import RemoteDataset
from .auto_annotation import Annotator, annotate_image as _annotate_image
from .core import _resolve_client
from .image_search import (
    convert_to_jpg,
    dhash,
    download_page,
    save_image_file,
    scan_bing_page,
    scan_google_page,
    search_bing,
    search_google,
    search_images,
)


def annotate_image(
    image_data,
    classes,
    segment=False,
    annotator=None,
    include_empty=False,
):
    """Compatibility wrapper preserving patchable dataset-level loaders."""
    return _annotate_image(
        image_data,
        classes,
        segment=segment,
        annotator=annotator,
        include_empty=include_empty,
        image_loader=load_image,
        url_loader=load_url,
    )


def download_file(path, folder, client=None):
    """Download one remote file when it is not already present locally."""
    client = _resolve_client(client)
    destination = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(destination):
        client.download_file(path, destination)
    return destination


def list_datasets(client=None):
    """Return remote folders containing an annotations file."""
    client = _resolve_client(client)
    folders = client.list_files()[1]

    def has_annotations(folder):
        try:
            return client.check_file(f"{folder['name']}/annotations.json")
        except Exception:
            return False

    if not folders:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(folders))) as executor:
        valid = executor.map(has_annotations, folders)
        return [
            folder["name"]
            for folder, is_valid in zip(folders, valid)
            if is_valid
        ]


def list_models(project, client=None):
    """Return ONNX model names stored in a remote project."""
    client = _resolve_client(client)
    files = client.list_files(f"{project}/")[0]
    return [file_data["name"] for file_data in files if file_data["name"].endswith(".onnx")]


def list_model_records(project, client=None):
    """Return model names with optional persisted evaluation metadata."""
    client = _resolve_client(client)
    records = []
    for name in list_models(project, client=client):
        record = {"name": name, "metrics": {}}
        if hasattr(client, "load_json"):
            stem = os.path.splitext(str(name))[0]
            try:
                metadata = client.load_json(f"{project}/{stem}.json")
            except Exception:
                metadata = None
            if isinstance(metadata, dict):
                record.update({
                    "metrics": metadata.get("metrics") or {},
                    "classes": metadata.get("classes") or [],
                    "task": normalize_task(metadata.get("task")),
                })
        records.append(record)
    return records


class Dataset(RemoteDataset):
    """Remote dataset containing standard image records."""

    def __init__(self, project, client=None):
        super().__init__(project, _resolve_client(client))

    def load(self, validate=True):
        if validate and self.project not in list_datasets(self.client):
            self.annotations = []
            self.images = []
            self.classes = []
            self.task = ""
            self._update_annotated()
            return self

        with ThreadPoolExecutor(max_workers=2) as executor:
            annotations_future = executor.submit(self._load_annotations, self.project)
            images_future = executor.submit(self._list_images, self.project)
            self.annotations = annotations_future.result()
            self.images = images_future.result()
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return self

    def _select_images(self, files):
        return [
            file_data
            for file_data in files
            if file_data["type"] in ["image/jpeg", "image/png"]
        ]

    def annotate(self, label, segment=False, callback=None):
        annotated_filenames = {annotation["filename"] for annotation in self.annotations}
        images = [
            image
            for image in self.images
            if image["name"] not in annotated_filenames
        ]
        annotator = Annotator(segment=segment)
        progress = tqdm(images) if callback is None else None
        iterable = progress if progress else images
        for index, row in enumerate(iterable):
            if progress:
                progress.set_description(f"Annotating {row['name']}")
            annotation = annotate_image(
                row,
                [label],
                segment=segment,
                annotator=annotator,
                include_empty=True,
            )
            self.annotations.append(annotation)
            self.save()
            if callback:
                callback({
                    "current": index + 1,
                    "total": len(images),
                    "filename": row["name"],
                })
        if progress:
            progress.close()
        self._update_annotated()
        return self.annotations

    def save(self):
        self.client.save_json(f"{self.project}/annotations.json", self.annotations)
        self._update_annotated()


def load_dataset(project, validate=True, client=None):
    """Load a remote standard-image dataset."""
    return Dataset(project, client=client).load(validate=validate)


__all__ = [
    "Annotator",
    "Dataset",
    "annotate_image",
    "convert_to_jpg",
    "dhash",
    "download_file",
    "download_page",
    "list_datasets",
    "list_models",
    "list_model_records",
    "load_dataset",
    "save_image_file",
    "scan_bing_page",
    "scan_google_page",
    "search_bing",
    "search_google",
    "search_images",
]
