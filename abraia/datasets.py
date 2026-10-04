"""Dependency-light remote dataset primitives.

This module owns dataset state and the small remote-loading contract shared by
the standard-image and multispectral adapters.  It intentionally does not
belong to :mod:`abraia.training`: datasets are also consumed by Studio and
the ``multiple`` package before any training operation is involved.
"""

import copy
import os
from concurrent.futures import ThreadPoolExecutor

from .tasks import normalize_task
from .utils import url_path


def canonical_filename(value):
    """Return a comparable basename from a path or display label."""
    if value is None:
        return ""
    lines = str(value).splitlines()
    return os.path.basename(lines[0].strip()) if lines else ""


def image_name_key(image):
    """Return the case-insensitive display-name key used by image lists."""
    name = image.get("name")
    if name:
        return str(name).casefold()
    return os.path.basename(str(image.get("path") or "")).casefold()


DEFAULT_IMAGE_SORT_ORDER = "name_asc"


def _sortable_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1


def _sortable_date(value):
    if hasattr(value, "timestamp"):
        return 0, value.timestamp()
    if isinstance(value, (int, float)):
        return 0, float(value)
    return 1, str(value or "")


def sort_images(images, sort_order=DEFAULT_IMAGE_SORT_ORDER):
    """Return image records in the requested stable display order."""
    records = list(images or ())
    sort_order = str(sort_order or DEFAULT_IMAGE_SORT_ORDER)
    rules = {
        "name_asc": (image_name_key, False),
        "name_desc": (image_name_key, True),
        "date_newest": (lambda image: _sortable_date(image.get("date")), True),
        "date_oldest": (lambda image: _sortable_date(image.get("date")), False),
        "size_largest": (lambda image: _sortable_number(image.get("size")), True),
        "size_smallest": (lambda image: _sortable_number(image.get("size")), False),
    }
    rule = rules.get(sort_order)
    if rule is None:
        return records
    key, reverse = rule
    return sorted(records, key=key, reverse=reverse)


class DatasetBase:
    """Common dataset state shared by remote dataset adapters."""

    def __init__(self, project):
        self.project = project
        self.annotations = []
        self.classes = []
        self.task = ""
        self.images = []
        self.annotated = False

    def _update_annotated(self):
        annotated_filenames = {
            annotation.get("filename")
            for annotation in self.annotations
            if isinstance(annotation, dict)
        }
        self.annotated = bool(self.images) and all(
            image.get("name") in annotated_filenames
            for image in self.images
            if isinstance(image, dict)
        )

    def set_image_order(self, images=None, sort_order=DEFAULT_IMAGE_SORT_ORDER):
        """Apply the shared display order to this dataset image list."""
        source = self.images if images is None else images
        self.images[:] = sort_images(source, sort_order)
        return self.images

    def replace_annotations(self, annotations):
        """Replace annotations and refresh metadata derived from them."""
        updated = copy.deepcopy(annotations or [])
        changed = updated != (self.annotations or [])
        self.annotations = updated
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return changed

    def upsert_annotation(self, filename, objects):
        """Update an image annotation, or append a record for a listed image."""
        key = canonical_filename(filename)
        annotations = self.annotations or []
        self.annotations = annotations
        for annotation in annotations:
            if (
                isinstance(annotation, dict)
                and canonical_filename(annotation.get("filename")) == key
            ):
                changed = annotation.get("objects") != objects
                annotation["objects"] = copy.deepcopy(objects)
                self.classes, self.task = self._process_annotations(self.annotations)
                self._update_annotated()
                return changed
        image = next(
            (
                record
                for record in self.images or []
                if canonical_filename(record.get("name") or record.get("path"))
                == key
            ),
            None,
        )
        if image is None:
            return False
        annotations.append(
            {
                "url": image.get("url"),
                "filename": image.get("name") or key,
                "objects": copy.deepcopy(objects),
            }
        )
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return True

    def add_annotations(self, annotations):
        """Append annotation records and refresh derived metadata."""
        additions = copy.deepcopy(list(annotations or []))
        if not additions:
            return False
        self.annotations = list(self.annotations or []) + additions
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return True

    def prune_orphaned_annotations(self):
        """Remove annotations whose image is absent from the dataset."""
        image_names = {
            canonical_filename(image.get("name") or image.get("path"))
            for image in self.images or []
            if isinstance(image, dict)
        }
        remaining = [
            annotation
            for annotation in self.annotations or []
            if isinstance(annotation, dict)
            and canonical_filename(annotation.get("filename")) in image_names
        ]
        removed = len(self.annotations or []) - len(remaining)
        if removed:
            self.replace_annotations(remaining)
        return removed

    @staticmethod
    def _process_annotations(annotations):
        """Return class names and the canonical task inferred from objects."""
        # Preserve first-seen order because this list is also the class-ID
        # mapping written into detection labels and model metadata.
        labels = {}
        has_class_labels = False
        has_boxes = False
        has_polygons = False
        for annotation in annotations or []:
            if not isinstance(annotation, dict):
                continue
            for obj in annotation.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                label = obj.get("label")
                if label:
                    labels.setdefault(label, None)
                    has_class_labels = True
                if obj.get("polygon") is not None:
                    has_polygons = True
                elif obj.get("box") is not None:
                    has_boxes = True
        task = (
            "segmentation"
            if has_polygons
            else "detection"
            if has_boxes
            else "classification"
            if has_class_labels
            else ""
        )
        return list(labels), normalize_task(task)


class RemoteDataset(DatasetBase):
    """Common remote dataset behavior for standard and spectral datasets."""

    def __init__(self, project, client):
        super().__init__(project)
        if client is None:
            raise ValueError("A remote dataset requires a client")
        self.client = client

    def _load_annotations(self, project):
        annotations = self.client.load_json(f"{project}/annotations.json")
        for annotation in annotations:
            filename = annotation.get("filename", "")
            path = f"{project}/{filename}"
            annotation["path"] = path
            annotation["url"] = url_path(f"{self.client.userid}/{path}")
        return annotations

    def _select_images(self, files):
        """Return displayable image records from a remote file listing."""
        return files

    def _list_images(self, project):
        files = self.client.list_files(f"{project}/")[0]
        images = self._select_images(files)
        for image in images:
            image["url"] = url_path(f"{self.client.userid}/{image['path']}")
        return images

    def copy_for_update(self):
        """Make an isolated working copy for a background dataset mutation."""
        dataset = copy.copy(self)
        dataset.images = list(self.images or [])
        dataset.annotations = copy.deepcopy(self.annotations or [])
        dataset.classes = list(self.classes or [])
        return dataset

    def refresh_images(self):
        """Refresh only image records while retaining this dataset object."""
        self.images = self._list_images(self.project)
        self._update_annotated()
        return self

    def load_contents(self, *, parallel=False):
        """Populate annotations, images, classes, and task in one place."""
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                annotations_future = executor.submit(
                    self._load_annotations, self.project
                )
                images_future = executor.submit(self._list_images, self.project)
                self.annotations = annotations_future.result()
                self.images = images_future.result()
        else:
            self.annotations = self._load_annotations(self.project)
            self.images = self._list_images(self.project)
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()
        return self

    def clear_contents(self):
        """Reset a dataset when its remote project is not available."""
        self.annotations = []
        self.images = []
        self.classes = []
        self.task = ""
        self._update_annotated()
        return self

    def delete_images(
        self,
        image_paths,
        *,
        file_paths=None,
        progress_callback=None,
        is_cancelled=None,
    ):
        """Delete remote files and update this dataset's in-memory state.

        file_paths may include companion files that are not image records.
        """
        image_paths = list(dict.fromkeys(path for path in image_paths if path))
        paths = list(dict.fromkeys(
            path for path in (file_paths if file_paths is not None else image_paths)
            if path
        ))
        is_cancelled = is_cancelled or (lambda: False)
        if progress_callback:
            progress_callback(0, len(paths), "Preparing image deletion", False)
        if not paths:
            return self

        for index, path in enumerate(paths, start=1):
            if is_cancelled():
                raise RuntimeError("Operation canceled")
            if progress_callback:
                progress_callback(index - 1, len(paths), path, False)
            self.client.remove_file(path)
            folder, name = os.path.split(path)
            try:
                self.client.remove_file(os.path.join(folder, f"tb_{name}"))
            except Exception:
                pass
            if progress_callback:
                progress_callback(index, len(paths), path, True)

        removed_paths = {str(path).strip("/") for path in image_paths}
        removed_names = {
            canonical_filename(path)
            for path in image_paths
            if str(path).splitlines()
        }
        remaining_images = []
        for image in self.images or []:
            image_path = str(image.get("path") or "").strip("/")
            image_name = os.path.basename(str(image.get("name") or "").strip("/"))
            if image_path in removed_paths or (
                not image_path and image_name in removed_names
            ):
                continue
            remaining_images.append(image)
        self.images = remaining_images

        annotations = self.annotations or []
        remaining_annotations = [
            annotation
            for annotation in annotations
            if not (
                isinstance(annotation, dict)
                and canonical_filename(annotation.get("filename"))
                in removed_names
            )
        ]
        if len(remaining_annotations) != len(annotations):
            self.replace_annotations(remaining_annotations)
            self.save()
        else:
            self._update_annotated()
        return self

    def save(self):
        """Persist annotations and refresh their derived dataset metadata."""
        self.client.save_json(f"{self.project}/annotations.json", self.annotations)
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()


__all__ = [
    "DEFAULT_IMAGE_SORT_ORDER",
    "DatasetBase",
    "RemoteDataset",
    "canonical_filename",
    "image_name_key",
    "sort_images",
]
