"""Remote dataset state shared by the SDK's dataset adapters.

Dataset operations are independent of training because Studio and the
``multiple`` package use them before any training step.
"""

import copy
import posixpath
from concurrent.futures import ThreadPoolExecutor

from ..client import Abraia
from ..tasks import normalize_task
from ..utils import url_path


def canonical_filename(value):
    """Return a comparable basename from a path or display label."""
    if value is None:
        return ""
    lines = str(value).splitlines()
    return posixpath.basename(
        lines[0].strip().replace("\\", "/")
    ) if lines else ""


def image_name_key(image):
    """Return the case-insensitive display-name key used by image lists."""
    name = image.get("name")
    if name:
        return str(name).casefold()
    return canonical_filename(image.get("path")).casefold()


DEFAULT_IMAGE_SORT_ORDER = "name_asc"


def _resolve_client(client=None):
    """Return an injected client or create one for this dataset operation."""
    return client if client is not None else Abraia()


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


class RemoteDataset:
    """Dataset state and remote behavior shared by dataset adapters."""

    def __init__(self, project, client):
        if client is None:
            raise ValueError("A remote dataset requires a client")
        self.project = str(project or "").strip("/")
        if not self.project:
            raise ValueError("A remote dataset requires a project")
        self.client = client
        self.annotations = []
        self.classes = []
        self.task = ""
        self.images = []
        self.annotated = False
        self.curation_findings = None

    def _invalidate_curation_findings(self):
        """Clear findings when the dataset contents they describe change."""
        self.curation_findings = None

    def _relative_path(self, path):
        """Normalize a dataset path and ensure it stays inside this project."""
        path = str(path or "").strip().replace("\\", "/")
        if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
            raise ValueError(f"Path must be relative to project {self.project!r}")
        path = posixpath.normpath(path)
        if path.startswith(f"{self.project}/"):
            raise ValueError(
                f"Pass paths relative to project {self.project!r}, without its prefix"
            )
        if path in ("", ".", "..") or path.startswith("../"):
            raise ValueError(f"Path must be relative to project {self.project!r}")
        if "/" in path:
            raise ValueError("Dataset files must be directly inside the project")
        return path

    def _listed_path(self, path):
        """Convert a client listing key to the dataset-relative form."""
        path = str(path or "").strip().replace("\\", "/").lstrip("/")
        prefix = f"{self.project}/"
        if path.startswith(prefix):
            path = path[len(prefix):]
        path = posixpath.normpath(path)
        if path in ("", ".", "..") or path.startswith("../"):
            raise ValueError(f"Path must be relative to project {self.project!r}")
        return path

    def _image_path(self, image):
        """Return an image record's project-relative path."""
        path = image.get("path") or image.get("name")
        return self._relative_path(path) if path else ""

    def _remote_path(self, path):
        """Build a remote object key from a path relative to this project."""
        return posixpath.join(self.project, self._relative_path(path))

    def _update_annotated(self):
        annotated_paths = {
            self._annotation_path(annotation)
            for annotation in self.annotations or []
            if isinstance(annotation, dict)
        }
        images = [
            image for image in self.images or [] if isinstance(image, dict)
        ]
        image_paths = [self._image_path(image) for image in images]
        self.annotated = bool(images) and all(
            path in annotated_paths for path in image_paths
        )

    def _refresh_annotation_metadata(self):
        self.classes, self.task = self._process_annotations(self.annotations)
        self._update_annotated()

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
        if changed:
            self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return changed

    def upsert_annotation(self, filename, objects):
        """Update an image annotation, or append a record for a listed image."""
        key = self._relative_path(filename)
        annotations = self.annotations or []
        self.annotations = annotations
        for annotation in annotations:
            if (
                isinstance(annotation, dict)
                and self._annotation_path(annotation) == key
            ):
                changed = annotation.get("objects") != objects
                annotation["objects"] = copy.deepcopy(objects)
                if changed:
                    self._invalidate_curation_findings()
                self._refresh_annotation_metadata()
                return changed
        image = next((
            record for record in self.images or []
            if self._image_path(record) == key
        ), None)
        if image is None:
            return False
        image_path = self._image_path(image)
        annotations.append(
            {
                "url": image.get("url"),
                "filename": image.get("name") or canonical_filename(key),
                "path": image_path,
                "objects": copy.deepcopy(objects),
            }
        )
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return True

    def add_annotations(self, annotations):
        """Append annotation records and refresh derived metadata."""
        additions = copy.deepcopy(list(annotations or []))
        if not additions:
            return False
        self.annotations = list(self.annotations or []) + additions
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return True

    def prune_orphaned_annotations(self):
        """Remove annotations whose image is absent from the dataset."""
        image_paths = {
            self._image_path(image)
            for image in self.images or []
            if isinstance(image, dict)
        }
        remaining = [
            annotation
            for annotation in self.annotations or []
            if isinstance(annotation, dict)
            and self._annotation_path(annotation) in image_paths
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

    def _load_annotations(self, project):
        annotations = self.client.load_json(self._remote_path("annotations.json"))
        for annotation in annotations:
            filename = annotation.get("filename", "")
            if filename:
                path = self._relative_path(filename)
                annotation["path"] = path
                annotation["url"] = url_path(
                    f"{self.client.userid}/{self._remote_path(path)}"
                )
        return annotations

    def _select_images(self, files):
        """Return displayable image records from a remote file listing."""
        return files

    def _list_images(self, project):
        files = self.client.list_files(f"{project}/")[0]
        images = self._select_images(files)
        flat_images = []
        for image in images:
            path = image.get("path") or image.get("name")
            relative_path = self._listed_path(path)
            if "/" in relative_path:
                continue
            image["path"] = self._relative_path(relative_path)
            image["url"] = url_path(
                f"{self.client.userid}/{self._remote_path(image['path'])}"
            )
            flat_images.append(image)
        return flat_images

    def copy_for_update(self):
        """Make an isolated working copy for a background dataset mutation."""
        dataset = copy.copy(self)
        dataset.images = list(self.images or [])
        dataset.annotations = copy.deepcopy(self.annotations or [])
        dataset.classes = list(self.classes or [])
        dataset.curation_findings = copy.deepcopy(self.curation_findings)
        return dataset

    def refresh_images(self):
        """Refresh only image records while retaining this dataset object."""
        self.images = self._list_images(self.project)
        self._invalidate_curation_findings()
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
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return self

    def clear_contents(self):
        """Reset a dataset when its remote project is not available."""
        self.annotations = []
        self.images = []
        self._invalidate_curation_findings()
        self.classes = []
        self.task = ""
        self.annotated = False
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
        image_paths = list(dict.fromkeys(
            self._relative_path(path) for path in image_paths if path
        ))
        paths = list(dict.fromkeys(
            self._relative_path(path)
            for path in (file_paths if file_paths is not None else image_paths)
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
            remote_path = self._remote_path(path)
            self.client.remove_file(remote_path)
            folder, name = posixpath.split(remote_path)
            try:
                self.client.remove_file(posixpath.join(folder, f"tb_{name}"))
            except Exception:
                pass
            if progress_callback:
                progress_callback(index, len(paths), path, True)

        removed_paths = set(image_paths)
        remaining_images = []
        for image in self.images or []:
            image_path = image.get("path") or image.get("name")
            if image_path and self._relative_path(image_path) in removed_paths:
                continue
            remaining_images.append(image)
        self.images = remaining_images
        if removed_paths:
            self._invalidate_curation_findings()

        annotations = self.annotations or []
        remaining_annotations = [
            annotation
            for annotation in annotations
            if not (
                isinstance(annotation, dict)
                and self._annotation_removed(annotation, removed_paths)
            )
        ]
        if len(remaining_annotations) != len(annotations):
            self.replace_annotations(remaining_annotations)
            self.save()
        else:
            self._refresh_annotation_metadata()
        return self

    def _annotation_path(self, annotation):
        """Return an annotation's project-relative image path."""
        path = annotation.get("path") or annotation.get("filename")
        return self._relative_path(path) if path else ""

    def _annotation_removed(self, annotation, removed_paths):
        """Return whether an annotation belongs to a removed image."""
        path = self._annotation_path(annotation)
        return path in removed_paths

    def save(self):
        """Persist annotations and refresh their derived dataset metadata."""
        self.client.save_json(self._remote_path("annotations.json"), self.annotations)
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()


class DatasetService:
    """Coordinate remote project operations with a dataset adapter."""

    def __init__(self, client, dataset_loader):
        self.client = client
        self.dataset_loader = dataset_loader

    def create_empty_dataset(self, project):
        self.client.save_json(f"{project}/annotations.json", [])
        return self.dataset_loader(project, validate=False)

    def save_annotation(self, project, filename, objects, dataset=None):
        dataset = dataset or self.dataset_loader(project)
        if dataset.upsert_annotation(filename, objects):
            dataset.save()
        return dataset

    def save_annotations(self, project, annotations, dataset=None):
        """Replace and persist the dataset annotation table in one update."""
        dataset = dataset or self.dataset_loader(project)
        dataset.replace_annotations(annotations)
        dataset.save()
        return dataset

    def delete_image(self, project, path, dataset=None):
        """Delete one remote image through the dataset mutation API."""
        return self.delete_images(project, [path], dataset=dataset)

    def delete_images(
        self, project, paths, progress_callback=None, is_cancelled=None,
        dataset=None,
    ):
        image_paths = list(dict.fromkeys(path for path in paths if path))
        is_cancelled = is_cancelled or (lambda: False)
        if not image_paths:
            if progress_callback:
                progress_callback(0, 0, "Preparing image deletion", False)
            return None
        dataset = dataset or self.dataset_loader(project)
        return dataset.delete_images(
            image_paths,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def delete_dataset(self, project, progress_callback=None, is_cancelled=None):
        """Remove every remote file belonging to a dataset project."""
        root = str(project or "").strip("/")
        if not root:
            raise ValueError("A dataset project is required")
        is_cancelled = is_cancelled or (lambda: False)
        pending = [root]
        visited = set()
        files = []

        def belongs_to_root(path):
            normalized = str(path or "").strip("/")
            return normalized == root or normalized.startswith(f"{root}/")

        while pending:
            folder = pending.pop(0).strip("/")
            if not folder or folder in visited:
                continue
            visited.add(folder)
            entries, child_folders = self.client.list_files(f"{folder}/")
            for entry in entries or []:
                path = entry.get("path") or posixpath.join(
                    folder, entry.get("name", "")
                )
                if belongs_to_root(path):
                    files.append(path)
            for child in child_folders or []:
                path = child.get("path") or posixpath.join(
                    folder, child.get("name", "")
                )
                if belongs_to_root(path):
                    pending.append(path)

        if progress_callback:
            progress_callback(0, len(files), "Preparing deletion", False)
        for index, path in enumerate(files, start=1):
            if is_cancelled():
                raise RuntimeError("Operation canceled")
            if progress_callback:
                progress_callback(index - 1, len(files), path, False)
            self.client.remove_file(path)
            if progress_callback:
                progress_callback(index, len(files), path, True)
        return root

__all__ = [
    "DEFAULT_IMAGE_SORT_ORDER",
    "DatasetService",
    "RemoteDataset",
    "canonical_filename",
    "image_name_key",
    "sort_images",
]
