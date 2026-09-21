"""Dependency-light remote dataset primitives.

This module owns dataset state and the small remote-loading contract shared by
the standard-image and multispectral adapters.  It intentionally does not
belong to :mod:`abraia.training`: datasets are also consumed by Studio and
the ``multiple`` package before any training operation is involved.
"""

from concurrent.futures import ThreadPoolExecutor

from .tasks import normalize_task
from .utils import url_path


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

    def save(self):
        """Persist the dataset annotations through the injected client."""
        self.client.save_json(f"{self.project}/annotations.json", self.annotations)
        self._update_annotated()


__all__ = ["DatasetBase", "RemoteDataset"]
