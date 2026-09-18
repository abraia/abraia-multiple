"""Shared dataset state and remote dataset helpers."""

from ..client import Abraia
from ..tasks import normalize_task
from ..utils import url_path


def _resolve_client(client=None):
    """Return an injected client or create one lazily for this operation."""
    return client if client is not None else Abraia()


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
        # Preserve the first-seen order because this list is also the class-ID
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
    """Common remote dataset behavior for standard and spectral datasets.

    Subclasses only need to decide which files represent displayable images.
    Format-specific scene grouping and metadata inspection remain in the
    multispectral package.
    """

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
