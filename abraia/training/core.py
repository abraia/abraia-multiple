"""Dependency-free dataset state and annotation metadata helpers."""

from ..tasks import normalize_task


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
