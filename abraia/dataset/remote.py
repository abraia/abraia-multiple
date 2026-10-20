"""Remote image dataset class and project-level operations."""

import posixpath
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping

from ..client import Abraia
from ..tasks import normalize_task
from ..utils.image import load_image
from ..utils.remote import load_url, url_path
from ..utils.concurrency import DEFAULT_MAX_REMOTE_WORKERS, bounded_map
from .annotation import Annotator, annotate_image as _annotate_image
from .splitting import DatasetSplit


def _resolve_client(client=None):
    """Return an injected client or create one for this dataset operation."""
    return client if client is not None else Abraia()


class Dataset:
    """Remote dataset containing standard image records."""

    def __init__(self, project, client=None):
        client = _resolve_client(client)
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
        self.split = DatasetSplit(self)

    def load(self, validate=True):
        """Load this dataset's images and annotations from the remote project."""
        if validate and self.project not in list_datasets(self.client):
            self.annotations = []
            self.images = []
            self.classes = []
            self.task = ""
            self.annotated = False
            self.curation_findings = None
            return self

        with ThreadPoolExecutor(max_workers=2) as executor:
            annotations = executor.submit(self._load_annotations)
            images = executor.submit(self._list_images)
            self.annotations = annotations.result()
            self.images = images.result()
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return self

    def annotate(
        self,
        label,
        segment=False,
        callback=None,
        *,
        task=None,
        include_empty=True,
        progress_callback=None,
        is_cancelled=None,
    ):
        """Annotate pending images, optionally reporting or canceling work."""
        from tqdm import tqdm

        annotated = {
            self._annotation_path(item)
            for item in self.annotations
            if isinstance(item, Mapping)
        }
        pending = [
            image for image in self.images
            if self._image_path(image) not in annotated
        ]
        progress = (
            tqdm(total=len(pending))
            if callback is None and progress_callback is None
            else None
        )
        is_cancelled = is_cancelled or (lambda: False)
        classification = normalize_task(task or self.task) == "classification"
        annotator = Annotator(segment=segment) if pending else None
        try:
            total = len(pending)
            for index, image_data in enumerate(pending, start=1):
                if is_cancelled():
                    break
                annotation = annotate_image(
                    image_data,
                    [label],
                    segment=segment,
                    annotator=annotator,
                    include_empty=include_empty,
                    classification=classification,
                )
                if annotation:
                    annotation["path"] = self._image_path(image_data)
                    self.annotations.append(annotation)
                self.save()
                if progress_callback:
                    progress_callback(
                        index,
                        total,
                        image_data["name"],
                        len(annotation.get("objects") or []) if annotation else 0,
                    )
                if progress:
                    progress.update(1)
                    progress.set_description(f"Annotating {image_data['name']}")
                if callback:
                    callback({
                        "current": index,
                        "total": total,
                        "filename": image_data["name"],
                    })
        finally:
            if annotator is not None:
                annotator.close()
            if progress:
                progress.close()
        return self.annotations

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

    def _refresh_annotation_metadata(self):
        self.classes, self.task = self._process_annotations(self.annotations)
        annotated_paths = {
            self._annotation_path(annotation)
            for annotation in self.annotations
            if isinstance(annotation, dict)
        }
        images = self.images
        image_paths = [self._image_path(image) for image in images]
        self.annotated = bool(images) and all(
            path in annotated_paths for path in image_paths
        )

    def replace_annotations(self, annotations):
        """Replace annotations and refresh metadata derived from them."""
        updated = list(annotations)
        changed = updated != self.annotations
        self.annotations = updated
        if changed:
            self._invalidate_curation_findings()
        self._refresh_annotation_metadata()
        return changed

    def prune_orphaned_annotations(self):
        """Remove annotations whose image is absent from the dataset."""
        image_paths = {
            self._image_path(image)
            for image in self.images
        }
        remaining = [
            annotation
            for annotation in self.annotations
            if isinstance(annotation, dict)
            and self._annotation_path(annotation) in image_paths
        ]
        removed = len(self.annotations) - len(remaining)
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
        for annotation in annotations:
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
        return list(labels), task

    def _load_annotations(self):
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

    def _list_images(self):
        files = self.client.list_files(f"{self.project}/")[0]
        flat_images = []
        for image in files:
            if image["type"] not in ("image/jpeg", "image/png"):
                continue
            path = image.get("path") or image.get("name")
            relative_path = self._listed_path(path)
            if "/" in relative_path:
                continue
            image["path"] = relative_path
            image["url"] = url_path(
                f"{self.client.userid}/{self._remote_path(image['path'])}"
            )
            flat_images.append(image)
        return flat_images

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
        paths = (
            image_paths
            if file_paths is None
            else list(dict.fromkeys(
                self._relative_path(path) for path in file_paths if path
            ))
        )
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
        for image in self.images:
            image_path = image.get("path") or image.get("name")
            if image_path and self._relative_path(image_path) in removed_paths:
                continue
            remaining_images.append(image)
        self.images = remaining_images
        if removed_paths:
            self._invalidate_curation_findings()

        annotations = self.annotations
        remaining_annotations = [
            annotation
            for annotation in annotations
            if not (
                isinstance(annotation, dict)
                and self._annotation_path(annotation) in removed_paths
            )
        ]
        if len(remaining_annotations) != len(annotations):
            self.annotations = remaining_annotations
            self.save()
        else:
            self._refresh_annotation_metadata()
        return self

    def _annotation_path(self, annotation):
        """Return an annotation's project-relative image path."""
        path = annotation.get("path") or annotation.get("filename")
        return self._relative_path(path) if path else ""

    def save(self):
        """Persist annotations and refresh their derived dataset metadata."""
        self.client.save_json(self._remote_path("annotations.json"), self.annotations)
        self._invalidate_curation_findings()
        self._refresh_annotation_metadata()

    def curate(
        self,
        work_dir=None,
        apply=False,
        confirm=None,
        progress_callback=None,
        is_cancelled=None,
        analyze_options=None,
    ):
        """Analyze dataset images and retain curation findings."""
        from .curation import analyze_curation

        project = self.project
        client = self.client
        images = self.images
        annotated = {
            self._annotation_path(annotation)
            for annotation in self.annotations
            if isinstance(annotation, Mapping)
        }
        records = []
        for image in images:
            record = dict(image)
            path = self._image_path(image)
            record["path"] = path
            records.append(record)
        protected = [
            record["path"] for record in records if record["path"] in annotated
        ]
        owner = str(getattr(client, "userid", "") or "").strip("/")
        report_progress = None
        if progress_callback:
            report_progress = (
                lambda current, total, detail, completed: progress_callback(
                    round(100 * current / max(total, 1)), 100, detail, completed
                )
            )
        report = analyze_curation(
            records,
            image_loader=lambda path: client.load_image(self._remote_path(path)),
            work_dir=work_dir,
            protected_paths=protected,
            cache_namespace=f"{owner}/{str(project).strip('/')}",
            progress_callback=report_progress,
            is_cancelled=is_cancelled,
            **(analyze_options or {}),
        )
        try:
            if apply:
                recommended_paths = report["recommended_paths"]
                if recommended_paths and confirm is not None and not confirm(
                    recommended_paths
                ):
                    return report
                paths = list(dict.fromkeys(
                    self._relative_path(path)
                    for path in recommended_paths
                ))
                if paths:
                    self.delete_images(paths, is_cancelled=is_cancelled)
                report["deleted"] = paths
        finally:
            self.curation_findings = report.get("findings", [])
        if progress_callback:
            progress_callback(100, 100, "Curation complete", True)
        return report


def annotate_image(
    image_data,
    classes,
    segment=False,
    annotator=None,
    include_empty=False,
    classification=False,
):
    return _annotate_image(
        image_data,
        classes,
        segment=segment,
        annotator=annotator,
        include_empty=include_empty,
        classification=classification,
        image_loader=load_image,
        url_loader=load_url,
    )


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
    worker_count = min(DEFAULT_MAX_REMOTE_WORKERS, len(folders))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        valid = bounded_map(
            executor, has_annotations, folders, max_pending=worker_count * 2
        )
        return [
            folder["name"]
            for folder, is_valid in zip(folders, valid)
            if is_valid
        ]


def load_dataset(project, validate=True, client=None):
    """Load a remote standard-image dataset."""
    return Dataset(project, client=client).load(validate=validate)


def delete_dataset(
    project, client=None, progress_callback=None, is_cancelled=None
):
    """Remove every remote file belonging to a dataset project."""
    client = _resolve_client(client)
    root = str(project or "").strip("/")
    if not root:
        raise ValueError("A dataset project is required")
    pending = [root]
    files = []

    while pending:
        folder = pending.pop()
        entries, child_folders = client.list_files(f"{folder}/")
        files.extend(
            entry.get("path") or posixpath.join(folder, entry.get("name", ""))
            for entry in entries or []
        )
        pending.extend(
            child.get("path") or posixpath.join(folder, child.get("name", ""))
            for child in child_folders or []
        )

    if progress_callback:
        progress_callback(0, len(files), "Preparing deletion", False)
    for index, path in enumerate(files, start=1):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Operation canceled")
        if progress_callback:
            progress_callback(index - 1, len(files), path, False)
        client.remove_file(path)
        if progress_callback:
            progress_callback(index, len(files), path, True)
    return root


def create_dataset(project, client=None):
    """Load a dataset and persist its initial annotations file."""
    dataset = load_dataset(project, client=client)
    dataset.save()
    return dataset


__all__ = [
    "Annotator",
    "Dataset",
    "annotate_image",
    "create_dataset",
    "delete_dataset",
    "list_datasets",
    "load_dataset",
]
