"""Remote dataset adapter and project-level image operations."""

from concurrent.futures import ThreadPoolExecutor
import copy
import os
from typing import Mapping

from tqdm import tqdm

from . import RemoteDataset, _resolve_client, canonical_filename
from ..utils import load_image, load_url
from ..utils.concurrency import DEFAULT_MAX_REMOTE_WORKERS, bounded_map
from .annotation import Annotator, annotate_image as _annotate_image
from .image_search import search_images
from ..tasks import normalize_task


def image_filename(image):
    """Return the best comparable filename from an image record."""
    return canonical_filename(image.get("name") or image.get("path"))


def find_image(images, filename):
    """Find an image record by name or path basename."""
    key = canonical_filename(filename)
    return next(
        (image for image in images or [] if image_filename(image) == key),
        None,
    )


def objects_for_image(dataset, path, display_name=""):
    """Return saved annotation objects for an image record."""
    if not dataset:
        return []
    filenames = {canonical_filename(display_name), canonical_filename(path)}
    for annotation in dataset.annotations or []:
        if canonical_filename(annotation.get("filename")) in filenames:
            return annotation.get("objects") or []
    return []


def annotation_counts(annotations):
    """Return saved annotation object counts keyed by filename."""
    counts = {}
    for annotation in annotations or []:
        filename = canonical_filename(annotation.get("filename"))
        if filename:
            counts[filename] = counts.get(filename, 0) + len(
                annotation.get("objects") or []
            )
    return counts


def dataset_has_annotations(dataset):
    """Return whether a dataset contains images and annotation metadata."""
    return bool(dataset and dataset.images and dataset.annotations)


def _curation_progress(callback, offset, span):
    if callback is None:
        return None

    def report(current, total, detail, completed):
        total = max(1, int(total or 1))
        current = max(0, min(int(current), total))
        progress = offset + round(span * current / total)
        callback(progress, 100, detail, completed)

    return report


def annotate_image(
    image_data,
    classes,
    segment=False,
    annotator=None,
    include_empty=False,
    classification=False,
):
    """Compatibility wrapper preserving patchable dataset-level loaders."""
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


class Dataset(RemoteDataset):
    """Remote dataset containing standard image records."""

    def __init__(self, project, client=None):
        super().__init__(project, _resolve_client(client))

    def load(self, validate=True):
        if validate and self.project not in list_datasets(self.client):
            return self.clear_contents()
        return self.load_contents(parallel=True)

    def _select_images(self, files):
        return [
            file_data
            for file_data in files
            if file_data["type"] in ["image/jpeg", "image/png"]
        ]

    def add_images(
        self, files=(), *, query=None, progress_callback=None, is_cancelled=None
    ):
        """Add local files and optional web-search results to this dataset."""
        files = list(files or ())
        is_cancelled = is_cancelled or (lambda: False)
        total = max(1, (50 if query else 0) + len(files))

        search_callback = None
        if query:
            if progress_callback:
                def report_search(event):
                    progress_callback(
                        event.get("current", 0),
                        total,
                        event.get("filename", "web image"),
                        True,
                    )
                search_callback = report_search

            search_images(
                query,
                f"{self.project}/",
                limit=50,
                client=self.client,
                callback=search_callback,
            )

        completed = 50 if query else 0
        for file_path in files:
            if is_cancelled():
                break
            filename = os.path.basename(file_path)
            if progress_callback:
                progress_callback(completed, total, filename, False)
            self.client.upload_file(file_path, f"{self.project}/")
            completed += 1
            if progress_callback:
                progress_callback(completed, total, filename, True)

        if progress_callback:
            progress_callback(total, total, "Saving dataset", True)
        self.refresh_images()
        self.save()
        return self

    def curate(
        self,
        work_dir=None,
        apply=False,
        analyze_fn=None,
        confirm=None,
        progress_callback=None,
        is_cancelled=None,
    ):
        """Analyze this dataset, return the report, and retain its findings."""
        from .curation import analyze_curation

        project = self.project
        client = self.client
        analyze_fn = analyze_fn or analyze_curation
        images = list(self.images or [])
        annotated = {
            self._annotation_path(annotation)
            for annotation in self.annotations or []
            if isinstance(annotation, Mapping)
        }
        paths = []
        for image in images:
            paths.append(self._image_path(image))
        protected = [path for path in paths if path in annotated]
        total = len(paths)
        downloaded = 0
        report_download_progress = _curation_progress(
            progress_callback, 0, 50
        )

        def path_loader(remote_path):
            nonlocal downloaded
            if is_cancelled and is_cancelled():
                raise RuntimeError("Operation canceled")
            if report_download_progress:
                report_download_progress(
                    downloaded, total, remote_path, False
                )
            local_path = client.download_file(self._remote_path(remote_path))
            downloaded += 1
            if report_download_progress:
                report_download_progress(
                    downloaded, total, remote_path, True
                )
            return local_path

        owner = str(getattr(client, "userid", "") or "").strip("/")
        report = analyze_fn(
            paths,
            path_loader=path_loader,
            work_dir=work_dir,
            protected_paths=protected,
            cache_namespace=f"{owner}/{str(project).strip('/')}",
            progress_callback=_curation_progress(
                progress_callback, 50, 50
            ),
            is_cancelled=is_cancelled,
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
            self.curation_findings = copy.deepcopy(
                report.get("findings", []) if isinstance(report, dict) else []
            )
        if progress_callback:
            progress_callback(100, 100, "Curation complete", True)
        return report

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
                    self.add_annotations([annotation])
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

def load_dataset(project, validate=True, client=None):
    """Load a remote standard-image dataset."""
    return Dataset(project, client=client).load(validate=validate)


__all__ = [
    "Annotator",
    "Dataset",
    "annotation_counts",
    "annotate_image",
    "dataset_has_annotations",
    "find_image",
    "image_filename",
    "list_datasets",
    "load_dataset",
    "objects_for_image",
]
