"""High-level dataset annotation and training services."""

from __future__ import annotations

import gc
import os
import posixpath

from ..tasks import normalize_task
from .annotations import canonical_filename, upsert_annotation
from .splitting import DEFAULT_EVALUATION_SPLIT
from .dataset import search_images


class TrainingService:
    """Own annotation and training model lifecycles for application clients."""

    def auto_annotate(
        self, dataset, label, progress_callback, is_cancelled, task=None
    ):
        """Annotate unannotated dataset images and persist each result."""
        from .dataset import Annotator, annotate_image

        task = normalize_task(task or dataset.task)
        segment = task == "segmentation"
        classification = task == "classification"
        completed = {annotation["filename"] for annotation in dataset.annotations}
        images = [image for image in dataset.images if image["name"] not in completed]
        total = len(images)
        annotator = Annotator(segment=segment) if images else None
        try:
            for index, image_data in enumerate(images, start=1):
                if is_cancelled():
                    break
                annotation = annotate_image(
                    image_data,
                    [label],
                    segment=segment,
                    annotator=annotator,
                    classification=classification,
                )
                image_annotations = [annotation] if annotation else []
                if image_annotations:
                    dataset.annotations.extend(image_annotations)
                dataset.save()
                annotation_count = sum(
                    len(annotation.get("objects") or [])
                    for annotation in image_annotations
                )
                progress_callback(index, total, image_data["name"], annotation_count)
        finally:
            try:
                dataset.save()
            finally:
                if annotator is not None:
                    annotator.close()
        return dataset, len(dataset.annotations), is_cancelled()

    def train_dataset(self, project, dataset, epochs, training_callback,
                      is_cancelled, model_size="small", split_options=None):
        """Prepare, train, validate, and save a project model."""
        # These orchestration helpers live in the public training package;
        # only annotation primitives live in ``training.dataset``.
        from . import ModelTrainer, prepare_dataset

        task = normalize_task(dataset.task)
        trainer = None
        try:
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Preparing dataset"})
            prepare_dataset(
                dataset,
                force=split_options is not None,
                split_options=split_options,
                callback=lambda event: training_callback(
                    {"stage": "Preparing dataset", **event}
                ),
            )
            if is_cancelled():
                raise RuntimeError("Training canceled")
            # Model construction can import/load a backend for several
            # seconds. Publish the transition before doing that work so the
            # UI does not remain stuck on dataset preparation.
            training_callback({"stage": "Loading model"})
            trainer_kwargs = {"client": getattr(dataset, "client", None)}
            if model_size != "small":
                trainer_kwargs["model_size"] = model_size
            trainer = ModelTrainer(project, task, dataset.classes, **trainer_kwargs)

            def on_epoch(metrics):
                if is_cancelled():
                    raise RuntimeError("Training canceled")
                training_callback({"stage": "Training", **metrics})

            training_callback({"stage": "Training"})
            trainer.train(
                epochs=epochs,
                callback=on_epoch,
                is_cancelled=is_cancelled,
            )
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Validating model"})
            # Final metrics must describe the held-out testing partition. The
            # validation split is used during training/model selection and is
            # therefore not an unbiased post-training report.
            stats = trainer.test(
                split=DEFAULT_EVALUATION_SPLIT,
                is_cancelled=is_cancelled,
            )
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Exporting model"})
            saved_model = trainer.save()
            return {
                "stats": stats,
                "epochs": epochs,
                "task": task,
                "saved_model": saved_model,
            }
        finally:
            del trainer
            gc.collect()


class DatasetProjectService:
    """Mutate a remote dataset through injected client and loader APIs.

    The loader is injected because ``multiple`` provides a spectral dataset
    adapter while ``abraia`` provides the standard image adapter.
    """

    def __init__(self, client, dataset_loader, training_service=None):
        self.client = client
        self.dataset_loader = dataset_loader
        self.training = training_service

    def create_empty_dataset(self, project):
        self.client.save_json(f"{project}/annotations.json", [])
        return self.dataset_loader(project, validate=False)

    def save_annotation(self, project, filename, objects):
        dataset = self.dataset_loader(project)
        if upsert_annotation(dataset, filename, objects):
            dataset.save()
        return dataset

    def delete_image(self, project, path):
        folder, name = os.path.split(path)
        self.client.remove_file(path)
        try:
            self.client.remove_file(os.path.join(folder, f"tb_{name}"))
        except Exception:
            pass
        dataset = self.dataset_loader(project)
        filename = canonical_filename(name or path)
        annotations = dataset.annotations or []
        remaining = [
            annotation
            for annotation in annotations
            if canonical_filename(annotation.get("filename")) != filename
        ]
        if len(remaining) != len(annotations):
            dataset.annotations = remaining
            dataset.save()
        self._remove_images_from_dataset(dataset, [path])
        return dataset

    @staticmethod
    def _remove_images_from_dataset(dataset, paths):
        """Keep an already-loaded dataset consistent with deleted files."""
        images = getattr(dataset, "images", None)
        if images is None:
            return

        removed_paths = {str(path).strip("/") for path in paths if path}
        removed_names = {os.path.basename(path) for path in removed_paths}
        remaining = []
        for image in images:
            image_path = str(image.get("path") or "").strip("/")
            image_name = str(image.get("name") or "").strip("/")
            matches_path = image_path in removed_paths
            matches_name = not image_path and image_name in removed_names
            if not matches_path and not matches_name:
                remaining.append(image)
        dataset.images = remaining

        update_annotated = getattr(dataset, "_update_annotated", None)
        if callable(update_annotated):
            update_annotated()

    def delete_images(self, project, paths, progress_callback=None, is_cancelled=None):
        """Remove several images with one dataset reload and annotation save."""
        paths = list(dict.fromkeys(path for path in paths if path))
        is_cancelled = is_cancelled or (lambda: False)
        if progress_callback:
            progress_callback(0, len(paths), "Preparing image deletion", False)
        if not paths:
            return None

        dataset = self.dataset_loader(project)
        removed_filenames = set()
        for index, path in enumerate(paths, start=1):
            if is_cancelled():
                raise RuntimeError("Operation canceled")
            if progress_callback:
                progress_callback(index - 1, len(paths), path, False)
            folder, name = os.path.split(path)
            self.client.remove_file(path)
            try:
                self.client.remove_file(os.path.join(folder, f"tb_{name}"))
            except Exception:
                pass
            removed_filenames.add(canonical_filename(name or path))
            if progress_callback:
                progress_callback(index, len(paths), path, True)

        annotations = dataset.annotations or []
        remaining = [
            annotation
            for annotation in annotations
            if canonical_filename(annotation.get("filename")) not in removed_filenames
        ]
        if len(remaining) != len(annotations):
            dataset.annotations = remaining
            dataset.save()
        self._remove_images_from_dataset(dataset, paths)
        return dataset

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

    def create_dataset(self, project, query, files, progress_callback, is_cancelled):
        dataset = self.dataset_loader(project)
        total = max(1, (50 if query else 0) + len(files))
        if query:
            search_images(
                query,
                f"{project}/",
                limit=50,
                callback=lambda event: progress_callback(
                    event.get("current", 0),
                    total,
                    event.get("filename", "web image"),
                    True,
                ),
            )
        completed = 50 if query else 0
        for file_path in files:
            if is_cancelled():
                break
            filename = os.path.basename(file_path)
            progress_callback(completed, total, filename, False)
            self.client.upload_file(file_path, f"{project}/")
            completed += 1
            progress_callback(completed, total, filename, True)
        progress_callback(total, total, "Saving dataset", True)
        dataset.save()
        return self.dataset_loader(project, validate=False)

    def auto_annotate(
        self, project, label, progress_callback, is_cancelled, task=None
    ):
        dataset = self.dataset_loader(project)
        if self.training is None:
            raise RuntimeError("A training service is required for auto-annotation")
        return self.training.auto_annotate(
            dataset, label, progress_callback, is_cancelled, task=task
        )


__all__ = ["DatasetProjectService", "TrainingService"]
