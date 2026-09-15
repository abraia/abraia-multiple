"""High-level dataset annotation and training services."""

from __future__ import annotations

import gc


class TrainingService:
    """Own annotation and training model lifecycles for application clients."""

    def auto_annotate(self, dataset, label, progress_callback, is_cancelled):
        """Annotate unannotated dataset images and persist each result."""
        from .dataset import Annotator, annotate_image

        segment = dataset.task == "segment"
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
            dataset.save()
        return dataset, len(dataset.annotations), is_cancelled()

    def train_dataset(self, project, dataset, epochs, training_callback, is_cancelled):
        """Prepare, train, validate, and save a project model."""
        # These orchestration helpers live in the public training package;
        # only annotation primitives live in ``training.dataset``.
        from . import ModelTrainer, prepare_dataset

        trainer = None
        try:
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Preparing dataset"})
            prepare_dataset(
                dataset,
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
            trainer = ModelTrainer(
                project,
                dataset.task,
                dataset.classes,
                client=getattr(dataset, "client", None),
            )

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
            stats = trainer.test(is_cancelled=is_cancelled)
            if is_cancelled():
                raise RuntimeError("Training canceled")
            training_callback({"stage": "Exporting model"})
            trainer.save()
            return {"stats": stats, "epochs": epochs, "task": dataset.task}
        finally:
            del trainer
            gc.collect()
