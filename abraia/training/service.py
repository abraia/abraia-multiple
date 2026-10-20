"""Client-facing training workflows."""

import os

from ..dataset import load_dataset, preparation
from ..dataset.splitting import DEFAULT_EVALUATION_SPLIT
from ..tasks import (
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_IMAGE_SIZE,
    TRAINING_TASKS,
    normalize_task,
)
from .artifacts import list_model_records
from .model_data import metric_average
from .orchestration import ModelTrainer, train_dataset


def display_metrics(metrics):
    metrics = dict(metrics or {})
    for name in ("acc", "accuracy", "mAP", "P", "R"):
        if name in metrics:
            metrics[name] = metric_average(metrics[name])
    return metrics


class Training:
    """Train and evaluate models, list records, and compile checkpoints."""

    def __init__(self, client, *, train_operation=None):
        self._client = client
        self._train_operation = train_operation or train_dataset

    @staticmethod
    def resolve_defaults(task):
        """Return the standard epoch and image-size defaults for a task."""
        task = normalize_task(task)
        if task not in TRAINING_TASKS:
            return {"epochs": 1, "image_size": 32}
        return {
            "epochs": DEFAULT_TRAINING_EPOCHS[task],
            "image_size": DEFAULT_TRAINING_IMAGE_SIZE[task],
        }

    def list_models(self, project):
        records = list_model_records(
            project, client=self._client, sort_by_accuracy=True
        )
        return [
            {**record, "metrics": display_metrics(record.get("metrics"))}
            for record in records
        ]

    def train(
        self, project, epochs=None, training_callback=None, is_cancelled=None,
        model_size="small", split_options=None, image_size=None,
        selected_model=None, backend=None, publish_common=False,
        analyze_thresholds=True,
    ):
        dataset = load_dataset(project, client=self._client)
        task = normalize_task(dataset.task)
        if task not in TRAINING_TASKS:
            raise ValueError(
                f"Unsupported training task: {task}. Supported tasks: {TRAINING_TASKS}"
            )
        if epochs is None:
            epochs = DEFAULT_TRAINING_EPOCHS[task]
        if image_size is None:
            image_size = DEFAULT_TRAINING_IMAGE_SIZE[task]
        if split_options is not None:
            dataset.split.set_options(split_options)
        selected_model = selected_model or {}
        artifacts = selected_model.get("artifacts") or {}
        checkpoint_path = artifacts.get("checkpoint")
        # Ultralytics plots during final validation. Force a headless backend so
        # Matplotlib does not auto-select QtAgg, which can fail with some Qt
        # bindings before training reaches model export.
        import matplotlib

        matplotlib.use("Agg", force=True)
        options = dict(
            model_size=model_size,
            imgsz=image_size,
            analyze_thresholds=analyze_thresholds,
        )
        if backend:
            options["backend"] = backend
        if publish_common:
            options["publish_common"] = True
        if checkpoint_path:
            options.update(
                backend=selected_model.get("backend") or backend or "ultralytics",
                checkpoint=self._client.download_file(checkpoint_path),
                model_size=selected_model.get("modelSize") or model_size,
            )
        result = self._train_operation(
            project,
            dataset,
            epochs,
            training_callback,
            is_cancelled,
            **options,
        )
        result["stats"] = display_metrics(result.get("stats"))
        result["models"] = self.list_models(project)
        saved_model = result.get("saved_model") or {}
        model_name = saved_model.get("name")
        if not model_name:
            model_path = saved_model.get("onnx")
            model_name = os.path.basename(str(model_path)) if model_path else None
        if model_name:
            result["model_name"] = model_name
        return result

    def evaluate(
        self, project, selected_model,
        split=DEFAULT_EVALUATION_SPLIT, split_options=None,
        training_callback=None, is_cancelled=None,
    ):
        """Evaluate a saved checkpoint and return its metrics."""
        if not isinstance(selected_model, dict):
            raise ValueError("Select a saved model to evaluate.")
        task = selected_model.get("task")
        if task not in ("detection", "segmentation"):
            raise ValueError(
                "Saved-checkpoint evaluation supports detection and "
                "segmentation models."
            )
        backend = selected_model.get("backend") or "ultralytics"
        if backend not in ("ultralytics", "libreyolo"):
            raise ValueError(
                f"Saved-model evaluation is not available for {backend} models."
            )
        checkpoint_path = (selected_model.get("artifacts") or {}).get(
            "checkpoint"
        )
        if not checkpoint_path:
            raise ValueError("The selected model has no saved checkpoint.")
        split = str(split or DEFAULT_EVALUATION_SPLIT).lower()
        split = "val" if split == "validation" else split
        if split not in ("val", "test"):
            raise ValueError("Evaluation split must be 'validation' or 'test'.")
        if is_cancelled and is_cancelled():
            raise RuntimeError("Evaluation canceled")

        dataset = load_dataset(project, client=self._client)
        dataset.prune_orphaned_annotations()
        if task != dataset.task:
            raise ValueError(
                f"The selected model task ({task}) does not match the "
                f"dataset task ({dataset.task})."
            )
        model_classes = selected_model.get("classes")
        if model_classes and model_classes != dataset.classes:
            raise ValueError(
                "The selected model classes do not match the dataset classes."
            )
        if split_options is not None:
            dataset.split.set_options(split_options)

        preparation_callback = None
        if training_callback:
            training_callback({"stage": "Preparing evaluation data"})
            preparation_callback = lambda event: training_callback(
                {"stage": "Preparing evaluation data", **event}
            )
        preparation.prepare_dataset(dataset, callback=preparation_callback)
        if is_cancelled and is_cancelled():
            raise RuntimeError("Evaluation canceled")

        input_shape = selected_model.get("inputShape") or [1, 3, 640, 640]
        trainer = ModelTrainer(
            project,
            task,
            dataset.classes,
            imgsz=int(input_shape[-1]),
            client=self._client,
            model_size=selected_model.get("modelSize") or "small",
            backend=backend,
            checkpoint=self._client.download_file(checkpoint_path),
        )
        if training_callback:
            training_callback({"stage": "Evaluating model"})
        metrics = display_metrics(
            trainer.test(split=split, is_cancelled=is_cancelled)
        )
        if is_cancelled and is_cancelled():
            raise RuntimeError("Evaluation canceled")
        return metrics

    def compile(
        self, target, project, selected_model, progress_callback=None,
        is_cancelled=None, **options,
    ):
        artifacts = selected_model.get("artifacts") or {}
        checkpoint_path = artifacts.get("checkpoint")
        if not checkpoint_path:
            raise RuntimeError("The selected model has no saved checkpoint.")

        from .orchestration import _compile_checkpoint

        model_stem = os.path.splitext(selected_model["name"])[0]
        version = model_stem.rsplit("_v", 1)[-1]
        input_shape = selected_model.get("inputShape") or [1, 3, 640, 640]
        checkpoint = self._client.download_file(checkpoint_path)
        config_path = (
            self._client.download_file(artifacts["config"])
            if artifacts.get("config") else None
        )
        return _compile_checkpoint(
            project,
            selected_model["task"],
            selected_model.get("classes") or [],
            checkpoint,
            selected_model.get("backend") or "ultralytics",
            model_size=selected_model.get("modelSize") or "small",
            imgsz=input_shape[-1],
            client=self._client,
            config_path=config_path,
            model_version=int(version) if version.isdigit() else None,
            device=target,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
            **options,
        )


__all__ = ["Training"]
