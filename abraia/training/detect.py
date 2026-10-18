from ..tasks import normalize_task
from ..dataset.splitting import DEFAULT_EVALUATION_SPLIT

import os
import io
import sys
import contextlib
import numpy as np
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO

from .artifacts import (
    HAILO_EXPORT_TARGETS,
    _resolve_client,
    save_hailo_bundle,
    save_versioned_model,
)

MODEL_SIZE_TYPES = {
    "small": "yolov8n",
    "medium": "yolov8m",
    "large": "yolov8l",
}


def _numeric_values(value):
    """Normalize Ultralytics tensor, scalar, and mapping metrics."""
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = (value,)
    numeric = []
    for item in values:
        if hasattr(item, "detach"):
            item = item.detach()
        if hasattr(item, "cpu"):
            item = item.cpu()
        if hasattr(item, "numpy"):
            item = item.numpy()
        numeric.extend(np.asarray(item, dtype=float).reshape(-1).tolist())
    return numeric


def _metric_value(metrics, key, default=0.0):
    """Read a metric from old and new Ultralytics metric containers."""
    if isinstance(metrics, dict):
        value = metrics.get(key, default)
    else:
        values = getattr(metrics, "results_dict", {}) or {}
        value = values.get(key, default)
    numeric = _numeric_values(value)
    return float(numeric[0]) if numeric else float(default)


def build_model_name(model_name, task):
    if task == 'segmentation':
        model_name = f"{model_name}-seg"
    return model_name


class Model:
    def __init__(self, task, model_type=None, imgsz=640, client=None,
                 model_size="small", checkpoint=None):
        task = normalize_task(task)
        if checkpoint and model_type is None:
            model_type = Path(checkpoint).stem
            if task == "segmentation" and model_type.endswith("-seg"):
                model_type = model_type[:-4]
        model_type = model_type or MODEL_SIZE_TYPES[model_size]
        model_name = build_model_name(model_type, task)
        self.model = YOLO(checkpoint or f"{model_name}.pt", verbose=False)
        self.model_name = model_name
        self.metrics = {}
        self.task = task
        self.model_size = model_size
        self.imgsz = imgsz
        self.client = _resolve_client(client)
        self._training_callbacks = {}
        self.model_version = None
        self.checkpoint_path = None

    def _remove_training_callbacks(self):
        for event, callback in self._training_callbacks.items():
            self._remove_callback(event, callback)
        self._training_callbacks.clear()

    def _remove_callback(self, event, callback):
        callbacks = self.model.callbacks.get(event, [])
        try:
            callbacks.remove(callback)
        except ValueError:
            pass

    def train(self, project, epochs=100, batch=32, callback=None,
              is_cancelled=None):
        self._remove_training_callbacks()
        if callback:
            def on_train_epoch_end(trainer):
                loss_values = _numeric_values(getattr(trainer, "loss_items", 0.0))
                loss = float(np.mean(loss_values)) if loss_values else 0.0
                metrics = getattr(trainer, "metrics", {})
                acc = _metric_value(metrics, "metrics/mAP50(B)")
                callback({
                    "epoch": trainer.epoch,
                    "epochs": trainer.epochs,
                    "loss": loss,
                    "acc": acc,
                })
            self._training_callbacks['on_train_epoch_end'] = on_train_epoch_end
            self.model.add_callback('on_train_epoch_end', on_train_epoch_end)
        if is_cancelled:
            def on_train_batch_end(_trainer):
                if is_cancelled():
                    raise RuntimeError("Training canceled")
            self._training_callbacks['on_train_batch_end'] = on_train_batch_end
            self.model.add_callback('on_train_batch_end', on_train_batch_end)
        data = f"{project}/data.yaml"
        output_dir = (
            Path(project).resolve()
            / ".abraia" / "training" / self.task / self.model_size
        )
        train_options = {
            'data': data,
            'batch': batch,
            'epochs': epochs,
            'imgsz': self.imgsz,
            'project': str(output_dir),
            'name': 'train',
            'exist_ok': True,
        }
        if sys.platform == 'darwin':
            # Ultralytics workers inherit locks when training is launched by
            # Studio's Python worker thread. A single loader is safer on macOS.
            train_options['workers'] = 0
        try:
            self.model.train(**train_options)
        finally:
            trainer = getattr(self.model, "trainer", None)
            last_checkpoint = getattr(trainer, "last", None)
            save_dir = getattr(trainer, "save_dir", None)
            self.checkpoint_path = (
                Path(last_checkpoint).resolve()
                if last_checkpoint is not None
                else (
                    Path(save_dir).resolve() / "weights" / "last.pt"
                    if save_dir is not None
                    else output_dir / "train" / "weights" / "last.pt"
                )
            )
            self._remove_training_callbacks()

    def test(self, split=DEFAULT_EVALUATION_SPLIT, is_cancelled=None):
        out = io.StringIO()
        validation_callback = None
        if is_cancelled:
            def on_val_batch_end(_validator):
                if is_cancelled():
                    raise RuntimeError("Training canceled")
            validation_callback = on_val_batch_end
            self.model.add_callback('on_val_batch_end', validation_callback)
        try:
            with contextlib.redirect_stderr(out):
                metrics = self.model.val(split=split)
        finally:
            if validation_callback:
                self._remove_callback('on_val_batch_end', validation_callback)
        self.metrics = {'mAP': float(metrics.box.map50), 'P': metrics.box.p.tolist(), 'R': metrics.box.r.tolist(), 
                        'confusionMatrix': metrics.confusion_matrix.matrix.tolist()}
        return self.metrics

    def save(self, project, classes, device='cpu', half=False):
        out = io.StringIO()
        model_src = None
        checkpoint = self.checkpoint_path or (
            Path(project).resolve()
            / ".abraia" / "training" / self.task / self.model_size
            / "train" / "weights" / "last.pt"
        )
        try:
            with contextlib.redirect_stdout(out):
                model_src = self.model.export(
                    format="onnx", device=device, opset=11, half=half
                )
            result = save_versioned_model(
                self.client,
                project,
                self.model_name,
                model_src,
                {
                    'backend': 'ultralytics',
                    'task': self.task,
                    'inputShape': [1, 3, self.imgsz, self.imgsz],
                    'classes': classes,
                    'metrics': self.metrics,
                    'modelSize': self.model_size,
                },
                artifacts={'checkpoint': checkpoint},
            )
            self.model_version = result["version"]
            return result
        finally:
            if model_src and os.path.isfile(model_src):
                try:
                    os.remove(model_src)
                except OSError:
                    pass

    def run(self, img):
        objects = []
        results = self.model.predict(img, verbose=False)[0]
        if results:
            for k, box in enumerate(results.boxes):
                class_id = int(box.cls)
                label = results.names[class_id]
                score = float(box.conf)
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                object = {'label': label, 'score': score, 'box': [x1, y1, x2 - x1, y2 - y1]}
                if self.task == 'segmentation':
                    object['polygon'] = results.masks[k].xy[0]
                objects.append(object)
        return objects
    
    def compile(
        self,
        project,
        classes,
        device="hailo8",
        conf=None,
        iou=None,
        progress_callback=None,
        is_cancelled=None,
    ):
        """Export the loaded YOLO model to a versioned Hailo bundle.

        Ultralytics performs the ONNX conversion, INT8 calibration, Hailo
        parsing, and HEF compilation in one export operation. The model may
        be trained in this process or loaded from its saved checkpoint.
        """
        target = str(device or "hailo8").strip().lower()
        if target not in HAILO_EXPORT_TARGETS:
            choices = ", ".join(HAILO_EXPORT_TARGETS)
            raise ValueError(f"Unsupported Hailo target '{target}'. Use: {choices}")

        calibration_data = os.path.join(project, "data.yaml")

        export_options = {
            "format": "hailo",
            "name": target,
            "quantize": 8,
            "imgsz": self.imgsz,
        }
        export_options["data"] = calibration_data
        if conf is not None:
            export_options["conf"] = conf
        if iou is not None:
            export_options["iou"] = iou

        if progress_callback:
            progress_callback(0, 100, "Checking Hailo compiler", "Running")
        if is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")
        if progress_callback:
            progress_callback(
                5, 100,
                "Preparing calibration data and exporting ONNX",
                "Running",
            )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if progress_callback:
                    from contextlib import ExitStack
                    from unittest.mock import patch
                    from hailo_sdk_client import ClientRunner

                    stages = (
                        ("translate_onnx_model", 15, 25, "Parsing ONNX model"),
                        ("optimize", 25, 85, "Optimizing INT8 model"),
                        ("compile", 85, 95, "Compiling HEF"),
                    )
                    with ExitStack() as patches:
                        for method_name, started, completed, label in stages:
                            original = getattr(ClientRunner, method_name)

                            def report_stage(
                                runner, *args, _original=original,
                                _started=started, _completed=completed, _label=label,
                                **kwargs,
                            ):
                                if is_cancelled and is_cancelled():
                                    raise RuntimeError("Hailo compilation canceled")
                                progress_callback(_started, 100, _label, "Running")
                                result = _original(runner, *args, **kwargs)
                                if is_cancelled and is_cancelled():
                                    raise RuntimeError(
                                        "Hailo compilation canceled"
                                    )
                                progress_callback(
                                    _completed, 100,
                                    _label + " complete", "Complete",
                                )
                                return result

                            patches.enter_context(
                                patch.object(ClientRunner, method_name, report_stage)
                            )
                        exported = self.model.export(**export_options)
                else:
                    exported = self.model.export(**export_options)
        except AssertionError as error:
            message = str(error).lower()
            if "linux x86_64" in message:
                raise RuntimeError(
                    "Hailo compilation requires a Linux x86_64 host with the "
                    "matching Hailo Dataflow Compiler installed."
                ) from error
            if "format" in message or "hailo" in message:
                raise RuntimeError(
                    "Install ultralytics>=8.4.97 and the matching Hailo "
                    "Dataflow Compiler to enable native Hailo export."
                ) from error
            raise
        except ImportError as error:
            if "hailo" in str(error).lower():
                raise RuntimeError(
                    "Install the matching Hailo Dataflow Compiler wheel to "
                    "enable native Hailo export."
                ) from error
            raise
        except ValueError as error:
            if "hailo" in str(error).lower() and "format" in str(error).lower():
                raise RuntimeError(
                    "Install ultralytics>=8.4.97 and the matching Hailo "
                    "Dataflow Compiler to enable native Hailo export."
                ) from error
            raise

        if isinstance(exported, (tuple, list)):
            exported = exported[0]
        bundle_path = Path(exported)
        if bundle_path.is_file():
            bundle_path = bundle_path.parent

        if isinstance(self.imgsz, (list, tuple)):
            input_shape = [1, 3, *self.imgsz]
        else:
            input_shape = [1, 3, self.imgsz, self.imgsz]
        manifest = {
            "format": "hailo",
            "target": target,
            "task": self.task,
            "inputShape": input_shape,
            "classes": classes,
            "metrics": self.metrics,
            "calibrationData": os.fspath(calibration_data) if calibration_data else None,
            "quantize": 8,
        }
        if progress_callback and is_cancelled and is_cancelled():
            raise RuntimeError("Hailo compilation canceled")
        result = save_hailo_bundle(
            self.client,
            project,
            self.model_name,
            bundle_path,
            target,
            version=self.model_version,
            metadata=manifest,
            progress_callback=progress_callback,
        )
        result["localBundle"] = str(bundle_path)
        self.hailo_model = result
        return result
