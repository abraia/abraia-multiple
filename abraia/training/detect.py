from ..tasks import normalize_model_size, normalize_task
from .splitting import DEFAULT_EVALUATION_SPLIT

import os
import io
import sys
import contextlib
import numpy as np
from pathlib import Path

os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO

from .core import (
    HAILO_EXPORT_TARGETS,
    _resolve_client,
    ensure_hailo_dfc,
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
    task = normalize_task(task)
    if task == 'segmentation':
        model_name = f"{model_name}-seg"
    return model_name


class Model:
    def __init__(self, task, model_type=None, imgsz=640, client=None,
                 model_size="small", checkpoint=None):
        task = normalize_task(task)
        if task not in ("detection", "segmentation"):
            raise ValueError(
                "Ultralytics detection models support detection or segmentation"
            )
        model_size = normalize_model_size(model_size)
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
        train_options = {'data': data, 'batch': batch, 'epochs': epochs, 'imgsz': self.imgsz}
        if sys.platform == 'darwin':
            # Ultralytics workers inherit locks when training is launched by
            # Studio's Python worker thread. A single loader is safer on macOS.
            train_options['workers'] = 0
        try:
            self.model.train(**train_options)
        finally:
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
                    'task': self.task,
                    'inputShape': [1, 3, self.imgsz, self.imgsz],
                    'classes': classes,
                    'metrics': self.metrics,
                },
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
        version=None,
        calibration_data=None,
        fraction=None,
        imgsz=None,
        conf=None,
        iou=None,
    ):
        """Export the trained YOLO model to a versioned Hailo bundle.

        Ultralytics performs the ONNX conversion, INT8 calibration, Hailo
        parsing, and HEF compilation in one export operation. This method
        must be called on the trained model instance; it deliberately does
        not download an ONNX file and construct a fresh base model.
        """
        target = str(device or "hailo8").strip().lower()
        if target not in HAILO_EXPORT_TARGETS:
            choices = ", ".join(HAILO_EXPORT_TARGETS)
            raise ValueError(f"Unsupported Hailo target '{target}'. Use: {choices}")

        if calibration_data is None:
            default_data = os.path.join(project, "data.yaml")
            calibration_data = default_data if os.path.isfile(default_data) else None
        elif not os.path.exists(calibration_data):
            raise FileNotFoundError(
                f"Hailo calibration data does not exist: {calibration_data}"
            )

        compile_imgsz = self.imgsz if imgsz is None else imgsz
        export_options = {
            "format": "hailo",
            "name": target,
            "quantize": 8,
            "imgsz": compile_imgsz,
        }
        if calibration_data is not None:
            export_options["data"] = os.fspath(calibration_data)
        if fraction is not None:
            export_options["fraction"] = fraction
        if conf is not None:
            export_options["conf"] = conf
        if iou is not None:
            export_options["iou"] = iou

        ensure_hailo_dfc(target)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
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

        version = self.model_version if version is None else version
        if isinstance(compile_imgsz, (list, tuple)):
            input_shape = [1, 3, *compile_imgsz]
        else:
            input_shape = [1, 3, compile_imgsz, compile_imgsz]
        manifest = {
            "format": "hailo",
            "target": target,
            "task": self.task,
            "inputShape": input_shape,
            "classes": classes,
            "metrics": self.metrics,
            "calibrationData": os.fspath(calibration_data) if calibration_data else None,
            "fraction": fraction,
            "quantize": 8,
        }
        result = save_hailo_bundle(
            self.client,
            project,
            self.model_name,
            bundle_path,
            target,
            version=version,
            metadata=manifest,
        )
        result["localBundle"] = str(bundle_path)
        self.hailo_model = result
        return result
