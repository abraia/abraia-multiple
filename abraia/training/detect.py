from ..client import Abraia
from ..tasks import normalize_task

import os
import io
import sys
import shutil
import contextlib
import numpy as np

os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO

abraia = Abraia()


def sorted_folders(dir):
    items = [os.path.join(dir, name) for name in os.listdir(dir)]
    sorted_items = sorted(items, key=os.path.getctime)
    return sorted_items


def build_model_name(model_name, task):
    task = normalize_task(task)
    if task == 'segmentation':
        model_name = f"{model_name}-seg"
    if task == 'classification':
        model_name = f"{model_name}-cls"
    return model_name


class Model:
    def __init__(self, task, model_type='yolov8n', imgsz=640, client=None):
        task = normalize_task(task)
        if task not in ("detection", "segmentation"):
            raise ValueError(
                "Ultralytics detection models support detection or segmentation"
            )
        model_name = build_model_name(model_type, task)
        self.model = YOLO(f"{model_name}.pt", verbose=False)
        self.model_name = model_name
        self.metrics = {}
        self.task = task
        self.imgsz = imgsz
        self.client = abraia if client is None else client
        self._training_callbacks = {}

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
                loss_items = trainer.loss_items.cpu().detach().numpy()
                loss = float(np.sum(loss_items)) / len(loss_items)
                acc = trainer.metrics.get('metrics/mAP50(B)', 0) if hasattr(trainer, 'metrics') else 0
                callback({'epoch': trainer.epoch, 'epochs': trainer.epochs, 'loss': loss, 'acc': float(acc)})
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

    def test(self, split='val', is_cancelled=None):
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
        # TODO: Add model name versioning
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            model_src = self.model.export(format="onnx", device=device, opset=11, half=half)
            shutil.copy(model_src, f"{self.model_name}.onnx")
        self.client.upload_file(f"{self.model_name}.onnx", f"{project}/{self.model_name}.onnx")
        self.client.save_json(f"{project}/{self.model_name}.json",
                         {'task': self.task, 'inputShape': [1, 3, self.imgsz, self.imgsz], 
                          'classes': classes, 'metrics': self.metrics})

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
    
    def compile(self, project, classes, device='hailo8'):
        self.client.download_file(f"{project}/{self.model_name}.onnx", f"{self.model_name}.onnx")
        print("Compile model for edge deployment to hailo hef format...")
        print(f"hailomz compile yolov8n --ckpt yolov8n.onnx --calib-path {project}/train/images --classes {len(classes)} --hw-arch {device} --performance")
