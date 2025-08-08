from ..client import Abraia

import os
import numpy as np
#os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO


abraia = Abraia()


def sorted_folders(dir):
    items = [os.path.join(dir, name) for name in os.listdir(dir)]
    sorted_items = sorted(items, key=os.path.getctime)
    return sorted_items


def build_model_name(model_name, task):
    if task == 'segment':
        model_name = f"{model_name}-seg"
    if task == 'classify':
        model_name = f"{model_name}-cls"
    return model_name


class Model:
    def __init__(self, task, model_type='yolov8n'):
        model_name = build_model_name(model_type, task)
        self.model = YOLO(f"{model_name}.pt", verbose=False)
        self.model_name = model_name
        self.task = task

    def train(self, dataset, epochs=100, batch=32, imgsz=640):
        data = f"{dataset}" if self.task == 'classify' else f"{dataset}/data.yaml"
        results = self.model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz)
        metrics = self.model.val(data=data)
        return metrics

    def save(self, dataset, classes, imgsz=640, device="cpu"):
        model_src = self.model.export(format="onnx", device=device)
        abraia.upload_file(model_src, f"{dataset}/{self.model_name}.onnx")
        abraia.save_json(f"{dataset}/{self.model_name}.json", {'task': self.task, 'inputShape': [1, 3, imgsz, imgsz], 'classes': classes})

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
                if self.task == 'segment':
                    object['polygon'] = results.masks[k].xy[0]
                objects.append(object)
        return objects
