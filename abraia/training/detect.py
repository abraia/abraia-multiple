from ..client import Abraia

import os
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
    def __init__(self):
        pass

    def train_model(self, dataset, task, batch=32, epochs=100, imgsz=640):
        model_name = build_model_name('yolov8n', task)
        model = YOLO(f"{model_name}.pt", verbose=False)
        data = f"{dataset}" if task == 'classify' else f"{dataset}/data.yaml"
        results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz)
        metrics = model.val(data=data)
        return model, model_name


    def save_model(self, model, model_name, dataset, task, classes, imgsz=640):
        model_src = model.export(format="onnx", device="cpu")
        abraia.upload_file(model_src, f"{dataset}/{model_name}.onnx")
        abraia.save_json(f"{dataset}/{model_name}.json", {'task': task, 'inputShape': [1, 3, imgsz, imgsz], 'classes': classes})


    def run_model(self, model, src, task='segment'):
        objects = []
        results = model.predict(src, verbose=False)[0]
        if results:
            for box, mask in zip(results.boxes, results.masks):
                class_id = int(box.cls)
                label = results.names[class_id]
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                object = {'label': label, 'confidence': confidence, 'color': get_color(class_id), 'box': [x1, y1, x2 - x1, y2 - y1]}
                if task == 'segment':
                    object['polygon'] = [(x, y) for x, y in mask.xy[0]]
                objects.append(object)
        return objects
