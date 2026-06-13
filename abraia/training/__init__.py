import os
import shutil
import itertools

from PIL import Image
from typing import Dict, Any
from tqdm.contrib.concurrent import process_map

from ..utils import save_text
from .dataset import list_datasets, load_dataset, search_images, list_models, download_file


def load_tasks(task):
    if task:
        tasks = ['classify', 'detect', 'segment']
        return tasks[:tasks.index(task)+1]
    return []


def save_annotation(annotation, folder, classes, task):
    im = Image.open(os.path.join(folder, 'images', annotation['filename']))
    label_lines = []
    for object in annotation.get('objects', []):
        label, box, polygon = object.get('label'), object.get('box'), object.get('polygon')
        if task == 'segment':
            if polygon:
                label_line = f"{classes.index(label)} " + ' '.join([f"{point[0] / im.width} {point[1] / im.height}" for point in polygon])
                label_lines.append(label_line)
        elif task == 'detect':
            if polygon:
                xx, yy = [point[0] for point in polygon], [point[1] for point in polygon]
                x1, y1, x2, y2 = min(xx), min(yy), max(xx), max(yy)
                box = [x1, y1, x2 - x1, y2 - y1]
            if box:
                label_line = f"{classes.index(label)} {(box[0] + box[2] / 2) / im.width} {(box[1] + box[3] / 2) / im.height} {box[2] / im.width} {box[3] / im.height}"
                label_lines.append(label_line)
    label_path = os.path.join(folder, 'labels',  f"{os.path.splitext(annotation['filename'])[0]}.txt")
    save_text(label_path, '\n'.join(label_lines))


def save_data(annotation, folder, classes, task):
    path = annotation['path']
    dest = folder if task == 'classify' else os.path.join(folder, 'images')
    if task == 'classify':
        label = next((obj.get('label', '') for obj in annotation.get('objects', [])), '')
        dest = os.path.join(dest, label)
    download_file(path, dest)
    if task != 'classify':
        save_annotation(annotation, folder, classes, task)


def save_config(dataset, classes):
    yaml_content = f'''
    train: {os.path.join(os.getcwd(), dataset, 'train/images')}
    val: {os.path.join(os.getcwd(), dataset, 'val/images')}
    test: {os.path.join(os.getcwd(), dataset, 'test/images')}
    names: {classes}
    '''
    path = os.path.join(dataset, 'data.yaml')
    save_text(path, yaml_content)

    
def prepare_dataset(dataset, force = False):
    if force or not os.path.exists(dataset.project):
        splits = list(zip(['train', 'val', 'test'], dataset.split()))
        all_annotations, all_folders = [], []
        for x, annotations in splits:
            folder = os.path.join(dataset.project, x)
            all_annotations.extend(annotations)
            all_folders.extend([folder] * len(annotations))
        process_map(save_data, all_annotations, all_folders, itertools.repeat(dataset.classes), itertools.repeat(dataset.task), max_workers=5, chunksize=1, desc="Downloading images")
        if dataset.task != 'classify':
            save_config(dataset.project, dataset.classes)


class ModelTrainer:
    """High-level trainer orchestrator using models and dataset utilities."""
    def __init__(self, project: str, task: str, classes: list, imgsz: int = None):
        self.project = project
        self.task = task
        self.classes = classes
        imgsz = imgsz or (224 if task == 'classify' else 640)
        if task == 'classify':
            from . import classify
            self.model = classify.Model()
        else:
            from . import detect
            self.model = detect.Model(task, imgsz=imgsz)

    def train(self, epochs: int = None, batch: int = 32, callback=None) -> None:
        epochs = epochs or (30 if self.task == 'classify' else 300)
        self.model.train(self.project, epochs=epochs, batch=batch, callback=callback)

    def test(self, split: str = 'val') -> Dict[str, Any]:
        return self.model.test(split=split)

    def save(self, device='cpu') -> None:
        self.model.save(self.project, self.classes, device=device)

    def run(self, img):
        return self.model.run(img)

    def compile(self, device='hailo8'):
        if self.task != 'detect':
            raise NotImplementedError("Model compilation is only implemented for detection models.")
        self.model.compile(self.project, self.classes, device=device)
    