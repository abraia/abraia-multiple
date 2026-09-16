import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from typing import Dict, Any
from tqdm import tqdm

from ..utils import save_text
from .ops import train_test_split
from .dataset import list_datasets, load_dataset, search_images, list_models, download_file, abraia
from .service import TrainingService
from ..tasks import TRAINING_TASKS, normalize_task, to_ultralytics_task


def save_annotation(annotation, folder, classes, task):
    task = normalize_task(task)
    im = Image.open(os.path.join(folder, 'images', annotation['filename']))
    label_lines = []
    for object in annotation.get('objects', []):
        label, box, polygon = object.get('label'), object.get('box'), object.get('polygon')
        if task == 'segmentation':
            if polygon:
                label_line = f"{classes.index(label)} " + ' '.join([f"{point[0] / im.width} {point[1] / im.height}" for point in polygon])
                label_lines.append(label_line)
        elif task == 'detection':
            if polygon:
                xx, yy = [point[0] for point in polygon], [point[1] for point in polygon]
                x1, y1, x2, y2 = min(xx), min(yy), max(xx), max(yy)
                box = [x1, y1, x2 - x1, y2 - y1]
            if box:
                label_line = f"{classes.index(label)} {(box[0] + box[2] / 2) / im.width} {(box[1] + box[3] / 2) / im.height} {box[2] / im.width} {box[3] / im.height}"
                label_lines.append(label_line)
    label_path = os.path.join(folder, 'labels',  f"{os.path.splitext(annotation['filename'])[0]}.txt")
    save_text(label_path, '\n'.join(label_lines))


def save_data(annotation, folder, classes, task, client=None):
    task = normalize_task(task)
    path = annotation['path']
    dest = folder if task == 'classification' else os.path.join(folder, 'images')
    if task == 'classification':
        label = next((obj.get('label', '') for obj in annotation.get('objects', [])), '')
        dest = os.path.join(dest, label)
    download_file(path, dest, client=client)
    if task != 'classification':
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


def split_dataset(annotations):
    backgrounds = [annotation for annotation in annotations if not annotation.get('objects')]
    annotations = [annotation for annotation in annotations if annotation.get('objects')]
    if not annotations:
        return backgrounds, [], []
    train, test = train_test_split(annotations, test_size=0.3, random_state=42)
    if test:
        val, test = train_test_split(test, test_size=0.5, random_state=42)
    else:
        val, test = [], []
    train.extend(backgrounds)
    return train, val, test

    
def prepare_dataset(dataset, force=False, callback=None):
    """Download and split a dataset, optionally reporting each file."""
    client = getattr(dataset, "client", abraia)
    if force or not os.path.exists(dataset.project):
        annotations = dataset.annotations
        dataset_path = f"{dataset.project}/dataset.json"
        if client.check_file(dataset_path):
            filenames = client.load_json(dataset_path)
            annotations = [a for a in annotations if a.get('filename') in filenames]
        splits = list(zip(['train', 'val', 'test'], split_dataset(annotations)))
        all_annotations, all_folders = [], []
        for x, annotations in splits:
            folder = os.path.join(dataset.project, x)
            all_annotations.extend(annotations)
            all_folders.extend([folder] * len(annotations))
        total = len(all_annotations)

        def report(current, annotation):
            if callback:
                callback({
                    "current": current,
                    "total": total,
                    "filename": annotation.get("filename", "image"),
                })

        if callback:
            callback({
                "current": 0,
                "total": total,
                "filename": "Starting download",
            })
        work = list(zip(all_annotations, all_folders))
        # Downloads are I/O-bound. Complete them in parallel while keeping
        # progress delivery on the caller thread.
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    save_data,
                    annotation,
                    folder,
                    dataset.classes,
                    dataset.task,
                    client,
                ): annotation
                for annotation, folder in work
            }
            completed = as_completed(futures)
            if callback is None:
                completed = tqdm(completed, total=total, desc="Downloading images")
            for current, future in enumerate(completed, start=1):
                future.result()
                report(current, futures[future])
        if normalize_task(dataset.task) != 'classification':
            save_config(dataset.project, dataset.classes)


class ModelTrainer:
    """High-level trainer orchestrator using models and dataset utilities."""
    def __init__(self, project: str, task: str, classes: list, imgsz: int = None,
                 client=None):
        task = normalize_task(task)
        if task not in TRAINING_TASKS:
            raise ValueError(f"Unsupported training task: {task}")
        self.project = project
        self.task = task
        self.classes = classes
        self.pbar = None
        imgsz = imgsz or (224 if task == 'classification' else 640)
        if task == 'classification':
            from . import classify
            self.model = classify.Model(client=client)
        else:
            from . import detect
            self.model = detect.Model(
                to_ultralytics_task(task), imgsz=imgsz, client=client
            )

    def _progress_callback(self, progress):
        if self.pbar is None:
            self.pbar = tqdm(total=progress['epochs'], initial=progress['epoch'])
        self.pbar.set_description(f"Loss: {progress['loss']:.4f} Acc: {progress['acc']:.4f}")
        self.pbar.update(1)

    def train(self, epochs: int = None, batch: int = 32, callback=None,
              is_cancelled=None) -> None:
        epochs = epochs or (30 if self.task == 'classification' else 300)
        callback = self._progress_callback if callback is None else callback
        try:
            self.model.train(
                self.project,
                epochs=epochs,
                batch=batch,
                callback=callback,
                is_cancelled=is_cancelled,
            )
        finally:
            if self.pbar:
                self.pbar.close()
                self.pbar = None

    def test(self, split: str = 'val', is_cancelled=None) -> Dict[str, Any]:
        if is_cancelled is None:
            return self.model.test(split=split)
        return self.model.test(split=split, is_cancelled=is_cancelled)

    def save(self, device='cpu') -> None:
        self.model.save(self.project, self.classes, device=device)

    def run(self, img):
        return self.model.run(img)

    def compile(self, device='hailo8'):
        if self.task != 'detection':
            raise NotImplementedError("Model compilation is only implemented for detection models.")
        self.model.compile(self.project, self.classes, device=device)
    
