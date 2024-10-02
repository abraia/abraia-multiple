
from ..client import Abraia
from ..utils import get_color
from . import classify, detect 

import os
import shutil
import itertools
from PIL import Image
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split


abraia = Abraia()


def load_projects():
    folders = abraia.list_files()[1]
    return [folder['name'] for folder in folders if abraia.check_file(f"{folder['name']}/annotations.json")]


def load_annotations(dataset):
    try:
        annotations = abraia.load_json(f"{dataset}/annotations.json")
        for annotation in annotations:
            annotation['path'] = f"{dataset}/{annotation['filename']}"
        return annotations
    except:
        return []


def load_labels(annotations):
    labels = []
    for annotation in annotations:
        for object in annotation.get('objects', []):
            label = object.get('label')
            if label and label not in labels:
                labels.append(label)
    return list(set(labels))


def load_task(annotations):
    classify, detect, segment = False, False, False
    for annotation in annotations:
        for object in annotation.get('objects', []):
            if 'polygon' in object:
                segment = True
            elif 'box' in object:
                detect = True
            elif 'label' in object:
                classify = True
    if segment:
        return 'segment'
    if detect:
        return 'detect'
    if classify:
        return 'classify'


def load_tasks(annotations):
    tasks = ['classify', 'detect', 'segment']
    task = load_task(annotations)
    if task:
        idx = tasks.index(task)
        return tasks[:idx+1]
    return []


def download_file(path, folder):
    dest = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(dest):
        abraia.download_file(path, dest)
    return dest


def save_annotation(annotation, folder, classes, task):
    if task == 'classify':
        for object in annotation.get('objects', []):
            label = object.get('label')
            if label:
                src = os.path.join(folder, annotation['filename'])
                dest = os.path.join(folder, label, annotation['filename'])
                shutil.move(src, dest)
                break
    else:
        im = Image.open(os.path.join(folder, 'images', annotation['filename']))
        label_lines = []
        for object in annotation.get('objects', []):
            label, box, polygon = object.get('label'), object.get('box'), object.get('polygon')
            # Convert polygon or box to yolo format
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
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))


def save_data(annotations, folder, classes, task):
    if (task == 'classify'):
        os.makedirs(os.path.join(folder), exist_ok=True)
        paths = [annotation['path'] for annotation in annotations]
        process_map(download_file, paths, itertools.repeat(folder), max_workers=5)
        for label in classes:
            os.makedirs(os.path.join(folder, label), exist_ok=True)
        for annotation in annotations:
            save_annotation(annotation, folder, classes, task)
    else:
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        paths = [annotation['path'] for annotation in annotations]
        process_map(download_file, paths, itertools.repeat(os.path.join(folder, 'images')), max_workers=5)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)
        for annotation in annotations:
            save_annotation(annotation, folder, classes, task)


def save_config(dataset, classes):
    yaml_content = f'''
    train: {os.path.join(os.getcwd(), dataset, 'train/images')}
    val: {os.path.join(os.getcwd(), dataset, 'val/images')}
    test: {os.path.join(os.getcwd(), dataset, 'test/images')}
    names: {classes}
    '''
    path = os.path.join(dataset, 'data.yaml')
    with open(path, 'w') as f:
        f.write(yaml_content)


def split_dataset(annotations):
    train, test = train_test_split(annotations, test_size=0.3)
    val, test = train_test_split(test, test_size=0.5)
    return train, val, test


def create_dataset(dataset, task, classes):
    #shutil.rmtree(f"{dataset}/")
    # TODO: If folder exists skip or recreate based on cached files?
    annotations = load_annotations(dataset)
    train, val, test = split_dataset(annotations)
    data_annotations = {'train': train, 'val': val, 'test': test}
    #TODO: Download files in one single step
    for x in ['train', 'val', 'test']:
        save_data(data_annotations[x], f"{dataset}/{x}", classes, task)
    save_config(dataset, classes)
