
from __future__ import print_function, division
from .multiple import Multiple, tempdir
from .detect import get_color

import os
import shutil
import itertools
from PIL import Image
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split


multiple = Multiple()


def load_projects():
    folders = multiple.list_files()[1]
    return [folder['name'] for folder in folders if folder['name'] not in ('export', '.export')]


def load_annotations(dataset):
    try:
        annotations = multiple.load_json(f"{dataset}/annotations.json")
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
    label, box, polygon = False, False, False
    for annotation in annotations:
        for object in annotation.get('objects', []):
            if 'polygon' in object:
                polygon = True
            elif 'box' in object:
                box = True
            elif 'label' in object:
                label = True
    if polygon:
        return 'segment'
    if box:
        return 'detect'
    if label:
        return 'classify'


def download_file(path, folder):
    dest = os.path.join(folder, os.path.basename(path))
    if not os.path.exists(dest):
        multiple.download_file(path, dest)
    return dest


def save_annotation(annotation, folder, classes, task):
    if task == 'classify':
        for object in annotation.get('objects', []):
            label = object.get('label')
            if label:
                src = os.path.join(folder, annotation['filename'])
                dest = os.path.join(folder, label, annotation['filename'])
                shutil.move(src, dest)
    else:
        im = Image.open(os.path.join(folder, 'images', annotation['filename']))
        label_lines = []
        for object in annotation.get('objects', []):
            label, bbox, cords = object.get('label'), object.get('box'), object.get('polygon')
            # Convert polygon or box to yolo format
            label_line = ''
            if task == 'segment' and cords:
                label_line = f"{classes.index(label)} " + ' '.join([f"{cord[0] / im.width} {cord[1] / im.height}" for cord in cords])
            elif task == 'detect' and bbox:
                label_line = f"{classes.index(label)} {(bbox[0] + bbox[2] / 2) / im.width} {(bbox[1] + bbox[3] / 2) / im.height} {bbox[2] / im.width} {bbox[3] / im.height}"
            elif task == 'classify':
                label_line = f"{classes.index(label)}"
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


# TORCH model

import torch
from torchvision import transforms
from abraia import torch as t


def t_create_dataset(dataset):
    imgsz = 224
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(imgsz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: t.Dataset(os.path.join(dataset), data_transforms[x]) for x in ['train', 'val']}
    # image_datasets = {x: t.Dataset(os.path.join(dataset, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, class_names


def t_train_model(dataloaders, class_names):
    model_conv = t.create_model(class_names)
    model = t.train_model(model_conv, dataloaders, num_epochs=25)
    return model


def t_save_model(model, model_name, dataset, classes):
    imgsz = 224
    t.export_onnx(f"{dataset}/{model_name}.onnx", model)
    multiple.save_json(f"{dataset}/{model_name}.json", {'inputShape': [1, 3, imgsz, imgsz], 'classes': classes})


#os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO


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


def train_model(dataset, task, batch=32, epochs=100, imgsz=640):
    model_name = build_model_name('yolov8n', task)
    model = YOLO(f"{model_name}.pt", verbose=False)
    data = f"{dataset}" if task == 'classify' else f"{dataset}/data.yaml"
    results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz)
    metrics = model.val(data=data)
    return model, model_name


def save_model(model, model_name, dataset, task, classes, imgsz=640):
    model_src = model.export(format="onnx", device="cpu")
    multiple.upload_file(model_src, f"{dataset}/{model_name}.onnx")
    multiple.save_json(f"{dataset}/{model_name}.json", {'task': task, 'inputShape': [1, 3, imgsz, imgsz], 'classes': classes})


def run_model(model, src, task='segment'):
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
