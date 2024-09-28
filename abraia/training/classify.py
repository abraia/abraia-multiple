from ..client import Abraia
from ..utils import temporal_src, load_image

import onnx
import torch
import torchvision
from torchvision import models, transforms, datasets

import os
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange


abraia = Abraia()


torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_image(path):
    dest = abraia.download_file(path, cache=True)
    return load_image(dest)


def load_dataset(dataset, shuffle=True):
    paths, labels = [], []
    annotations = abraia.load_json(f"{dataset}/annotations.json")
    keys = list(filter(lambda k: k != 'filename', annotations[0].keys()))
    paths = [f"{dataset}/{annotation['filename']}" for annotation in annotations]
    labels = [annotation['label'] for annotation in annotations]
    if shuffle:
        ids = list(range(len(paths)))
        random.shuffle(ids)
        paths = [paths[id] for id in ids]
        labels = [labels[id] for id in ids]
    return paths, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        paths, labels = load_dataset(root_dir)
        self.paths = paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(np.sort(np.unique(labels)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.paths[idx]
        label = self.labels[idx]
        label = self.classes.index(label)
        img = read_image(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


def create_model(class_names, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.to(device)
    return model


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


# License: BSD
# Author: Sasank Chilamkurthy

def train_model(model, dataloaders, criterion=None, optimizer=None, scheduler=None, num_epochs=25):
    criterion = criterion or torch.nn.CrossEntropyLoss()
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer = optimizer or torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # for epoch in trange(num_epochs, desc="Epoch"):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_data(dataloader):
    class_names = dataloader.dataset.classes
    inputs, classes = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs)  # Make a grid from batch
    imshow(out, title=[class_names[x] for x in classes])


def visualize_model(model, dataloader, num_images=6):
    class_names = dataloader.dataset.classes
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


class Model:
    def __init__(self):
        self.imgsz = 224
        self.input_shape = [1, 3, self.imgsz, self.imgsz]
        self.model_name = 'resnet18'

    def create_dataset(self, dataset):
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.imgsz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.imgsz),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # image_datasets = {x: Dataset(os.path.join(dataset), data_transforms[x]) for x in ['train', 'val']}
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset, x), transform=data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val']}
        classes = image_datasets['train'].classes
        return dataloaders, classes

    def train(self, dataset, epochs=25):
        dataloaders, classes = self.create_dataset(dataset)
        model_conv = create_model(classes)
        self.classes = classes
        self.model = train_model(model_conv, dataloaders, num_epochs=epochs)
        return self.model

    def save(self, dataset, classes, device='cpu'):
        self.model.to(device)
        model_src = temporal_src(f"{dataset}/{self.model_name}.onnx")
        dummy_input = torch.randn(1, 3, self.imgsz, self.imgsz)
        torch.onnx.export(self.model, dummy_input, model_src, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
        onnx_model = onnx.load(model_src)
        onnx.checker.check_model(onnx_model)
        abraia.upload_file(model_src, f"{dataset}/{self.model_name}.onnx")
        abraia.save_json(f"{dataset}/{self.model_name}.json", {'inputShape': self.input_shape, 'classes': classes})

    def run(self, img):
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)
        output = self.model(input_batch)
        pred = torch.softmax(output.squeeze(0), dim=0)
        idx = int(pred.argmax())
        confidence = float(pred[idx])
        label = self.classes[idx]
        return [{'label': label, 'confidence': confidence}]
