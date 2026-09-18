import os
import copy
import sys
import time
import onnx
import torch
import numpy as np

from torchvision import models, transforms, datasets

from ..utils import temporal_src
from ..tasks import normalize_model_size
from .core import _resolve_client


CLASSIFICATION_BACKBONES = {
    "small": ("resnet18", models.resnet18, models.ResNet18_Weights),
    "medium": ("resnet50", models.resnet50, models.ResNet50_Weights),
    "large": ("resnet101", models.resnet101, models.ResNet101_Weights),
}


def default_device():
    """Return the preferred training device without storing global state."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_model(class_names, pretrained=True, device=None, model_size="small"):
    device = device or default_device()
    model_size = normalize_model_size(model_size)
    _name, model_factory, weights_enum = CLASSIFICATION_BACKBONES[model_size]
    model = model_factory(
        weights=weights_enum.IMAGENET1K_V1 if pretrained else None
    )
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.to(device)
    return model


# License: BSD
# Author: Sasank Chilamkurthy

def train_model(model, dataloaders, criterion=None, optimizer=None, scheduler=None,
                num_epochs=25, callback=None, device=None, is_cancelled=None):
    device = device or next(model.parameters()).device
    criterion = criterion or torch.nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = scheduler or torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Training canceled")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                if is_cancelled and is_cancelled():
                    raise RuntimeError("Training canceled")
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
            if not callback:
                print(f"Epoch {epoch+1}/{num_epochs} [{phase}] loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
            if callback and phase == 'val':
                callback({'epoch': epoch, 'epochs': num_epochs, 'loss': epoch_loss, 'acc': float(epoch_acc)})
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    if not callback:
        print(f"Best val Acc: {best_acc:.4f}")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class Model:
    def __init__(self, client=None, model_size="small"):
        self.model_size = normalize_model_size(model_size)
        self.input_shape = [1, 3, 224, 224]
        self.model_name = CLASSIFICATION_BACKBONES[self.model_size][0]
        self.metrics = {}
        self.device = default_device()
        self.client = _resolve_client(client)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

    def create_dataset(self, dataset, batch=8):
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': self.transform,
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset, x), transform=data_transforms[x]) for x in ['train', 'val']}
        # Multiprocessing DataLoader workers started from Studio's background
        # thread can leave native locks behind on macOS. Keep the GUI path
        # single-process there; other platforms retain the faster loaders.
        num_workers = 0 if sys.platform == 'darwin' else 4
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
        classes = image_datasets['train'].classes
        return dataloaders, classes

    def train(self, dataset, epochs=25, batch=8, callback=None, is_cancelled=None):
        dataloaders, classes = self.create_dataset(dataset, batch=batch)
        model_conv = create_model(
            classes,
            device=self.device,
            model_size=self.model_size,
        )
        self.dataloaders = dataloaders
        self.classes = classes
        since = time.time()
        self.model = train_model(
            model_conv,
            dataloaders,
            num_epochs=epochs,
            callback=callback,
            device=self.device,
            is_cancelled=is_cancelled,
        )
        time_elapsed = time.time() - since
        if not callback:
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    def test(self, split='val', is_cancelled=None):
        self.model.eval()
        model_device = next(self.model.parameters()).device
        running_corrects = 0
        confusion_matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        with torch.no_grad():
            for inputs, labels in self.dataloaders[split]:
                if is_cancelled and is_cancelled():
                    raise RuntimeError("Training canceled")
                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        acc = running_corrects.double() / len(self.dataloaders[split].dataset)
        self.metrics = {'acc': float(acc), 'confusionMatrix': confusion_matrix.tolist()}
        return self.metrics

    def save(self, dataset, classes, device='cpu'):
        target_device = torch.device(device)
        self.model.to(target_device)
        self.device = target_device
        model_src = temporal_src(f"{dataset}/{self.model_name}.onnx")
        dummy_input = torch.randn(1, 3, 224, 224, device=target_device)
        torch.onnx.export(self.model, dummy_input, model_src, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])
        onnx_model = onnx.load(model_src)
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, model_src)
        self.client.upload_file(model_src, f"{dataset}/{self.model_name}.onnx")
        self.client.save_json(
            f"{dataset}/{self.model_name}.json",
            {
                'task': 'classification',
                'kind': 'resnet',
                'backbone': self.model_name,
                'inputShape': self.input_shape,
                'classes': classes,
                'metrics': self.metrics,
            },
        )

    def run(self, img):
        self.model.eval()
        input_tensor = self.transform(img)
        model_device = next(self.model.parameters()).device
        input_batch = input_tensor.unsqueeze(0).to(model_device)
        with torch.no_grad():
            output = self.model(input_batch)
            pred = torch.softmax(output.squeeze(0), dim=0)
        idx = int(pred.argmax())
        score = float(pred[idx])
        label = self.classes[idx]
        return [{'label': label, 'score': score}]
