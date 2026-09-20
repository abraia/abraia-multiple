"""Inference adapter for classification models exported by training."""

import os
import cv2
import numpy as np

from ..postprocess.classification import postprocess
from ..postprocess.detection import validate_model_config
from ..session import OnnxSessionMixin
from ...utils import load_json, resolve_model_file


def preprocess_resnet(img, input_shape=(1, 3, 224, 224), resize_size=256):
    """Match the Torchvision Resize/CenterCrop transform used in training."""
    image = np.asarray(img)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("ResNet classification expects an image with three channels")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    target_height, target_width = map(int, input_shape[2:4])
    resize_size = int(resize_size)
    height, width = image.shape[:2]
    if width < height:
        resized_width = resize_size
        resized_height = round(height * resize_size / width)
    else:
        resized_height = resize_size
        resized_width = round(width * resize_size / height)
    image = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    top = max(0, (resized_height - target_height) // 2)
    left = max(0, (resized_width - target_width) // 2)
    image = image[top:top + target_height, left:left + target_width]
    if image.shape[:2] != (target_height, target_width):
        image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
    image = image.astype(np.float32) / 255
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    return image.transpose((2, 0, 1))[None].astype(np.float32)


class ResNetClassifier(OnnxSessionMixin):
    """Run ResNet18/50/101 ONNX models exported by classification training."""

    def __init__(self, model_uri, providers=None, accelerator=None):
        model_uri = os.fspath(model_uri)
        if os.path.isabs(model_uri) and not os.path.isfile(model_uri):
            raise FileNotFoundError(f"Model file not found: {model_uri}")
        config_uri = f"{os.path.splitext(model_uri)[0]}.json"
        model_path = resolve_model_file(model_uri)
        config_path = resolve_model_file(config_uri)
        self.config = load_json(config_path)
        self.task, self.input_shape, self.classes = validate_model_config(self.config)
        if self.task != "classification":
            raise ValueError("ResNet models require the classification task")
        self._init_onnx_session(
            model_path,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(self, img, top_k=1, score_threshold=None, labels=None,
            conf_threshold=None, **kwargs):
        """Classify one image with the exported ResNet model."""
        del kwargs
        self._ensure_open()
        if score_threshold is None:
            score_threshold = conf_threshold
        input_tensor = preprocess_resnet(img, self.input_shape)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return postprocess(
            outputs,
            self.classes,
            top_k=top_k,
            score_threshold=score_threshold,
            labels=labels,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["ResNetClassifier", "preprocess_resnet"]
