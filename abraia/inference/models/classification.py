"""Preprocessing and decoding for image-classification models."""

import cv2
import numpy as np

from ..geometry import normalize, softmax
from .detection import validate_model_config
from ..session import OnnxSessionMixin, load_onnx_model_config


def preprocess(img, size=224):
    """Resize and center-crop an image to a classifier input size."""
    if isinstance(size, (tuple, list)):
        target_height, target_width = map(int, size)
    else:
        target_height = target_width = int(size)
    scale = max(target_width / img.shape[1], target_height / img.shape[0])
    width = max(target_width, round(scale * img.shape[1]))
    height = max(target_height, round(scale * img.shape[0]))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    img = img[top:top + target_height, left:left + target_width]
    img = normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.expand_dims(img.transpose((2, 0, 1)), axis=0)


def postprocess(outputs, classes, top_k=1, score_threshold=None, labels=None):
    """Decode classification logits into sorted class predictions."""
    logits = np.asarray(outputs[0]).reshape(-1)
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    results = []
    for idx in order[:int(top_k)]:
        class_id = int(idx)
        score = float(probs[idx])
        if score_threshold is not None and score < score_threshold:
            continue
        if labels and classes[class_id] not in labels:
            continue
        results.append({
            "label": classes[class_id],
            "score": score,
            "class_id": class_id,
        })
    return results


def preprocess_resnet(img, input_shape=(1, 3, 224, 224), resize_size=256):
    """Match the Torchvision Resize/CenterCrop transform used in training."""
    image = np.asarray(img)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("ResNet classification expects an image with three channels")
    image = image[:, :, :3]
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
        model_path, self.config = load_onnx_model_config(model_uri)
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

__all__ = ["ResNetClassifier", "postprocess", "preprocess", "preprocess_resnet"]
