"""Preprocessing and decoding for image-classification models."""

import cv2
import numpy as np

from .common import normalize, softmax


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
    for idx in order[:max(1, int(top_k))]:
        score = float(probs[idx])
        if score_threshold is not None and score < score_threshold:
            continue
        if int(idx) >= len(classes):
            continue
        if labels and classes[idx] not in labels:
            continue
        results.append({
            "label": classes[idx],
            "score": score,
            "class_id": int(idx),
        })
    return results


class ClassificationDecoder:
    """Decode image-classification logits for one model."""

    def __init__(self, classes):
        self.classes = list(classes)

    def __call__(self, outputs, **kwargs):
        return postprocess(outputs, self.classes, **kwargs)


__all__ = ["ClassificationDecoder", "postprocess", "preprocess"]
