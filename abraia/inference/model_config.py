"""Built-in model kinds, defaults, and runtime option definitions."""

import os
from pathlib import Path


ONNX_MODEL_KINDS = frozenset({
    "onnx",
    "object_detection",
    "instance_segmentation",
    "pose",
})
GROUNDING_DINO_MODEL_KINDS = frozenset({"grounding_dino", "groundingdino"})
HAILO_MODEL_KINDS = frozenset({"hailo"})
RESNET_MODEL_KINDS = frozenset({"classification", "resnet", "resnet_classifier"})
MODEL_RUN_OPTIONS = (
    "labels",
    "conf_threshold",
    "iou_threshold",
    "approx",
    "top_k",
    "score_threshold",
    "prompt",
    "text_prompt",
    "box_threshold",
    "text_threshold",
)


MODEL_SIZE_URIS = {
    "object_detection": {
        "small": "multiple/models/yolov8n.onnx",
        "medium": "multiple/models/yolov8m.onnx",
        "large": "multiple/models/yolov8l.onnx",
    },
    "instance_segmentation": {
        "small": "multiple/models/yolov8n-seg.onnx",
        "medium": "multiple/models/yolov8m-seg.onnx",
        "large": "multiple/models/yolov8l-seg.onnx",
    },
    "pose": {
        "small": "multiple/models/yolov8n_pose.onnx",
        "medium": "multiple/models/yolov8m_pose.onnx",
        "large": "multiple/models/yolov8l_pose.onnx",
    },
}

DEFAULT_MODEL_URIS = {
    kind: sizes["small"] for kind, sizes in MODEL_SIZE_URIS.items()
}


def is_resnet_model_uri(uri):
    """Return whether a URI names a model produced by classification training."""
    if not uri:
        return False
    stem = Path(os.fspath(uri)).stem.lower()
    return stem in {"resnet18", "resnet50", "resnet101"}


__all__ = [
    "DEFAULT_MODEL_URIS",
    "GROUNDING_DINO_MODEL_KINDS",
    "HAILO_MODEL_KINDS",
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "ONNX_MODEL_KINDS",
    "RESNET_MODEL_KINDS",
    "is_resnet_model_uri",
]
