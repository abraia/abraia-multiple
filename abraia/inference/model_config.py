"""Built-in model kinds, defaults, and runtime option definitions."""

import os
from pathlib import Path

from ..tasks import HAILO_TASKS, PIPELINE_TASKS


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

# Public pipeline catalog shared by runtime validation and Studio's editor.
# Keep this next to the model backend definitions so adding a model cannot
# silently leave the configuration editor and runtime out of sync.
PIPELINE_MODEL_KINDS = (
    "onnx",
    "object_detection",
    "instance_segmentation",
    "pose",
    "classification",
    "resnet",
    "hailo",
    "face",
    "license_plate",
    "ocr",
)
PIPELINE_MODEL_TASKS = PIPELINE_TASKS
PIPELINE_HAILO_TASKS = HAILO_TASKS
MODEL_DEFAULT_THRESHOLDS = {
    ("onnx", "detection"): (0.25, 0.45),
    ("object_detection", "detection"): (0.25, 0.7),
    ("instance_segmentation", "detection"): (0.25, 0.7),
    ("pose", "pose"): (0.25, 0.7),
    ("classification", "classification"): (0.25, 0.0),
    ("resnet", "classification"): (0.25, 0.0),
    ("hailo", "detection"): (0.25, 0.7),
    ("hailo", "segmentation"): (0.25, 0.7),
    ("hailo", "pose"): (0.25, 0.7),
    ("face", "detection"): (0.75, 0.5),
    ("face", "recognition"): (0.45, 0.5),
    ("license_plate", "detection"): (0.5, 0.1),
    ("license_plate", "recognition"): (0.85, 0.15),
    ("ocr", "recognition"): (0.5, 0.0),
}


def model_defaults(kind, task):
    """Return editor defaults for a model kind/task pair."""
    return MODEL_DEFAULT_THRESHOLDS.get((kind, task), (0.25, 0.45))


def model_control_visibility(kind, task):
    """Return which pipeline-editor controls apply to a model."""
    is_model_uri = kind in (
        "onnx",
        "object_detection",
        "instance_segmentation",
        "pose",
        "classification",
        "resnet",
        "hailo",
    )
    return {
        "uri": is_model_uri,
        "index": kind == "face" and task == "recognition",
        "labels": is_model_uri,
        "confidence": True,
        "iou": (
            not (kind == "face" and task == "recognition")
            and kind not in ("ocr", "classification", "resnet")
        ),
        "approx": kind == "instance_segmentation" and task == "detection",
    }


def _sized_model_options(kind, label, task):
    size_labels = {"small": "Small", "medium": "Medium", "large": "Large"}
    return tuple(
        (
            f"{kind}_{size}",
            f"{label} ({size_labels[size]})",
            kind,
            task,
            uri,
        )
        for size, uri in MODEL_SIZE_URIS[kind].items()
    )


PIPELINE_MODEL_OPTIONS = (
    *_sized_model_options("object_detection", "Object detection", "detection"),
    *_sized_model_options("instance_segmentation", "Instance segmentation", "detection"),
    *_sized_model_options("pose", "Pose estimation", "pose"),
    ("resnet", "ResNet classification", "resnet", "classification", ""),
    ("face_detection", "Face detection", "face", "detection", ""),
    ("face_recognition", "Face recognition", "face", "recognition", ""),
    ("license_plate_detection", "License plate detection", "license_plate", "detection", ""),
    ("license_plate_recognition", "License plate recognition", "license_plate", "recognition", ""),
    ("ocr_recognition", "OCR recognition", "ocr", "recognition", ""),
)


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
    "MODEL_DEFAULT_THRESHOLDS",
    "ONNX_MODEL_KINDS",
    "PIPELINE_HAILO_TASKS",
    "PIPELINE_MODEL_KINDS",
    "PIPELINE_MODEL_OPTIONS",
    "PIPELINE_MODEL_TASKS",
    "RESNET_MODEL_KINDS",
    "is_resnet_model_uri",
    "model_control_visibility",
    "model_defaults",
]
