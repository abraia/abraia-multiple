"""Hailo model metadata, resource lookup, and HEF resolution."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ...tasks import normalize_task
from ...utils import download_url, get_remote_file_size
from .device import HAILO8L_ARCH, detect_hailo_arch


logger = logging.getLogger(__name__)

HAILO_FILE_EXTENSION = ".hef"
HAILO_MODEL_ZOO_DEFAULT_VERSION = "v2.17.0"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
RESOURCES_MODELS_DIR_NAME = "models"

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def resolve_model_type(model_type, hef_path, task):
    """Resolve the postprocessor architecture for a Hailo model."""
    if model_type:
        return model_type
    normalized_task = normalize_task(task)
    if normalized_task not in ("detection", "segmentation", "pose"):
        raise ValueError(
            f"Unsupported Hailo task: {normalized_task}. "
            "Use detection, segmentation, or pose"
        )
    if normalized_task == "pose":
        return "v8"
    if normalized_task == "segmentation" and "yolov8" in str(hef_path).lower():
        return "v8"
    return "v5"

RESOURCES_CONFIG = {
    "detect": {
        "models": {
            "hailo8": {
                "default": [{"name": "yolov8m", "source": "mz"}],
                "extra": [
                    {"name": "yolov8n", "source": "mz"}, {"name": "yolov8s", "source": "mz"},
                    {"name": "yolov8l", "source": "mz"}, {"name": "yolov8x", "source": "mz"},
                    {"name": "yolov11n", "source": "mz"}, {"name": "yolov11s", "source": "mz"},
                    {"name": "yolov11m", "source": "mz"}, {"name": "yolov11l", "source": "mz"},
                    {"name": "yolov11x", "source": "mz"},
                ],
            },
            "hailo8l": {
                "default": [{"name": "yolov8s", "source": "mz"}],
                "extra": [
                    {"name": "yolov8n", "source": "mz"}, {"name": "yolov8m", "source": "mz"},
                    {"name": "yolov8l", "source": "mz"}, {"name": "yolov8x", "source": "mz"},
                    {"name": "yolov11n", "source": "mz"}, {"name": "yolov11s", "source": "mz"},
                    {"name": "yolov11m", "source": "mz"}, {"name": "yolov11l", "source": "mz"},
                    {"name": "yolov11x", "source": "mz"},
                ],
            },
        },
    },
    "segment": {
        "models": {
            "hailo8": {
                "default": [{"name": "yolov5m_seg_with_nms", "source": "s3"}],
                "extra": [
                    {"name": "yolov5m_seg", "source": "mz"}, {"name": "yolov5l_seg", "source": "mz"},
                    {"name": "yolov5n_seg", "source": "mz"}, {"name": "yolov5s_seg", "source": "mz"},
                    {"name": "yolov8n_seg", "source": "mz"}, {"name": "yolov8m_seg", "source": "mz"},
                    {"name": "yolov8s_seg", "source": "mz"},
                ],
            },
            "hailo8l": {
                "default": [{"name": "yolov5n_seg", "source": "mz"}],
                "extra": [
                    {"name": "yolov5l_seg", "source": "mz"}, {"name": "yolov5m_seg", "source": "mz"},
                    {"name": "yolov5s_seg", "source": "mz"}, {"name": "yolov8m_seg", "source": "mz"},
                    {"name": "yolov8n_seg", "source": "mz"}, {"name": "yolov8s_seg", "source": "mz"},
                ],
            },
        },
    },
    "pose": {
        "models": {
            "hailo8": {
                "default": [{"name": "yolov8m_pose", "source": "mz"}],
                "extra": [{"name": "yolov8s_pose", "source": "mz"}],
            },
            "hailo8l": {
                "default": [{"name": "yolov8s_pose", "source": "mz"}],
            },
        },
    },
}


def get_model_url(task, model_name, hailo_arch):
    """Return a download URL and destination for a catalog model."""
    app_cfg = RESOURCES_CONFIG.get(task, {}).get("models", {}).get(hailo_arch, {})
    for entry in app_cfg.get("default", []) + app_cfg.get("extra", []):
        name = entry.get("name")
        if name != model_name:
            continue
        url = entry.get("url")
        if not url:
            source = entry.get("source", "mz")
            if source == "s3":
                s3_arch = "h8l" if hailo_arch == HAILO8L_ARCH else "h8"
                url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
            elif source == "mz":
                url = f"{MODEL_ZOO_URL}/{HAILO_MODEL_ZOO_DEFAULT_VERSION}/{hailo_arch}/{name}{HAILO_FILE_EXTENSION}"
        if url:
            dest_name = name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION
            dest = Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_MODELS_DIR_NAME / hailo_arch / dest_name
            return url, dest
    logger.warning("Model '%s' not found for task '%s'", model_name, task)
    return None, None


def execute_download(url, dest_path):
    """Download a model unless the local file already has the remote size."""
    remote_size = get_remote_file_size(url)
    if dest_path.exists() and remote_size and dest_path.stat().st_size == remote_size:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading: %s", url)
        download_url(url, str(dest_path))
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        logger.warning("Failed to download %s: %s", url, exc)


def get_resource_path(resource_type: str, name: str, arch: Optional[str] = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch:
            raise RuntimeError("Could not detect Hailo architecture.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)
    return root / resource_type / name


def get_default_model(task: str, arch: str) -> Optional[str]:
    """Return the first configured default model for a task and architecture."""
    entries = RESOURCES_CONFIG.get(task, {}).get("models", {}).get(arch, {}).get("default", [])
    for entry in entries:
        name = entry.get("name")
        if isinstance(name, str) and name.lower() != "none":
            return name
    return None


def resolve_hef_path(hef_path: Optional[str], task: str,
                     arch: Optional[str] = None) -> Optional[Path]:
    """Resolve a local or catalog HEF path, downloading it when necessary."""
    arch = arch or detect_hailo_arch()
    if not arch:
        raise RuntimeError("Could not detect Hailo architecture.")

    if hef_path is None:
        hef_path = get_default_model(task, arch)
        if not hef_path:
            logger.error("No default model found for %s/%s", task, arch)
            return None
        logger.info("Using default model: %s", hef_path)

    path = Path(hef_path)
    if path.exists():
        return path.resolve()
    if not path.suffix and path.with_suffix(HAILO_FILE_EXTENSION).exists():
        return path.with_suffix(HAILO_FILE_EXTENSION).resolve()

    model_name = path.stem
    resource_path = get_resource_path(RESOURCES_MODELS_DIR_NAME, model_name, arch)
    if resource_path.exists():
        return resource_path

    logger.warning("Model '%s' not found. Downloading...", model_name)
    url, dest = get_model_url(task, model_name, arch)
    if url and dest:
        execute_download(url, dest)
        if dest.exists():
            return dest
    logger.error("Model '%s' not found.", model_name)
    return None


def get_labels(labels_path: Optional[str]) -> list:
    """Load labels from a file, falling back to COCO labels."""
    if labels_path is None or not os.path.exists(labels_path):
        return COCO_LABELS
    with open(labels_path, "r", encoding="utf-8") as labels_file:
        return labels_file.read().splitlines()


__all__ = [
    "HAILO_FILE_EXTENSION",
    "HAILO_MODEL_ZOO_DEFAULT_VERSION",
    "MODEL_ZOO_URL",
    "S3_RESOURCES_BASE_URL",
    "RESOURCES_ROOT_PATH_DEFAULT",
    "RESOURCES_MODELS_DIR_NAME",
    "COCO_LABELS",
    "RESOURCES_CONFIG",
    "resolve_model_type",
    "get_model_url",
    "execute_download",
    "get_resource_path",
    "get_default_model",
    "resolve_hef_path",
    "get_labels",
]
