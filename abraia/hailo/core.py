from __future__ import annotations
"""Core helpers: arch detection, buffer utils, model resolution."""

import os
import shlex
import requests
import subprocess

from pathlib import Path
from typing import Any, Optional, Tuple, Dict

from ..utils import download_url

import logging
logger = logging.getLogger(__name__)

# Base Defaults
HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
HAILO10H_ARCH = "hailo10h"
HAILO_FILE_EXTENSION = ".hef"
HAILO_MODEL_ZOO_DEFAULT_VERSION = "v2.17.0"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
RESOURCES_MODELS_DIR_NAME = "models"

# Core defaults
RPI_POSSIBLE_NAME = "Raspberry Pi"
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
USB_CAMERA = "usb"

# Queue and async inference defaults
MAX_INPUT_QUEUE_SIZE = 60
MAX_OUTPUT_QUEUE_SIZE = 60
MAX_ASYNC_INFER_JOBS = 20

# Image / camera defaults
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}

# Base project paths
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

RESOURCES_CONFIG = {}
RESOURCES_CONFIG["object_detection"] = {
    "models": {
        "hailo8": {
            "default": [{"name": "yolov8m", "source": "mz"}],
            "extra": [
                {"name": "yolov5m_wo_spp", "source": "mz"},
                {"name": "yolov8s", "source": "mz"},
                {"name": "yolov5s", "source": "mz"},
                {"name": "yolov5m", "source": "mz"},
                {"name": "yolov6n", "source": "mz"},
                {"name": "yolov7", "source": "mz"},
                {"name": "yolov8n", "source": "mz"},
                {"name": "yolov8l", "source": "mz"},
                {"name": "yolov8x", "source": "mz"},
                {"name": "yolov9c", "source": "mz"},
                {"name": "yolov10n", "source": "mz"},
                {"name": "yolov10s", "source": "mz"},
                {"name": "yolov10b", "source": "mz"},
                {"name": "yolov10x", "source": "mz"},
                {"name": "yolov11n", "source": "mz"},
                {"name": "yolov11s", "source": "mz"},
                {"name": "yolov11m", "source": "mz"},
                {"name": "yolov11l", "source": "mz"},
                {"name": "yolov11x", "source": "mz"},
                {"name": "yolov8s-hailo8-barcode", "url": "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8s-hailo8l-barcode.hef", "source": "s3"}
            ]
        },
        "hailo8l": {
            "default": [{"name": "yolov8s", "source": "mz"}],
            "extra": [
                {"name": "yolov5s", "source": "mz"},
                {"name": "yolov5m", "source": "mz"},
                {"name": "yolov6n", "source": "mz"},
                {"name": "yolov7", "source": "mz"},
                {"name": "yolov5m_wo_spp", "source": "mz"},
                {"name": "yolov8n", "source": "mz"},
                {"name": "yolov8m", "source": "mz"},
                {"name": "yolov8l", "source": "mz"},
                {"name": "yolov8x", "source": "mz"},
                {"name": "yolov9c", "source": "mz"},
                {"name": "yolov10n", "source": "mz"},
                {"name": "yolov10s", "source": "mz"},
                {"name": "yolov10b", "source": "mz"},
                {"name": "yolov10x", "source": "mz"},
                {"name": "yolov11n", "source": "mz"},
                {"name": "yolov11s", "source": "mz"},
                {"name": "yolov11m", "source": "mz"},
                {"name": "yolov11l", "source": "mz"},
                {"name": "yolov11x", "source": "mz"},
                {"name": "yolov8s-hailo8l-barcode", "url": "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8l_rpi/yolov8s-hailo8l-barcode.hef", "source": "s3"}
            ]
        },
        "hailo10h": {
            "default": [{"name": "yolov8m", "source": "mz"}],
            "extra": [
                {"name": "yolov5s", "source": "mz"},
                {"name": "yolov5m", "source": "mz"},
                {"name": "yolov6n", "source": "mz"},
                {"name": "yolov7", "source": "mz"},
                {"name": "yolov7x", "source": "mz"},
                {"name": "yolov8s", "source": "mz"},
                {"name": "yolov8n", "source": "mz"},
                {"name": "yolov8l", "source": "mz"},
                {"name": "yolov8x", "source": "mz"},
                {"name": "yolov9c", "source": "mz"},
                {"name": "yolov10n", "source": "mz"},
                {"name": "yolov10s", "source": "mz"},
                {"name": "yolov10b", "source": "mz"},
                {"name": "yolov10x", "source": "mz"},
                {"name": "yolov11n", "source": "mz"},
                {"name": "yolov11s", "source": "mz"},
                {"name": "yolov11m", "source": "mz"},
                {"name": "yolov11l", "source": "mz"},
                {"name": "yolov11x", "source": "mz"}
            ]
        }
    }
}
RESOURCES_CONFIG["instance_segmentation"] = {
    "models": {
        "hailo8": {
            "default": [
                {"name": "yolov5m_seg_with_nms", "source": "s3"}
            ],
            "extra": [
                {"name": "yolov5m_seg", "source": "mz"},
                {"name": "yolov5l_seg", "source": "mz"},
                {"name": "yolov5n_seg", "source": "mz"},
                {"name": "yolov5s_seg", "source": "mz"},
                {"name": "yolov8m_seg", "source": "mz"},
                {"name": "yolov8n_seg", "source": "mz"},
                {"name": "yolov8s_seg", "source": "mz"},
                {"name": "fast_sam_s", "source": "s3"}
            ]
        },
        "hailo8l": {
            "default": [{"name": "yolov5n_seg", "source": "mz"}],
            "extra": [
                {"name": "yolov5l_seg", "source": "mz"},
                {"name": "yolov5m_seg", "source": "mz"},
                {"name": "yolov5s_seg", "source": "mz"},
                {"name": "yolov8m_seg", "source": "mz"},
                {"name": "yolov8n_seg", "source": "mz"},
                {"name": "yolov8s_seg", "source": "mz"}
            ]
        },
        "hailo10h": {
            "default": [
                {"name": "yolov5m_seg_with_nms", "source": "s3"}
            ],
            "extra": [
                {"name": "yolov5n_seg_with_nms", "source": "s3"},
                {"name": "yolov5s_seg_with_nms", "source": "s3"},
                {"name": "yolov5m_seg", "source": "mz"},
                {"name": "yolov5l_seg", "source": "mz"},
                {"name": "yolov5n_seg", "source": "mz"},
                {"name": "yolov5s_seg", "source": "mz"},
                {"name": "yolov8m_seg", "source": "mz"},
                {"name": "yolov8n_seg", "source": "mz"},
                {"name": "yolov8s_seg", "source": "mz"}
            ]
        }
    }
}
RESOURCES_CONFIG["pose_estimation"] = {
    "models": {
        "hailo8": {
            "default": [{"name": "yolov8m_pose", "source": "mz"}],
            "extra": [{"name": "yolov8s_pose", "source": "mz"}]
        },
        "hailo8l": {
            "default": [{"name": "yolov8s_pose", "source": "mz"}]
        },
        "hailo10h": {
            "default": [{"name": "yolov8m_pose", "source": "mz"}],
            "extra": [{"name": "yolov8s_pose", "source": "mz"}]
        }
    }
}

DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get Content-Length of a remote file via HEAD request."""
    try:
        r = requests.head(url, headers={'User-Agent': DEFAULT_USER_AGENT}, timeout=timeout, allow_redirects=True)
        length = r.headers.get('Content-Length')
        return int(length) if length else None
    except Exception:
        return None


def get_model_url(app_name, model_name, hailo_arch):
    """Return a download task tuple for a specific app model, or None if not found."""
    app_cfg = RESOURCES_CONFIG.get(app_name, {}).get("models", {}).get(hailo_arch, {})
    for tier in ["default", "extra"]:
        entries = app_cfg.get(tier, [])
        for e in entries:
            name = e.get("name") if isinstance(e, dict) else {}
            if name == model_name:
                url = e.get("url")
                if not url:
                    s3_arch = "h8l" if hailo_arch == HAILO8L_ARCH else "h8"
                    source = e.get("source", "mz")
                    if source == "s3":
                        url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
                    elif source == "mz":
                        url = f"{MODEL_ZOO_URL}/{HAILO_MODEL_ZOO_DEFAULT_VERSION}/{hailo_arch}/{name}{HAILO_FILE_EXTENSION}"
                if url:
                    dest_name = name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION
                    dest = Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_MODELS_DIR_NAME / hailo_arch / dest_name
                    return url, dest
    logger.warning(f"Model '{model_name}' not found for app '{app_name}'")
    return None, None


def execute_download(url, dest_path):
    """Execute a single download task."""
    remote_size = get_remote_file_size(url)

    if dest_path.exists() and remote_size and dest_path.stat().st_size == remote_size:
        logger.info("File already exists and valid")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading: {url}")
        download_url(url, str(dest_path))
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        logger.warning(f"Failed to download {url}: {e}")


def get_resource_path(resource_type: str, name: str, arch: str | None = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch:
            raise RuntimeError("Could not detect Hailo architecture for model path.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)

    return root / resource_type / name


def _get_default_model(app_name: str, arch: str) -> Optional[str]:
    default_entries = RESOURCES_CONFIG.get(app_name, {}).get("models", {}).get(arch, {}).get("default", [])
    for entry in default_entries:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if isinstance(name, str) and name.lower() != "none":
            return name
    return None


def resolve_hef_path(hef_path: Optional[str], app_name: str, arch: Optional[str] = None) -> Optional[Path]:
    """Resolve HEF path, downloading it if necessary."""
    arch = arch or detect_hailo_arch()
    if not arch:
        raise RuntimeError("Could not detect Hailo architecture.")
    logger.debug(f"App Name: {app_name}, Requested HEF: {hef_path}, Arch: {arch}")
    
    if hef_path is None:
        default_model = _get_default_model(app_name, arch)
        if not default_model:
            logger.error(f"No default model found for {app_name}/{arch}")
            return None
        hef_path = default_model
        logger.info(f"Using default model: {default_model}")

    path = Path(hef_path)
    if path.exists():
        return path.resolve()

    if not path.suffix:
        candidate = path.with_suffix(HAILO_FILE_EXTENSION)
        if candidate.exists():
            return candidate.resolve()

    model_name = path.stem
    resource_path = get_resource_path(RESOURCES_MODELS_DIR_NAME, model_name, arch)
    if resource_path.exists():
        return resource_path

    logger.warning(f"\n⚠️  WARNING: Model '{model_name}' not found. Downloading for {app_name}/{arch}...")
    try:
        url, dest = get_model_url(app_name, model_name, arch)
        if url and dest:
            execute_download(url, dest)
            if dest.exists():
                return dest
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")

    logger.error(f"Model '{model_name}' not found.")
    return None


# =============================================================================
# Hardware Detection
# =============================================================================

def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return RPI_POSSIBLE_NAME in f.read()
    except Exception:
        return False


def detect_hailo_arch() -> Optional[str]:
    """Detect the connected Hailo device architecture."""
    logger.debug("Detecting Hailo architecture...")
    try:
        res = subprocess.run(shlex.split(HAILO_FW_CONTROL_CMD), capture_output=True, text=True)
        if res.returncode != 0:
            return None
        
        stdout = res.stdout.upper()
        if HAILO8L_ARCH_CAPS in stdout:
            return HAILO8L_ARCH
        if HAILO8_ARCH_CAPS in stdout:
            return HAILO8_ARCH
        if HAILO10H_ARCH_CAPS in stdout or HAILO15H_ARCH_CAPS in stdout:
            return HAILO10H_ARCH
    except Exception as e:
        logger.error(f"Error detecting Hailo architecture: {e}")
        
    logger.warning("Could not determine Hailo architecture.")
    return None
