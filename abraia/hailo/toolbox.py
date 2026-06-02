import os
import sys
import cv2
import time
import queue
import shlex
import logging
import requests
import threading
import subprocess
import collections
import numpy as np

from pathlib import Path
from functools import partial
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

from ..utils import download_url
from ..utils.draw import (
    render_resolution,
    render_status,
    calculate_optimal_thickness,
    calculate_optimal_text_scale,
)

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
    from hailo_platform.pyhailort.pyhailort import FormatOrder
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


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

HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"

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


def get_resource_path(resource_type: str, name: str, arch: Optional[str] = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch:
            raise RuntimeError("Could not detect Hailo architecture for model path.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)

    return root / resource_type / name


def get_default_model(app_name: str, arch: str) -> Optional[str]:
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
        default_model = get_default_model(app_name, arch)
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


def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return "Raspberry Pi" in f.read()
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


class PiCamera2CaptureAdapter:
    """
    Adapter that makes Picamera2 behave like cv2.VideoCapture.
    """

    def __init__(self, picam2):
        self.picam2 = picam2
        self._opened = True
        self._io_lock = threading.Lock()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        # prevent stop/close while capturing
        with self._io_lock:
            if not self._opened: # re-check after taking lock
                return False, None
            frame = self.picam2.capture_array()

        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            try:
                cfg = self.picam2.camera_configuration()
                size = cfg.get("main", {}).get("size", None)
                if size and len(size) == 2:
                    w, h = int(size[0]), int(size[1])
                    return float(w if prop_id == cv2.CAP_PROP_FRAME_WIDTH else h)
            except Exception:
                pass
            return 0.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return None

    def release(self):
        # stop new reads ASAP
        self._opened = False

        # wait if a read() is currently inside capture_array()
        with self._io_lock:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass

def get_source_fps(cap: Any, source_name: str) -> Optional[float]:
    """
    Read FPS from an opened capture source.
    """
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        logger.debug(f"{source_name} FPS not reported by source.")
        return None
    return source_fps


def open_cv_capture(src: Any, source_type: str) -> Any:
    """
    Open an OpenCV-based capture source.
    """
    if source_type == "video" and not os.path.exists(src):
        logger.error(f"File not found: {src}")
        sys.exit(1)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Failed to open {source_type} source: {src}")
        sys.exit(1)

    logger.info(f"Using {source_type} input: {src}")
    return cap


def open_usb_camera(input_src: str, resolution: Optional[str]):
    camera_index = int(str(input_src))
    cap = open_cv_capture(camera_index, "USB camera index")
    if resolution in CAMERA_RESOLUTION_MAP:
        width, height = CAMERA_RESOLUTION_MAP[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.debug(f"Camera resolution forced to {width}x{height}")
    return cap


def open_rpi_camera() -> Optional[Any]:
    """
    Open Raspberry Pi camera using Picamera2.
    """
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.error(f"Picamera2 not available: {e}")
        return None

    try:
        picam2 = Picamera2()
        width, height = 800, 600
        fps = 30
        main = {"size": (width, height), "format": "RGB888"}
        config = picam2.create_video_configuration(main=main, controls={"FrameRate": fps})

        picam2.configure(config)
        picam2.start()

        logger.debug(f"RPi camera started ({width}x{height}) @ {fps} FPS")

        return PiCamera2CaptureAdapter(picam2)

    except Exception as e:
        logger.error(f"Failed to open RPi camera: {e}")
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        return None


def is_stream_url(src: str) -> bool:
    """
    Return True if the input looks like a supported network stream URL.
    """
    src_lower = src.lower()
    return (
        src_lower.startswith("rtsp://")
        or src_lower.startswith("http://")
        or src_lower.startswith("https://")
    )


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    path = Path(images_path)
    def read_rgb(p: Path):
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        img = read_rgb(path)
        return [img] if img is not None else []
    elif path.is_dir():
        images = [read_rgb(img) for img in path.glob("*") if img.suffix.lower() in IMAGE_EXTENSIONS]
        return [img for img in images if img is not None]
    return []


def get_labels(labels_path: str) -> list:
    if labels_path is None or not os.path.exists(labels_path):
        return COCO_LABELS
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset, y_offset = (model_w - new_img_w) // 2, (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image
    return padded_image


class VideoInput:
    def __init__(
        self,
        input_src: str,
        batch_size: int = 1,
        resolution: Optional[str] = None,
        frame_rate: Optional[float] = None,
        video_unpaced: bool = False,
        stop_event: Optional[threading.Event] = None,
    ):
        self.input_src = input_src
        self.batch_size = batch_size
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.video_unpaced = video_unpaced
        self.stop_event = stop_event or threading.Event()

        self.input_type = "unknown"
        self.cap = None
        self.images = None
        self.source_fps = None
        self.width = None
        self.height = None

        self._init_input()

    def _init_input(self):
        src = str(self.input_src)

        if src.isdigit():
            if is_raspberry_pi():
                self.input_type = "rpi_camera"
                self.cap = open_rpi_camera()
                self.source_fps = 30
                logger.info("Using Raspberry Pi camera at 800x600, 30 FPS")
            else:
                self.input_type = "usb_camera"
                self.cap = open_usb_camera(src, self.resolution)
                self.source_fps = get_source_fps(self.cap, "USB camera")
        elif is_stream_url(src):
            self.input_type = "stream"
            self.cap = open_cv_capture(src, "stream")
            self.source_fps = get_source_fps(self.cap, "stream camera")
        elif os.path.exists(src):
            if any(src.lower().endswith(suffix) for suffix in VIDEO_SUFFIXES):
                self.input_type = "video"
                self.cap = open_cv_capture(src, "video")
                self.source_fps = get_source_fps(self.cap, "video file")
            else:
                self.input_type = "images"
                self.images = load_images_opencv(src)
        else:
            logger.error(f"Invalid input '{src}'.")
            sys.exit(1)

        if self.cap is not None:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    @property
    def has_capture(self) -> bool:
        return self.cap is not None

    @property
    def has_images(self) -> bool:
        return self.images is not None and len(self.images) > 0

    def _generate_frames(self) -> Generator[np.ndarray, None, None]:
        if self.has_images:
            for image in self.images:
                yield image
            return

        is_camera = self.input_type in ("usb_camera", "rpi_camera", "stream")
        is_video = self.input_type == "video"
        has_target_fps = self.frame_rate is not None and self.frame_rate > 0
        has_source_fps = self.source_fps is not None and self.source_fps > 0

        if is_video and self.video_unpaced and has_target_fps:
            logger.warning("--frame-rate is ignored when --video-unpaced is enabled.")

        should_drop = has_target_fps and (not has_source_fps or self.frame_rate < self.source_fps)
        if has_target_fps and has_source_fps and self.frame_rate >= self.source_fps:
            logger.warning(
                f"Requested frame rate ({self.frame_rate}) is greater than or equal to "
                f"the source FPS ({self.source_fps}); no frame dropping will be applied."
            )
        should_pace = is_video and not self.video_unpaced

        next_keep_timestamp = time.monotonic()
        keep_period = (1.0 / float(self.frame_rate) if is_camera and should_drop else None)
        video_start_ms, wall_start_time = None, None
        next_keep_video_ms = None
        video_keep_period_ms = (1000.0 / float(self.frame_rate) if is_video and should_drop and not self.video_unpaced else None)

        while not self.stop_event.is_set():
            ret, frame_bgr = self.cap.read()
            if not ret:
                break
            
            current_pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if should_pace:
                if video_start_ms is None:
                    video_start_ms, wall_start_time = current_pos_ms, time.monotonic()
                
                if video_keep_period_ms:
                    if next_keep_video_ms is None:
                        next_keep_video_ms = current_pos_ms
                    if current_pos_ms + 1e-3 < next_keep_video_ms:
                        continue
                    while current_pos_ms + 1e-3 >= next_keep_video_ms:
                        next_keep_video_ms += video_keep_period_ms
                
                desired_wall_time = wall_start_time + (current_pos_ms - video_start_ms) / 1000.0
                current_wall_time = time.monotonic()
                if current_wall_time < desired_wall_time:
                    time.sleep(desired_wall_time - current_wall_time)
            
            if keep_period:
                current_time = time.monotonic()
                if current_time < next_keep_timestamp:
                    continue
                next_keep_timestamp += keep_period
            
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        self.cap.release()

    def preprocess(
        self,
        input_queue: queue.Queue,
        model_input_width: int,
        model_input_height: int,
        preprocess_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
    ) -> None:
        preprocess_fn = preprocess_fn or default_preprocess
        raw_frames, processed_frames = [], []
        for frame in self._generate_frames():
            raw_frames.append(frame)
            processed_frames.append(preprocess_fn(frame, model_input_width, model_input_height))
            if len(raw_frames) >= self.batch_size:
                input_queue.put((raw_frames, processed_frames))
                raw_frames, processed_frames = [], []
        if raw_frames:
            input_queue.put((raw_frames, processed_frames))
        input_queue.put(None)


class VideoVisualizer:
    def __init__(
        self,
        output_dir: str = "output",
        save_output: bool = False,
        source_fps: Optional[float] = None,
        frame_rate: Optional[float] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        self.output_dir = output_dir
        self.save_output = save_output
        self.source_fps = source_fps
        self.frame_rate = frame_rate
        self.stop_event = stop_event or threading.Event()

        self._count = 0
        self._start_time = None

        self.video_writer = None
        self.image_index = 0
        self.window_name = "Output"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def _init_writer(self, width: int, height: int):
        self.writer_frame_width = width
        self.writer_frame_height = height

        if self.save_output:
            output_fps = self.frame_rate or (self.source_fps if self.source_fps and self.source_fps > 1 else 30.0)
            os.makedirs(self.output_dir, exist_ok=True)
            output_video_path = os.path.join(self.output_dir, "output.avi")
            self.video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                output_fps,
                (self.writer_frame_width, self.writer_frame_height)
            )

    def show(self, frame: np.ndarray, fps: float, is_capture: bool = True) -> bool:
        if not hasattr(self, 'thickness'):
            self.thickness = calculate_optimal_thickness(frame.shape[:2])
            self.text_scale = calculate_optimal_text_scale(frame.shape[:2])

        render_status(frame, fps, thickness=self.thickness, text_scale=self.text_scale)
        render_resolution(frame, thickness=self.thickness, text_scale=self.text_scale)

        output_bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_to_show = output_bgr_frame

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, frame_to_show)
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return False

        if self.save_output:
            if is_capture:
                if self.video_writer is None:
                    height, width = frame.shape[:2]
                    self._init_writer(width, height)
                if self.video_writer is not None:
                    self.video_writer.write(frame_to_show)
            else:
                os.makedirs(self.output_dir, exist_ok=True)
                output_image_path = os.path.join(self.output_dir, f"output_{self.image_index}.png")
                cv2.imwrite(output_image_path, frame_to_show)
                self.image_index += 1
            
        return True

    def start(self):
        self._start_time = time.time()

    def increment(self, n: int = 1):
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        return f"Processed {self.count} frames at {self.fps:.2f} FPS, Total time: {self.elapsed:.2f} seconds"

    def visualize(
        self,
        output_queue: queue.Queue,
        callback: Callable[[Any, Any], None],
        is_capture: bool = True,
    ) -> None:
        self.start()
        with self:
            while True:
                result = output_queue.get()
                try:
                    if result is None:
                        break
                    original_frame, inference_result, *metadata = result
                    if self.stop_event.is_set():
                        continue
                    if isinstance(inference_result, list) and len(inference_result) == 1:
                        inference_result = inference_result[0]
                    
                    frame_with_detections = callback(original_frame, inference_result, *metadata)
                    self.increment()

                    if not self.show(frame_with_detections, self.fps, is_capture=is_capture):
                        self.stop_event.set()

                finally:
                    output_queue.task_done()

        self.stop_event.set()


if HAILO_AVAILABLE:
    class HailoInfer:
        def __init__(
            self, hef_path: str, batch_size: int = 1,
                input_type: Optional[str] = None, output_type: Optional[str] = None,
                priority: Optional[int] = 0) -> None:

            """
            Initialize the HailoAsyncInference class to perform asynchronous inference using a Hailo HEF model.

            Args:
                hef_path (str): Path to the HEF model file.
                batch_size (optional[int]): Number of inputs processed per inference. Defaults to 1.
                input_type (Optional[str], optional): Input data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
                output_type (Optional[str], optional): Output data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
                priority (optional[int]): Scheduler priority value for the model within the shared VDevice context. Defaults to 0.
            """
            params = VDevice.create_params()
            # Set the scheduling algorithm to round-robin to activate the scheduler
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            vDevice = VDevice(params)

            self.target = vDevice
            hef_path = os.fspath(hef_path)
            self.hef = HEF(hef_path)

            self.infer_model = self.target.create_infer_model(hef_path)
            self.infer_model.set_batch_size(batch_size)

            self._set_input_type(input_type)
            self._set_output_type(output_type)

            self.config_ctx = self.infer_model.configure()
            self.configured_model = self.config_ctx.__enter__()
            self.configured_model.set_scheduler_priority(priority)
            self.last_infer_job = None

        def _set_input_type(self, input_type: Optional[str] = None) -> None:
            """
            Set the input type for the HEF model. If the model has multiple inputs,
            it will set the same type of all of them.

            Args:
                input_type (Optional[str]): Format type of the input stream.
            """

            if input_type is not None:
                self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        def _set_output_type(self, output_type: Optional[str] = None) -> None:
            """
            Set the output type for each model output.

            Args:
                output_type (Optional[str]): Desired output data type. Common values:
                    'UINT8', 'UINT16', 'FLOAT32'.
            """

            self.nms_postprocess_enabled = False

            # If the model uses HAILO_NMS_WITH_BYTE_MASK format (e.g.,instance segmentation),
            if self.infer_model.outputs[0].format.order == FormatOrder.HAILO_NMS_WITH_BYTE_MASK:
                # Use UINT8 and skip setting output formats
                self.nms_postprocess_enabled = True
                self.output_type = self._output_data_type2dict("UINT8")
                return

            # Otherwise, set the format type based on the provided output_type argument
            self.output_type = self._output_data_type2dict(output_type)

            # Apply format to each output layer
            for name, dtype in self.output_type.items():
                self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))

        def get_vstream_info(self) -> Tuple[list, list]:
            """
            Get information about input and output stream layers.

            Returns:
                Tuple[list, list]: List of input stream layer information, List of 
                                   output stream layer information.
            """
            return (
                self.hef.get_input_vstream_infos(), 
                self.hef.get_output_vstream_infos()
            )

        def get_hef(self) -> HEF:
            """
            Get a HEF instance
            
            Returns:
                HEF: A HEF (Hailo Executable File) containing the model.
            """
            return self.hef

        def get_input_shape(self) -> Tuple[int, ...]:
            """
            Get the shape of the model's input layer.

            Returns:
                Tuple[int, ...]: Shape of the model's input layer.
            """
            return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

        def run(self, input_batch: List[np.ndarray], inference_callback_fn) -> object:
            """
            Run an asynchronous inference job on a batch of preprocessed inputs.

            This method reuses a preconfigured model (no reconfiguration overhead),
            prepares input/output bindings, launches async inference, and returns
            the job handle so that the caller can wait on it if needed.

            Args:
                input_batch (List[np.ndarray]): A batch of preprocessed model inputs.
                inference_callback_fn (Callable): Function to be invoked when inference is complete.
                                                  It receives `bindings_list` and additional context.

            Returns:
                Async job handle returned by `run_async`, which can be used to wait for completion or check status.
            """
            bindings_list = self._create_bindings(self.configured_model, input_batch)
            self.configured_model.wait_for_async_ready(timeout_ms=10000)

            # Launch async inference and attach the result handler
            self.last_infer_job = self.configured_model.run_async(
                bindings_list,
                partial(inference_callback_fn, bindings_list=bindings_list)
            )
            return self.last_infer_job

        def _create_bindings(self, configured_model, input_batch):
            """
            Create a list of input-output bindings for a batch of frames.

            Args:
                configured_model: The configured inference model.
                input_batch (List[np.ndarray]): List of input frames, preprocessed and ready.

            Returns:
                List[Bindings]: A list of bindings for each frame's input and output buffers.
            """

            def _frame_binding(frame: np.ndarray):
                output_buffers = {
                    name: np.empty(
                        self.infer_model.output(name).shape,
                        dtype=(getattr(np, self.output_type[name].lower()))
                    )
                    for name in self.output_type
                }

                binding = configured_model.create_bindings(output_buffers=output_buffers)
                binding.input().set_buffer(np.array(frame))
                return binding

            return [_frame_binding(frame) for frame in input_batch]

        def is_nms_postprocess_enabled(self) -> bool:
            """
            Returns True if the HEF model includes an NMS postprocess node.
            """
            return self.nms_postprocess_enabled

        def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
            """
            Generate a dictionary mapping each output layer name to its corresponding
            data type. If no data type is provided, use the type defined in the HEF.

            Args:
                data_type (Optional[str]): The desired data type for all output layers.
                                           Valid values: 'float32', 'uint8', 'uint16'.
                                           If None, uses types from the HEF metadata.

            Returns:
                Dict[str, str]: A dictionary mapping output layer names to data types.
            """
            valid_types = {"float32", "uint8", "uint16"}
            data_type_dict = {}

            for output_info in self.hef.get_output_vstream_infos():
                name = output_info.name
                if data_type is None:
                    # Extract type from HEF metadata
                    hef_type = str(output_info.format.type).split(".")[-1]
                    data_type_dict[name] = hef_type
                else:
                    if data_type.lower() not in valid_types:
                        raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                    data_type_dict[name] = data_type

            return data_type_dict

        def close(self):
            # Wait for the final job to complete before exiting
            if self.last_infer_job is not None:
                self.last_infer_job.wait(10000)

            if self.config_ctx:
                self.config_ctx.__exit__(None, None, None)

    class HailoAsyncInference(HailoInfer):
        def infer(self, input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
            """
            Main inference loop that pulls data from the input queue, runs asynchronous
            inference, and pushes results to the output queue.

            Each item in the input queue is expected to be a tuple:
                (input_batch, preprocessed_batch)
                - input_batch: Original frames (used for visualization or tracking)
                - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

            Args:
                input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
                output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.
                stop_event (threading.Event): Event to signal stopping the inference loop.
            """
            pending_jobs = collections.deque()

            while True:
                next_batch = input_queue.get()
                if not next_batch:
                    break

                if stop_event.is_set():
                    continue

                input_batch, preprocessed_batch = next_batch

                inference_callback_fn = partial(
                    self._inference_callback,
                    input_batch=input_batch,
                    output_queue=output_queue
                )

                while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                    pending_jobs.popleft().wait(10000)

                job = self.run(preprocessed_batch, inference_callback_fn)
                pending_jobs.append(job)

            self.close()
            output_queue.put(None)

        def _inference_callback(self, completion_info, bindings_list: list, input_batch: list, output_queue: queue.Queue) -> None:
            if completion_info.exception:
                logger.error(f'Inference error: {completion_info.exception}')
            else:
                for i, bindings in enumerate(bindings_list):
                    if len(bindings._output_names) == 1:
                        result = bindings.output().get_buffer()
                    else:
                        result = {
                            name: np.expand_dims(
                                bindings.output(name).get_buffer(), axis=0
                            )
                            for name in bindings._output_names
                        }
                    output_queue.put((input_batch[i], result))
else:
    HailoInfer = None
    HailoAsyncInference = None
