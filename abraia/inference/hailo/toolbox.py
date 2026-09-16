import os
import cv2
import shlex
import logging
import subprocess
import collections
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional, Tuple

from ...utils import download_url, get_remote_file_size
from ...tasks import normalize_task, to_hailo_task
from ..ops import sigmoid, softmax, nms
            

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
    from hailo_platform.pyhailort.pyhailort import FormatOrder
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry shared by preprocessing and Hailo output postprocessing."""

    original_height: int
    original_width: int
    model_height: int
    model_width: int
    scale: float
    resized_height: int
    resized_width: int
    pad_y: int
    pad_x: int

    @classmethod
    def from_dimensions(cls, original_dim, model_dim):
        original_height, original_width = original_dim
        model_height, model_width = model_dim
        scale = min(model_width / original_width, model_height / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)
        return cls(
            original_height, original_width, model_height, model_width,
            scale, resized_height, resized_width,
            (model_height - resized_height) // 2,
            (model_width - resized_width) // 2,
        )

    def box_to_original(self, box):
        xmin, ymin, xmax, ymax = box
        return [
            max(0, min(self.original_width, int((xmin - self.pad_x) / self.scale))),
            max(0, min(self.original_height, int((ymin - self.pad_y) / self.scale))),
            max(0, min(self.original_width, int((xmax - self.pad_x) / self.scale))),
            max(0, min(self.original_height, int((ymax - self.pad_y) / self.scale))),
        ]

    def keypoints_to_original(self, keypoints):
        mapped = np.array(keypoints, copy=True)
        mapped[:, 0] = np.clip(
            (mapped[:, 0] - self.pad_x) / self.scale,
            0, self.original_width - 1,
        )
        mapped[:, 1] = np.clip(
            (mapped[:, 1] - self.pad_y) / self.scale,
            0, self.original_height - 1,
        )
        return mapped

    def mask_to_original(self, mask):
        unpadded = mask[
            self.pad_y:self.pad_y + self.resized_height,
            self.pad_x:self.pad_x + self.resized_width,
        ]
        return cv2.resize(
            unpadded,
            (self.original_width, self.original_height),
            interpolation=cv2.INTER_LINEAR,
        )


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    img_h, img_w, _ = image.shape[:3]
    transform = LetterboxTransform.from_dimensions(
        (img_h, img_w), (model_h, model_w)
    )
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    padded_image[
        transform.pad_y:transform.pad_y + transform.resized_height,
        transform.pad_x:transform.pad_x + transform.resized_width,
    ] = resized
    return padded_image


# Hardware and Architecture
HAILO8_ARCH, HAILO8L_ARCH, HAILO10H_ARCH = "hailo8", "hailo8l", "hailo10h"
HAILO_ARCHS = {
    "HAILO8L": HAILO8L_ARCH,
    "HAILO8": HAILO8_ARCH,
    "HAILO10H": HAILO10H_ARCH,
    "HAILO15H": HAILO10H_ARCH
}
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"


def detect_hailo_arch() -> Optional[str]:
    """Detect the connected Hailo device architecture."""
    try:
        res = subprocess.run(
            shlex.split(HAILO_FW_CONTROL_CMD),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode == 0:
            stdout = res.stdout.upper()
            for key, arch in HAILO_ARCHS.items():
                if key in stdout:
                    return arch
    except Exception as e:
        logger.error(f"Error detecting Hailo architecture: {e}")
    return None

# Base Defaults
HAILO_FILE_EXTENSION = ".hef"
HAILO_MODEL_ZOO_DEFAULT_VERSION = "v2.17.0"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
RESOURCES_MODELS_DIR_NAME = "models"

# Async inference defaults
MAX_ASYNC_INFER_JOBS = 20

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
                    {"name": "yolov11x", "source": "mz"}
                ]
            },
            "hailo8l": {
                "default": [{"name": "yolov8s", "source": "mz"}],
                "extra": [
                    {"name": "yolov8n", "source": "mz"}, {"name": "yolov8m", "source": "mz"},
                    {"name": "yolov8l", "source": "mz"}, {"name": "yolov8x", "source": "mz"},
                    {"name": "yolov11n", "source": "mz"}, {"name": "yolov11s", "source": "mz"},
                    {"name": "yolov11m", "source": "mz"}, {"name": "yolov11l", "source": "mz"},
                    {"name": "yolov11x", "source": "mz"}
                ]
            }
        }
    },
    "segment": {
        "models": {
            "hailo8": {
                "default": [{"name": "yolov5m_seg_with_nms", "source": "s3"}],
                "extra": [
                    {"name": "yolov5m_seg", "source": "mz"}, {"name": "yolov5l_seg", "source": "mz"},
                    {"name": "yolov5n_seg", "source": "mz"}, {"name": "yolov5s_seg", "source": "mz"},
                    {"name": "yolov8n_seg", "source": "mz"}, {"name": "yolov8m_seg", "source": "mz"},
                    {"name": "yolov8s_seg", "source": "mz"}
                ]
            },
            "hailo8l": {
                "default": [{"name": "yolov5n_seg", "source": "mz"}],
                "extra": [
                    {"name": "yolov5l_seg", "source": "mz"}, {"name": "yolov5m_seg", "source": "mz"},
                    {"name": "yolov5s_seg", "source": "mz"}, {"name": "yolov8m_seg", "source": "mz"},
                    {"name": "yolov8n_seg", "source": "mz"}, {"name": "yolov8s_seg", "source": "mz"}
                ]
            }
        }
    },
    "pose": {
        "models": {
            "hailo8": {
                "default": [{"name": "yolov8m_pose", "source": "mz"}],
                "extra": [{"name": "yolov8s_pose", "source": "mz"}]
            },
            "hailo8l": {
                "default": [{"name": "yolov8s_pose", "source": "mz"}]
            }
        }
    }
}

SEGMENT_CONFIG = {
    "v5": {
        "arch": "yolov5_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "sizes": [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        },
        "input_shape": [640, 640], "mask_channels": 32, "score_threshold": 0.001, "nms_iou_thresh": 0.6, "classes": 80,
        "layers": [[1, 160, 160, "mask_channels"], [1, 20, 20, "detection_channels"], [1, 40, 40, "detection_channels"], [1, 80, 80, "detection_channels"]]
    },
    "v8": {
        "arch": "yolov8_seg",
        "anchors": {"strides": [8, 16, 32], "regression_length": 15},
        "input_shape": [640, 640], "mask_channels": 32, "score_threshold": 0.001, "nms_iou_thresh": 0.7, "meta_arch": "yolov8_seg_postprocess", "classes": 80,
        "layers": [[1, 20, 20, "detection_output_channels"], [1, 20, 20, "classes"], [1, 20, 20, "mask_channels"], [1, 40, 40, "detection_output_channels"], [1, 40, 40, "classes"], [1, 40, 40, "mask_channels"], [1, 80, 80, "detection_output_channels"], [1, 80, 80, "classes"], [1, 80, 80, "mask_channels"], [1, 160, 160, "mask_channels"]]
    }
}


def get_model_url(task, model_name, hailo_arch):
    """Return a download task tuple for a specific app model, or None if not found."""
    app_cfg = RESOURCES_CONFIG.get(task, {}).get("models", {}).get(hailo_arch, {})
    for entry in app_cfg.get("default", []) + app_cfg.get("extra", []):
        name = entry.get("name")
        if name == model_name:
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
    logger.warning(f"Model '{model_name}' not found for task '{task}'")
    return None, None

def execute_download(url, dest_path):
    """Execute a single download task."""
    remote_size = get_remote_file_size(url)
    if dest_path.exists() and remote_size and dest_path.stat().st_size == remote_size:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading: {url}")
        download_url(url, str(dest_path))
    except Exception as e:
        if dest_path.exists(): dest_path.unlink()
        logger.warning(f"Failed to download {url}: {e}")

def get_resource_path(resource_type: str, name: str, arch: Optional[str] = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch: raise RuntimeError("Could not detect Hailo architecture.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)
    return root / resource_type / name

def get_default_model(task: str, arch: str) -> Optional[str]:
    default_entries = RESOURCES_CONFIG.get(task, {}).get("models", {}).get(arch, {}).get("default", [])
    for entry in default_entries:
        name = entry.get("name")
        if isinstance(name, str) and name.lower() != "none":
            return name
    return None

def resolve_hef_path(hef_path: Optional[str], task: str, arch: Optional[str] = None) -> Optional[Path]:
    """Resolve HEF path, downloading it if necessary."""
    arch = arch or detect_hailo_arch()
    if not arch: raise RuntimeError("Could not detect Hailo architecture.")
    
    if hef_path is None:
        hef_path = get_default_model(task, arch)
        if not hef_path:
            logger.error(f"No default model found for {task}/{arch}")
            return None
        logger.info(f"Using default model: {hef_path}")

    path = Path(hef_path)
    if path.exists(): return path.resolve()
    if not path.suffix and path.with_suffix(HAILO_FILE_EXTENSION).exists():
        return path.with_suffix(HAILO_FILE_EXTENSION).resolve()

    model_name = path.stem
    resource_path = get_resource_path(RESOURCES_MODELS_DIR_NAME, model_name, arch)
    if resource_path.exists(): return resource_path

    logger.warning(f"Model '{model_name}' not found. Downloading...")
    url, dest = get_model_url(task, model_name, arch)
    if url and dest:
        execute_download(url, dest)
        if dest.exists(): return dest
    
    logger.error(f"Model '{model_name}' not found.")
    return None


def get_labels(labels_path: str) -> list:
    if labels_path is None or not os.path.exists(labels_path):
        return COCO_LABELS
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


def resize_mask_to_unpadded_box(mask_1d, box_on_input_image, box_on_padded_image):
    """
    Resize the mask from the padded box to match the unpadded box size.

    Args:
        mask_1d (np.ndarray): 1D binary mask.
        padded_box (list): [ymin, xmin, ymax, xmax] in 640x640 padded image.
        unpadded_box (list): [ymin, xmin, ymax, xmax] after unpadding.

    Returns:
        np.ndarray: Resized 2D mask for the unpadded box size.
    """
    try:
        x1_p, y1_p, x2_p, y2_p = box_on_padded_image
        w_p, h_p = x2_p - x1_p, y2_p - y1_p
        
        mask_1d = np.asarray(mask_1d)
        if h_p <= 0 or w_p <= 0 or mask_1d.size != h_p * w_p:
            logger.warning(
                "Ignoring Hailo mask with %d values; expected %d (%dx%d)",
                mask_1d.size, h_p * w_p, h_p, w_p,
            )
            return None
        mask_2d = mask_1d.reshape((h_p, w_p))

        x1_u, y1_u, x2_u, y2_u = box_on_input_image
        resized_mask = cv2.resize(mask_2d.astype(np.uint8), (x2_u - x1_u, y2_u - y1_u), interpolation=cv2.INTER_NEAREST)

    except Exception:
        return None

    return resized_mask


def convert_nms_box_from_normalized(normalized_box, model_dim, original_dim):
    """Map a Hailo byte-mask box to original and model-space coordinates.

    Hailo's byte-mask payload is an ROI whose dimensions are measured in
    model-input pixels. The original frame box has a different scale when
    letterbox preprocessing is used, so the two coordinate systems must stay
    separate.
    """
    model_height, model_width = model_dim
    normalized_box = np.asarray(normalized_box, dtype=np.float32)
    normalized_box = np.clip(normalized_box, 0.0, 1.0)
    x_min, y_min, x_max, y_max = normalized_box
    model_box = [
        float(x_min * model_width),
        float(y_min * model_height),
        float(x_max * model_width),
        float(y_max * model_height),
    ]
    transform = LetterboxTransform.from_dimensions(original_dim, model_dim)
    original_box = transform.box_to_original(model_box)
    mask_box = [
        0,
        0,
        max(0, int(np.ceil((x_max - x_min) * model_width))),
        max(0, int(np.ceil((y_max - y_min) * model_height))),
    ]
    return original_box, mask_box


def convert_box_from_normalized(normalized_box: list,
                                 padded_image_size: int,
                                 padding: int,
                                 input_image_height: int,
                                 input_image_width: int) -> tuple:
    """
    Converts a normalized bounding box to:
    1. Coordinates in the original input image (after removing padding)
    2. Coordinates in the model's padded output image (e.g. 640x640)

    Args:
        normalized_box (list): Normalized [x_min, y_min, x_max, y_max] in range [0, 1].
        padded_image_size (int): Size of the square padded image (typically 640).
        padding (int): Amount of padding applied to center the image.
        input_image_height (int): Height of the original input image.
        input_image_width (int): Width of the original input image.

    Returns:
        tuple:
            box_on_input_image (list): Box mapped to original image resolution.
            box_on_padded_image (list): Box mapped to padded model output image.
    """

    box_on_padded_image = [
        min(max(round(norm_val * padded_image_size), 0), padded_image_size)
        for norm_val in normalized_box
    ]

    transform = LetterboxTransform(
        input_image_height,
        input_image_width,
        padded_image_size,
        padded_image_size,
        1.0,
        input_image_height,
        input_image_width,
        0 if padded_image_size == input_image_height else padding,
        0 if padded_image_size == input_image_width else padding,
    )
    box_on_input_image = transform.box_to_original(box_on_padded_image)

    return box_on_input_image, box_on_padded_image


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
            self.target = None
            self.hef = None
            self.infer_model = None
            self.config_ctx = None
            self.configured_model = None
            self.pending_jobs = collections.deque()
            self._closed = True
            self._config_entered = False
            try:
                self.target = VDevice(params)
                hef_path = os.fspath(hef_path)
                self.hef = HEF(hef_path)

                self.infer_model = self.target.create_infer_model(hef_path)
                self.infer_model.set_batch_size(batch_size)

                self._set_input_type(input_type)
                self._set_output_type(output_type)

                self.config_ctx = self.infer_model.configure()
                self.configured_model = self.config_ctx.__enter__()
                self._config_entered = True
                self.configured_model.set_scheduler_priority(priority)
            except Exception:
                self._release_resources()
                raise
            self.last_infer_job = None
            self._closed = False

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
            if self._closed:
                raise RuntimeError("Hailo inference has already been closed")

            while len(self.pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                self.pending_jobs.popleft().wait(10000)

            bindings_list = self._create_bindings(self.configured_model, input_batch)
            self.configured_model.wait_for_async_ready(timeout_ms=10000)

            # Launch async inference and attach the result handler
            self.last_infer_job = self.configured_model.run_async(
                bindings_list,
                partial(inference_callback_fn, bindings_list=bindings_list)
            )
            self.pending_jobs.append(self.last_infer_job)
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
            """Wait for all submitted jobs and release the configured model."""
            if self._closed:
                return
            self._closed = True
            try:
                self.wait()
            finally:
                self._release_resources()

        def _release_resources(self):
            """Release partially or fully initialized Hailo resources."""
            if self.config_ctx is not None and self._config_entered:
                try:
                    self.config_ctx.__exit__(None, None, None)
                except Exception:
                    logger.warning("Failed to close Hailo configuration", exc_info=True)
                finally:
                    self.config_ctx = None
                    self.configured_model = None
                    self._config_entered = False

            release = getattr(self.target, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    logger.warning("Failed to release Hailo device", exc_info=True)
            self.target = None

        def wait(self):
            """Wait for all submitted jobs without releasing the backend."""
            while self.pending_jobs:
                self.pending_jobs.popleft().wait(10000)


def segment_xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def segment_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=32, multi_label=True):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            output.append({
                "detection_boxes": np.zeros((0, 4), dtype=np.float32),
                "mask": np.zeros((0, nm), dtype=np.float32),
                "detection_classes": np.zeros((0,), dtype=np.float32),
                "detection_scores": np.zeros((0,), dtype=np.float32),
            })
            continue

        x[:, 5:] *= x[:, 4:5]
        boxes = segment_xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label &= nc > 1
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)
            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)

        # Invalid box distributions (for example, an all-infinite Hailo
        # output passed through softmax) decode to NaN/Inf coordinates.
        # Keep those candidates out of NMS and mask cropping; converting
        # them to integer pixel coordinates otherwise raises an exception.
        finite = np.isfinite(x[:, :5]).all(axis=1)
        if not finite.all():
            logger.debug("Discarding %d non-finite segmentation candidates", np.count_nonzero(~finite))
            x = x[finite]

        x = x[x[:, 4].argsort()[::-1]]
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        keep = nms(np.hstack([boxes.astype(np.float32), conf.astype(np.float32)]), iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        out = x[keep]
        output.append({
            "detection_boxes": out[:, :4],
            "mask": out[:, 6:],
            "detection_classes": out[:, 5],
            "detection_scores": out[:, 4]
        })

    return output


def segment_process_mask_optimized(protos, masks_in, bboxes, shape, upsample=True, downsample=False):
    mh, mw, c = protos.shape
    ih, iw = shape
    protos_flat = protos.reshape(-1, c).T
    masks = masks_in @ protos_flat
    masks = sigmoid(masks).reshape(-1, mh, mw)

    bboxes = bboxes.copy()
    if downsample:
        bboxes[:, [0, 2]] *= mw / iw
        bboxes[:, [1, 3]] *= mh / ih
        masks = segment_crop_mask_roi_vectorized(masks, bboxes)

    if upsample:
        resized = np.empty((masks.shape[0], ih, iw), dtype=np.float32)
        for i in range(masks.shape[0]):
            resized[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
        masks = resized

    if not downsample:
        masks = segment_crop_mask_roi_vectorized(masks, bboxes)

    return masks


def segment_crop_mask_roi_vectorized(masks, boxes):
    N, H, W = masks.shape
    output = np.zeros_like(masks)
    boxes = np.round(boxes).astype(int)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)
    for i in range(N):
        x1, y1, x2, y2 = boxes[i]
        output[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return output


def segment_make_grid(anchors, stride, bs=8, nx=20, ny=20):
    na = len(anchors) // 2
    y, x = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing="ij")
    grid = np.stack((xv, yv), 2)
    grid = np.stack([grid for _ in range(na)], 0) - 0.5
    grid = np.stack([grid for _ in range(bs)], 0)
    anchor_grid = np.reshape(anchors * stride, (na, -1))
    anchor_grid = np.stack([anchor_grid for _ in range(ny)], axis=1)
    anchor_grid = np.stack([anchor_grid for _ in range(nx)], axis=2)
    anchor_grid = np.stack([anchor_grid for _ in range(bs)], 0)
    return grid, anchor_grid


def segment_yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes):
    BS, H, W = output.shape[0:3]
    stride = stride_list[branch_idx]
    anchors = anchor_list[branch_idx] / stride
    num_anchors = len(anchors) // 2
    grid, anchor_grid = segment_make_grid(anchors, stride, BS, W, H)
    output = output.transpose((0, 3, 1, 2)).reshape((BS, num_anchors, -1, H, W)).transpose((0, 1, 3, 4, 2))
    xy, wh, conf, mask = np.array_split(output, [2, 4, 4 + num_classes + 1], axis=4)
    xy = (sigmoid(xy) * 2 + grid) * stride
    wh = (sigmoid(wh) * 2) ** 2 * anchor_grid
    out = np.concatenate((xy, wh, sigmoid(conf), mask), 4)
    return out.reshape((BS, num_anchors * H * W, -1)).astype(np.float32)


def normalize_yolov8_scores(scores):
    """Return YOLOv8 class scores as probabilities.

    Hailo model-zoo HEFs normally apply sigmoid in the model script, but
    host-postprocessed/custom HEFs can expose the class logits instead.
    Support both forms without applying sigmoid twice to probabilities.
    """
    if not np.issubdtype(scores.dtype, np.floating):
        return scores
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size and (finite_scores.min() < 0 or finite_scores.max() > 1):
        return sigmoid(scores)
    return scores


def segment_yolov8_decoding(raw_boxes, strides, image_dims, reg_max):
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        shape = [int(x / stride) for x in image_dims]
        grid_x, grid_y = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
        ct_row, ct_col = grid_y.flatten() * stride, grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1))
        finite_distribution = np.isfinite(box_distribute).all(axis=(2, 3))
        safe_distribution = np.where(
            finite_distribution[..., None, None], box_distribute, 0
        )
        box_distance = softmax(safe_distribution)
        # Retain the invalid-anchor marker so NMS can discard it. This
        # avoids passing NaN coordinates to integer pixel conversion.
        box_distance[~finite_distribution] = np.nan
        box_distance = np.sum(box_distance * np.reshape(reg_range, (1, 1, 1, -1)), axis=-1) * stride
        box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        decode_box = np.expand_dims(center, axis=0) + box_distance
        xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes


def segment_yolov5_postprocess(endnodes, **kwargs):
    img_dims = tuple(kwargs["input_shape"])
    protos = endnodes[0]
    anchor_list = np.array(kwargs["anchors"]["sizes"][::-1])
    stride_list = kwargs["anchors"]["strides"][::-1]
    num_classes = kwargs["classes"]
    outputs = []
    for branch_idx, output in enumerate(endnodes[1:]):
        outputs.append(segment_yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes))
    outputs = np.concatenate(outputs, 1)
    outputs = segment_non_max_suppression(outputs, kwargs["score_threshold"], kwargs["nms_iou_thresh"], nm=protos.shape[-1])
    for batch_idx, output in enumerate(outputs):
        output["mask"] = segment_process_mask_optimized(protos[batch_idx].astype(np.float32, copy=False), output["mask"].astype(np.float32, copy=False), output["detection_boxes"], img_dims, upsample=True)
        output["detection_boxes"][:, [0, 2]] /= img_dims[1]
        output["detection_boxes"][:, [1, 3]] /= img_dims[0]
    return outputs


def segment_yolov8_postprocess(endnodes, **kwargs):
    num_classes, strides, image_dims, reg_max = kwargs["classes"], kwargs["anchors"]["strides"][::-1], tuple(kwargs["input_shape"]), kwargs["anchors"]["regression_length"]
    raw_boxes = endnodes[:7:3]
    scores = np.concatenate([np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes)) for s in endnodes[1:8:3]], axis=1)
    scores = normalize_yolov8_scores(scores)
    decoded_boxes = segment_yolov8_decoding(raw_boxes, strides, image_dims, reg_max)
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape
    scores_obj = np.concatenate([np.ones((scores.shape[0], scores.shape[1], 1)), scores], axis=-1)
    coeffs = np.concatenate([np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]], axis=1)
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    # One class per anchor keeps host-side NMS bounded to the number of
    # anchors. Multi-label expansion can turn 8,400 anchors into tens of
    # thousands of candidates and starve the asynchronous video callback.
    nms_res = segment_non_max_suppression(
        predictions,
        conf_thres=kwargs["score_threshold"],
        iou_thres=kwargs["nms_iou_thresh"],
        multi_label=False,
    )
    outputs = []
    for b in range(batch_size):
        masks = segment_process_mask_optimized(proto_data[b].astype(np.float32, copy=False), nms_res[b]["mask"].astype(np.float32, copy=False), nms_res[b]["detection_boxes"], image_dims)
        outputs.append({
            "detection_boxes": np.array(nms_res[b]["detection_boxes"]) / np.tile(image_dims, 2),
            "mask": masks,
            "detection_scores": np.array(nms_res[b]["detection_scores"]),
            "detection_classes": np.array(nms_res[b]["detection_classes"]).astype(int)
        })
    return outputs

def decode_pose_results(raw_boxes: np.ndarray, raw_kpts: np.ndarray, strides: List[int], image_dims: Tuple[int, int], reg_max: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes = None
    decoded_kpts = None

    for box_distribute, kpts, stride in zip(raw_boxes, raw_kpts, strides):
        shape = [int(x / stride) for x in image_dims]
        grid_x, grid_y = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
        ct_row, ct_col = grid_y.flatten() * stride, grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        box_distribute = np.reshape(
            box_distribute,
            (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1),
        )
        finite_distribution = np.isfinite(box_distribute).all(axis=(2, 3))
        safe_distribution = np.where(
            finite_distribution[..., None, None], box_distribute, 0
        )
        box_distance = softmax(safe_distribution)
        # Preserve invalid anchors so the pose postprocessor can discard
        # them instead of turning a failed softmax into a real detection.
        box_distance[~finite_distribution] = np.nan
        box_distance = np.sum(box_distance * np.reshape(np.arange(reg_max + 1), (1, 1, 1, -1)), axis=-1) * stride

        decode_box = np.expand_dims(center, axis=0) + np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

        decoded_kpts_for_layer = np.array(kpts, copy=True)
        decoded_kpts_for_layer[..., :2] = (
            stride * (decoded_kpts_for_layer[..., :2] * 2 - 0.5)
            + center[None, :, None, :2]
        )
        kpts = decoded_kpts_for_layer
        decoded_kpts = kpts if decoded_kpts is None else np.concatenate([decoded_kpts, kpts], axis=1)

    return boxes, decoded_kpts

def map_box_to_orig(box: list, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> list:
    return LetterboxTransform.from_dimensions(orig_dim, model_dim).box_to_original(box)

def map_keypoints_to_orig(keypoints: np.ndarray, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> np.ndarray:
    return LetterboxTransform.from_dimensions(
        orig_dim, model_dim
    ).keypoints_to_original(keypoints)

def map_mask_to_orig(mask: np.ndarray, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> np.ndarray:
    """Map a mask from letterboxed model space to the original image."""
    return LetterboxTransform.from_dimensions(
        orig_dim, model_dim
    ).mask_to_original(mask)

def resolve_shape(layer, model_type, arch_cfg):
    b, h, w, c_tag = layer
    mask_channels = arch_cfg["mask_channels"]
    if isinstance(c_tag, str):
        if c_tag == "mask_channels": c = mask_channels
        elif c_tag == "detection_channels": c = (arch_cfg['classes'] + 4 + 1 + mask_channels) * len(arch_cfg['anchors']['strides'])
        elif c_tag == "detection_output_channels": c = (arch_cfg["classes"] + 4 + 1 + mask_channels) * len(arch_cfg['anchors']['strides']) if model_type == 'v5' else (arch_cfg['anchors']['regression_length'] + 1) * 4
        elif c_tag == "classes": c = arch_cfg["classes"]
        else: raise ValueError(f"Unsupported channel tag: {c_tag}")
    else: c = c_tag
    return (b, h, w, c)


def select_output_layers(outputs: Dict[str, np.ndarray], expected_shapes):
    """Select output tensors by shape and reject ambiguous model layouts."""
    layers_by_shape = {}
    duplicates = set()
    for name, output in outputs.items():
        shape = tuple(output.shape)
        if shape in layers_by_shape:
            duplicates.add(shape)
        layers_by_shape[shape] = name
    if duplicates:
        raise ValueError(
            "Hailo output shapes are ambiguous: "
            + ", ".join(map(str, sorted(duplicates, key=str)))
        )

    missing = [tuple(shape) for shape in expected_shapes if tuple(shape) not in layers_by_shape]
    if missing:
        available = ", ".join(map(str, layers_by_shape)) or "none"
        raise ValueError(
            f"Hailo model is missing output shapes {missing}; available: {available}"
        )
    return [outputs[layers_by_shape[tuple(shape)]] for shape in expected_shapes]


if HAILO_AVAILABLE:
    class ModelInference(HailoInfer):
        def __init__(self, hef_path: str, task: str, labels: list, batch_size: int = 1, score_threshold: float = 0.25, mask_threshold: float = 0.45, model_type: str = 'v8'):
            task = normalize_task(task)
            backend_task = to_hailo_task(task)
            hef_path = resolve_hef_path(hef_path, backend_task)
            if hef_path is None:
                raise FileNotFoundError(f"Unable to resolve Hailo model for task '{task}'")
            # The v5/v8/pose postprocessors consume logits and DFL
            # distributions. Request dequantized output buffers from HailoRT;
            # using a HEF's native UINT8/UINT16 stream type makes those values
            # invalid for host-side softmax, box decoding, and NMS. Hailo's
            # byte-mask NMS path is still handled by HailoInfer itself.
            super().__init__(hef_path, batch_size, output_type="FLOAT32")
            self.task = backend_task
            self.labels = labels
            self.score_threshold = score_threshold
            self.mask_threshold = mask_threshold
            self.model_type = model_type
            self._postprocessors = {
                "detect": self._process_detect_results,
                "segment": self._process_segment_results,
                "pose": self._process_pose_results,
            }

        def _label(self, class_id):
            class_id = int(class_id)
            if self.labels and 0 <= class_id < len(self.labels):
                return self.labels[class_id]
            return str(class_id)

        def _get_results(self, bindings):
            if len(bindings._output_names) == 1:
                return bindings.output().get_buffer()
            return {name: np.expand_dims(bindings.output(name).get_buffer(), axis=0) for name in bindings._output_names}

        def _process_nms_results(self, result, image):
            infer_results = result if isinstance(result, list) else [result]
            img_height, img_width = image.shape[:2]
            model_height, model_width, _ = self.get_input_shape()
            detections = []
            for det in infer_results:
                if det.score < self.score_threshold:
                    continue
                box_on_input_image, box_on_padded_image = convert_nms_box_from_normalized(
                    [det.x_min, det.y_min, det.x_max, det.y_max],
                    (model_height, model_width),
                    (img_height, img_width),
                )
                xmin, ymin, xmax, ymax = box_on_input_image
                if xmax <= xmin or ymax <= ymin:
                    continue
                detection = {'label': self._label(det.class_id),
                    'score': float(det.score), 'box': [xmin, ymin, xmax - xmin, ymax - ymin], 'class_id': det.class_id}
                if self.task == 'segment':
                    mask = resize_mask_to_unpadded_box(det.mask, box_on_input_image, box_on_padded_image)
                    if mask is not None:
                        detection['mask'] = mask
                detections.append(detection)
            return detections

        def _process_detect_results(self, result, image):
            infer_results = result if isinstance(result, list) else [result]
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            detections = []
            for class_id, detection in enumerate(infer_results):
                for det in detection:
                    bbox, score = det[:4], det[4]
                    if score >= self.score_threshold:
                        # Hailo's raw detection output is normalized as
                        # [ymin, xmin, ymax, xmax, score].
                        xmin, ymin, xmax, ymax = map_box_to_orig(
                            [bbox[1] * mw, bbox[0] * mh, bbox[3] * mw, bbox[2] * mh],
                            (oh, ow),
                            (mh, mw),
                        )
                        detections.append({'label': self._label(class_id), 'score': float(score), 'box': [xmin, ymin, xmax - xmin, ymax - ymin], 'class_id': class_id})
            return detections
        
        def _process_segment_results(self, raw_detections, image):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            arch_cfg = SEGMENT_CONFIG[self.model_type]
            expected_shapes = [
                resolve_shape(layer, self.model_type, arch_cfg)
                for layer in arch_cfg["layers"]
            ]
            endnodes = select_output_layers(raw_detections, expected_shapes)
            # Use the runtime threshold during host-side NMS. The model-zoo
            # default (0.001) is useful for evaluation but allows almost every
            # low-confidence class through multi-label NMS, which can make the
            # callback appear stalled before results reach the renderer.
            postprocess_cfg = {**arch_cfg, "score_threshold": self.score_threshold}
            if self.model_type == "v5": result = segment_yolov5_postprocess(endnodes, **postprocess_cfg)[0]
            elif self.model_type == "v8": result = segment_yolov8_postprocess(endnodes, **postprocess_cfg)[0]
            else: raise ValueError(f"Unsupported architecture key: {self.model_type}")
            boxes, masks, scores, classes = result['detection_boxes'], result['mask'], result['detection_scores'], result['detection_classes']
            detections = []
            for i in range(len(boxes)):
                if scores[i] > self.score_threshold:
                    x1, y1, x2, y2 = boxes[i]
                    xmin, ymin, xmax, ymax = map_box_to_orig(
                        [x1 * mw, y1 * mh, x2 * mw, y2 * mh],
                        (oh, ow),
                        (mh, mw),
                    )
                    mask = map_mask_to_orig(masks[i], (oh, ow), (mh, mw))
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    detections.append({
                        'label': self._label(classes[i]), 'score': float(scores[i]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'mask': (mask[ymin:ymax, xmin:xmax] > self.mask_threshold).astype(np.uint8), 'class_id': int(classes[i])
                    })
            return detections

        def _process_pose_results(self, result, image, iou_thres=0.7, max_det=300):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            raw_detections = result
            reg_len = 15
            detection_out_channels = (reg_len + 1) * 4
            expected_shapes = [
                (1, h, w, c)
                for h, w, c in (
                    (20, 20, detection_out_channels),
                    (20, 20, 1),
                    (20, 20, 51),
                    (40, 40, detection_out_channels),
                    (40, 40, 1),
                    (40, 40, 51),
                    (80, 80, detection_out_channels),
                    (80, 80, 1),
                    (80, 80, 51),
                )
            ]
            endnodes = select_output_layers(raw_detections, expected_shapes)
            batch_size = endnodes[0].shape[0]
            strides = [32, 16, 8]
            raw_boxes = endnodes[:7:3]
            scores = np.concatenate([np.reshape(s, (-1, s.shape[1] * s.shape[2], 1)) for s in endnodes[1:8:3]], axis=1)
            scores = normalize_yolov8_scores(scores)
            kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]]
            decoded_boxes, decoded_kpts = decode_pose_results(raw_boxes, kpts, strides, (mh, mw), reg_len)
            predictions = np.concatenate([decoded_boxes, scores, np.reshape(decoded_kpts, (batch_size, -1, 51))], axis=2)
            detections = []
            x = predictions[0]
            x = x[
                (x[:, 4] > self.score_threshold)
                & np.isfinite(x).all(axis=1)
            ]
            if x.shape[0] > 0:
                boxes = np.copy(x[:, :4])
                boxes[:, 0], boxes[:, 1] = x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2
                boxes[:, 2], boxes[:, 3] = x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2
                finite_boxes = np.isfinite(boxes).all(axis=1)
                if not finite_boxes.all():
                    logger.debug("Discarding %d non-finite pose boxes", np.count_nonzero(~finite_boxes))
                    boxes = boxes[finite_boxes]
                    x = x[finite_boxes]
                indices = nms(np.concatenate((boxes, x[:, 4:5]), axis=1), iou_thres)[:max_det]
                for idx in indices:
                    xmin, ymin, xmax, ymax = map_box_to_orig(boxes[idx], (oh, ow), (mh, mw))
                    mapped_kpts = map_keypoints_to_orig(x[idx, 5:].reshape(17, 3)[..., :2], (oh, ow), (mh, mw))
                    detections.append({
                        'label': self._label(0) if self.labels else 'person', 'score': float(x[idx, 4]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'keypoints': mapped_kpts, 'joint_scores': sigmoid(x[idx, 5:].reshape(17, 3)[..., 2]), 'class_id': 0
                    })
            return detections

        def _inference_callback(
            self,
            completion_info,
            bindings_list: list,
            input_batch: list,
            output_callback,
            image_batch=None,
            stop_event=None,
        ) -> None:
            image_batch = image_batch if image_batch is not None else input_batch

            def emit_result(index, result=None, error=None):
                if stop_event is not None and stop_event.is_set():
                    return False
                return output_callback(index, result, error) is not False

            if completion_info.exception:
                logger.error("Hailo inference failed: %s", completion_info.exception)
                for index in range(len(input_batch)):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if not emit_result(index, error=completion_info.exception):
                        break
                return

            for i, bindings in enumerate(bindings_list):
                try:
                    result = self._get_results(bindings)
                    image = image_batch[i]
                    if self.is_nms_postprocess_enabled():
                        processed_result = self._process_nms_results(result, image)
                    else:
                        processor = self._postprocessors.get(self.task)
                        processed_result = (
                            processor(result, image)
                            if processor is not None
                            else result
                        )
                except Exception as exc:
                    logger.exception("Failed to post-process Hailo %s result", self.task)
                    if not emit_result(i, error=exc):
                        break
                    continue
                if not emit_result(i, processed_result):
                    break

        def infer_batch(
            self, input_batch, output_callback, stop_event=None, image_batch=None
        ):
            """Submit one batch and emit decoded results by frame index."""
            if stop_event is not None and stop_event.is_set():
                return
            if image_batch is not None and len(image_batch) != len(input_batch):
                raise ValueError("image_batch must match input_batch length")
            inference_callback_fn = partial(
                self._inference_callback,
                input_batch=input_batch,
                image_batch=image_batch,
                output_callback=output_callback,
                stop_event=stop_event,
            )
            self.run(input_batch, inference_callback_fn)

else:
    HailoInfer = None
    ModelInference = None
