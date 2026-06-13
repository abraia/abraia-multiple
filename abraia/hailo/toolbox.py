import os
import cv2
import queue
import shlex
import logging
import threading
import subprocess
import collections
import numpy as np

from pathlib import Path
from functools import partial
from typing import Dict, List, Optional, Tuple

from ..utils import download_url, get_remote_file_size
from ..inference.ops import sigmoid, softmax

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
    from hailo_platform.pyhailort.pyhailort import FormatOrder
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset, y_offset = (model_w - new_img_w) // 2, (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image
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
        res = subprocess.run(shlex.split(HAILO_FW_CONTROL_CMD), capture_output=True, text=True)
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

# Queue and async inference defaults
MAX_INPUT_QUEUE_SIZE = 60
MAX_OUTPUT_QUEUE_SIZE = 60
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


def find_shape_closest_to_target(mask_size, target_height, target_width):
    """
    Find the (height, width) pair whose product equals ``mask_size`` and whose
    Manhattan distance to (target_height, target_width) is minimal.

    Manhattan distance used:
        |h − target_height| + |w − target_width|

    Args:
        mask_size (int): Total number of pixels in the flattened mask.
        target_height (int): Desired height.
        target_width (int): Desired width.

    Returns:
        tuple[int, int] | None: Best-matching (height, width), or None if none found.
    """
    best_shape = None
    min_diff = float("inf")

    for h in range(1, mask_size + 1):
        if mask_size % h:
            continue
        w = mask_size // h
        diff = abs(h - target_height) + abs(w - target_width)
        if diff < min_diff:
            min_diff = diff
            best_shape = (h, w)

    return best_shape


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
        
        try:
            mask_2d = mask_1d.reshape((h_p, w_p))
        except ValueError:
            closest_shape = find_shape_closest_to_target(mask_1d.size, h_p, w_p)
            if not closest_shape:
                return None
            h, w = closest_shape
            mask_2d = mask_1d.reshape((h, w))

        x1_u, y1_u, x2_u, y2_u = box_on_input_image
        resized_mask = cv2.resize(mask_2d.astype(np.uint8), (x2_u - x1_u, y2_u - y1_u), interpolation=cv2.INTER_NEAREST)

    except Exception:
        return None

    return resized_mask


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

    box_on_padded_image = []
    box_on_input_image = []

    for i, norm_val in enumerate(normalized_box):
        # 1. Scale to padded image space (e.g. 640)
        padded_coord = round(norm_val * padded_image_size)
        padded_coord = min(max(padded_coord, 0), padded_image_size)
        box_on_padded_image.append(round(norm_val * 640))

        # 2. Remove padding to get input image space
        if i % 2 == 0:  # x coordinate
            input_coord = padded_coord - padding if padded_image_size != input_image_width else padded_coord
            input_coord = min(max(input_coord, 0), input_image_width)
        else:  # y coordinate
            input_coord = padded_coord - padding if padded_image_size != input_image_height else padded_coord
            input_coord = min(max(input_coord, 0), input_image_height)

        box_on_input_image.append(input_coord)

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
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, 32)),
                    "detection_classes": np.zeros((0, 80)),
                    "detection_scores": np.zeros((0, 80)),
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

            x = x[x[:, 4].argsort()[::-1]]
            cls_shift = x[:, 5:6] * max_wh
            boxes = x[:, :4] + cls_shift
            conf = x[:, 4:5]
            from ..inference.ops import nms as ops_nms
            keep = ops_nms(np.hstack([boxes.astype(np.float32), conf.astype(np.float32)]), iou_thres)

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


    def segment_yolov8_decoding(raw_boxes, strides, image_dims, reg_max):
        boxes = None
        for box_distribute, stride in zip(raw_boxes, strides):
            shape = [int(x / stride) for x in image_dims]
            grid_x, grid_y = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
            ct_row, ct_col = grid_y.flatten() * stride, grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1))
            box_distance = np.sum(softmax(box_distribute) * np.reshape(reg_range, (1, 1, 1, -1)), axis=-1) * stride
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
        decoded_boxes = segment_yolov8_decoding(raw_boxes, strides, image_dims, reg_max)
        proto_data = endnodes[9]
        batch_size, _, _, n_masks = proto_data.shape
        scores_obj = np.concatenate([np.ones((scores.shape[0], scores.shape[1], 1)), scores], axis=-1)
        coeffs = np.concatenate([np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]], axis=1)
        predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
        nms_res = segment_non_max_suppression(predictions, conf_thres=kwargs["score_threshold"], iou_thresh=kwargs["nms_iou_thresh"], multi_label=True)
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


    def pose_nms(dets: np.ndarray, thresh: float) -> np.ndarray:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        suppressed = np.zeros(dets.shape[0], dtype=int)
        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i] == 1:
                continue
            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j] == 1:
                    continue

                xx1 = max(x1[idx_i], x1[idx_j])
                yy1 = max(y1[idx_i], y1[idx_j])
                xx2 = min(x2[idx_i], x2[idx_j])
                yy2 = min(y2[idx_i], y2[idx_j])
                w = max(0.0, xx2 - xx1 + 1)
                h = max(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[idx_i] + areas[idx_j] - inter)

                if ovr >= thresh:
                    suppressed[idx_j] = 1

        return np.where(suppressed == 0)[0]


    def decode_pose_results(raw_boxes: np.ndarray, raw_kpts: np.ndarray, strides: List[int], image_dims: Tuple[int, int], reg_max: int) -> Tuple[np.ndarray, np.ndarray]:
        boxes = None
        decoded_kpts = None

        for box_distribute, kpts, stride in zip(raw_boxes, raw_kpts, strides):
            shape = [int(x / stride) for x in image_dims]
            grid_x, grid_y = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
            ct_row, ct_col = grid_y.flatten() * stride, grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            box_distance = softmax(np.reshape(box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)))
            box_distance = np.sum(box_distance * np.reshape(np.arange(reg_max + 1), (1, 1, 1, -1)), axis=-1) * stride

            decode_box = np.expand_dims(center, axis=0) + np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
            xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
            xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
            boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

            kpts = stride * (kpts * 2 - 0.5) + np.expand_dims(center[..., :2], axis=1)
            decoded_kpts = kpts if decoded_kpts is None else np.concatenate([decoded_kpts, kpts], axis=1)

        return boxes, decoded_kpts


    def map_box_to_orig(box: list, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> list:
        oh, ow = orig_dim
        mh, mw = model_dim
        scale = min(mw / ow, mh / oh)
        pad_w, pad_h = (mw - int(ow * scale)) // 2, (mh - int(oh * scale)) // 2
        xmin, ymin, xmax, ymax = box
        xmin = max(0, min(ow - 1, int((xmin - pad_w) / scale)))
        ymin = max(0, min(oh - 1, int((ymin - pad_h) / scale)))
        xmax = max(0, min(ow - 1, int((xmax - pad_w) / scale)))
        ymax = max(0, min(oh - 1, int((ymax - pad_h) / scale)))
        return [xmin, ymin, xmax, ymax]

    def map_keypoints_to_orig(keypoints: np.ndarray, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> np.ndarray:
        oh, ow = orig_dim
        mh, mw = model_dim
        scale = min(mw / ow, mh / oh)
        pad_w, pad_h = (mw - int(ow * scale)) // 2, (mh - int(oh * scale)) // 2
        keypoints[:, 0] = np.clip((keypoints[:, 0] - pad_w) / scale, 0, ow - 1)
        keypoints[:, 1] = np.clip((keypoints[:, 1] - pad_h) / scale, 0, oh - 1)
        return keypoints

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

    class ModelInference(HailoInfer):
        def __init__(self, hef_path: str, task: str, labels: list, batch_size: int = 1, score_threshold: float = 0.25, mask_threshold: float = 0.45, model_type: str = 'v8'):
            hef_path = resolve_hef_path(hef_path, task)
            super().__init__(hef_path, batch_size)
            self.task = task
            self.labels = labels
            self.score_threshold = score_threshold
            self.mask_threshold = mask_threshold
            self.model_type = model_type

        def _get_results(self, bindings):
            if len(bindings._output_names) == 1:
                return bindings.output().get_buffer()
            return {name: np.expand_dims(bindings.output(name).get_buffer(), axis=0) for name in bindings._output_names}

        def _process_nms_results(self, result, image):
            infer_results = result if isinstance(result, list) else [result]
            img_height, img_width = image.shape[:2]
            size = max(img_height, img_width)
            padding_length = int(abs(img_height - img_width) / 2)
            detections = []
            for det in infer_results:
                if det.score < self.score_threshold: break
                box_on_input_image, box_on_padded_image = convert_box_from_normalized(
                    [det.x_min, det.y_min, det.x_max, det.y_max], size, padding_length, img_height, img_width)
                xmin, ymin, xmax, ymax = box_on_input_image
                detection = {'label': self.labels[det.class_id] if self.labels else str(det.class_id),
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
                        # Denormalize and map to original coords
                        xmin, ymin, xmax, ymax = map_box_to_orig([bbox[1] * mw, bbox[0] * mh, bbox[3] * mw, bbox[2] * mh], (oh, ow), (mh, mw))
                        detections.append({'label': self.labels[class_id] if self.labels else str(class_id), 'score': float(score), 'box': [xmin, ymin, xmax - xmin, ymax - ymin], 'class_id': class_id})
            return detections
        
        def _process_segment_results(self, raw_detections, image):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            arch_cfg = SEGMENT_CONFIG[self.model_type]
            raw_detections_keys = list(raw_detections.keys())
            layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}
            endnodes = [raw_detections[layer_from_shape[resolve_shape(layer, self.model_type, arch_cfg)]] for layer in arch_cfg["layers"]]
            if self.model_type == "v5": result = segment_yolov5_postprocess(endnodes, **arch_cfg)[0]
            elif self.model_type == "v8": result = segment_yolov8_postprocess(endnodes, **arch_cfg)[0]
            else: raise ValueError(f"Unsupported architecture key: {self.model_type}")
            boxes, masks, scores, classes = result['detection_boxes'], result['mask'], result['detection_scores'], result['detection_classes']
            detections = []
            for i in range(len(boxes)):
                if scores[i] < self.score_threshold: continue
                cx, cy, w, h = boxes[i]
                xmin, ymin, xmax, ymax = map_box_to_orig([(cx - w / 2) * mw, (cy - h / 2) * mh, (cx + w / 2) * mw, (cy + h / 2) * mh], (oh, ow), (mh, mw))
                detections.append({
                    'label': self.labels[classes[i]] if self.labels else str(classes[i]), 'score': float(scores[i]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'mask': (masks[i, ymin:ymax, xmin:xmax] > self.mask_threshold).astype(np.uint8), 'class_id': int(classes[i])
                })
            return detections

        def _process_pose_results(self, result, image):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            raw_detections = result
            raw_detections_keys = list(raw_detections.keys())
            layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}
            reg_len = 15
            detection_out_channels = (reg_len + 1) * 4
            endnodes = [raw_detections[layer_from_shape[1, h, w, c]] for h, w, c in [(20, 20, detection_out_channels), (20, 20, 1), (20, 20, 51), (40, 40, detection_out_channels), (40, 40, 1), (40, 40, 51), (80, 80, detection_out_channels), (80, 80, 1), (80, 80, 51)]]
            batch_size = endnodes[0].shape[0]
            strides = [32, 16, 8]
            raw_boxes = endnodes[:7:3]
            scores = np.concatenate([np.reshape(s, (-1, s.shape[1] * s.shape[2], 1)) for s in endnodes[1:8:3]], axis=1)
            kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]]
            decoded_boxes, decoded_kpts = decode_pose_results(raw_boxes, kpts, strides, (mh, mw), reg_len)
            predictions = np.concatenate([decoded_boxes, scores, np.reshape(decoded_kpts, (batch_size, -1, 51))], axis=2)
            detections = []
            x = predictions[0][predictions[0, :, 4] > self.score_threshold]
            if x.shape[0] > 0:
                boxes = np.copy(x[:, :4])
                boxes[:, 0], boxes[:, 1] = x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2
                boxes[:, 2], boxes[:, 3] = x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2
                indices = pose_nms(np.concatenate((boxes, x[:, 4:5]), axis=1), 0.7)[:300]
                for idx in indices:
                    xmin, ymin, xmax, ymax = map_box_to_orig(boxes[idx], (oh, ow), (mh, mw))
                    mapped_kpts = map_keypoints_to_orig(x[idx, 5:].reshape(17, 3)[..., :2], (oh, ow), (mh, mw))
                    detections.append({
                        'label': self.labels[0] if self.labels else 'person', 'score': float(x[idx, 4]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'keypoints': mapped_kpts, 'joint_scores': sigmoid(x[idx, 5:].reshape(17, 3)[..., 2]), 'class_id': 0
                    })
            return detections

        def _inference_callback(self, completion_info, bindings_list: list, input_batch: list, output_queue: queue.Queue) -> None:
            if completion_info.exception:
                logger.error(f'Inference error: {completion_info.exception}')
            else:
                for i, bindings in enumerate(bindings_list):
                    result = self._get_results(bindings)
                    if self.is_nms_postprocess_enabled(): processed_result = self._process_nms_results(result, input_batch[i])
                    elif self.task == 'detect': processed_result = self._process_detect_results(result, input_batch[i])
                    elif self.task == 'segment': processed_result = self._process_segment_results(result, input_batch[i])
                    elif self.task == 'pose': processed_result = self._process_pose_results(result, input_batch[i])
                    else: processed_result = result
                    output_queue.put((input_batch[i], processed_result))

        def infer(self, input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
            pending_jobs = collections.deque()

            while True:
                next_batch = input_queue.get()
                if not next_batch:
                    break

                if stop_event.is_set():
                    continue

                input_batch, preprocessed_batch = next_batch
                inference_callback_fn = partial(self._inference_callback, input_batch=input_batch, output_queue=output_queue)

                while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                    pending_jobs.popleft().wait(10000)

                job = self.run(preprocessed_batch, inference_callback_fn)
                pending_jobs.append(job)

            self.close()
            output_queue.put(None)

else:
    HailoInfer = None
    ModelInference = None
