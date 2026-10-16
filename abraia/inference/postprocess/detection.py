"""Preprocessing and decoding for YOLO detection and segmentation models."""

import math

import cv2
import numpy as np

from .common import non_maximum_suppression, sigmoid
from ...tasks import normalize_task


def validate_model_config(config):
    """Validate and normalize metadata required by an ONNX model."""
    if not isinstance(config, dict):
        raise ValueError("ONNX model metadata must be an object")
    task = normalize_task(config.get("task"), default="detection")
    classes = config.get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("ONNX model metadata must define a non-empty classes list")
    input_shape = config.get("inputShape")
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
        raise ValueError("ONNX model metadata inputShape must have four dimensions")
    if any(value is None for value in input_shape[2:]):
        raise ValueError("ONNX model metadata requires static spatial dimensions")
    return task, list(input_shape), list(classes)


def prepare_input(img, shape, return_transform=False):
    """Letterbox an image and optionally return its coordinate transform."""
    model_height, model_width = int(shape[2]), int(shape[3])
    image_height, image_width = img.shape[:2]
    scale = min(model_width / image_width, model_height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    resized = cv2.resize(
        img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_left = (model_width - resized_width) // 2
    pad_top = (model_height - resized_height) // 2
    pad_right = model_width - resized_width - pad_left
    pad_bottom = model_height - resized_height - pad_top
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    tensor = padded.transpose((2, 0, 1)).astype(np.float32) / 255
    tensor = tensor.reshape(shape)
    if return_transform:
        return tensor, scale, (pad_left, pad_top)
    return tensor


def prepare_rtmdet_input(img, shape, return_transform=False):
    """Prepare an image using the RTMDet training pipeline's fixed normalization."""
    model_height, model_width = int(shape[2]), int(shape[3])
    image_height, image_width = img.shape[:2]
    scale = min(model_width / image_width, model_height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    resized = cv2.resize(
        img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mean = np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.asarray([57.375, 57.12, 58.395], dtype=np.float32)
    resized = (resized.astype(np.float32) - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    padded = np.zeros((model_height, model_width, 3), dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    tensor = padded.transpose((2, 0, 1))[None]
    if return_transform:
        return tensor, scale, (0, 0)
    return tensor


def _rtmdet_mask_to_original(
    mask, image_shape, model_shape, resized_shape, padding, box, threshold
):
    """Resize an RTMDet mask to source coordinates and crop it to its box."""
    image_height, image_width = image_shape
    model_height, model_width = model_shape
    resized_height, resized_width = resized_shape
    x1, y1, x2, y2 = box
    mask = np.squeeze(np.asarray(mask, dtype=np.float32))
    if mask.ndim != 2:
        return None
    if mask.shape != (image_height, image_width):
        if mask.shape != (model_height, model_width):
            mask = cv2.resize(
                mask, (model_width, model_height), interpolation=cv2.INTER_LINEAR
            )
        pad_x, pad_y = map(int, padding)
        mask = mask[
            pad_y:pad_y + resized_height,
            pad_x:pad_x + resized_width,
        ]
        mask = cv2.resize(
            mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR
        )
    return (mask[y1:y2, x1:x2] > threshold).astype(np.uint8)


def process_rtmdet_output(
    outputs, output_names, size, shape, classes, conf_threshold=0.25,
    iou_threshold=0.7, transform=None, task="detection",
):
    """Decode MMDeploy RTMDet ONNX outputs into Abraia detections."""
    named_outputs = dict(zip(output_names, outputs))
    dets = next(
        (value for name, value in named_outputs.items() if "det" in name.lower()),
        outputs[0] if outputs else None,
    )
    labels = next(
        (value for name, value in named_outputs.items() if "label" in name.lower()),
        outputs[1] if len(outputs) > 1 else None,
    )
    masks = next(
        (value for name, value in named_outputs.items() if "mask" in name.lower()),
        outputs[2] if len(outputs) > 2 else None,
    )
    if dets is None or labels is None:
        raise ValueError("RTMDet ONNX output must include detections and labels")

    dets = np.asarray(dets)
    labels = np.asarray(labels)
    if dets.ndim == 3:
        dets = dets[0]
    if labels.ndim > 1:
        labels = labels[0]
    if dets.ndim != 2 or dets.shape[-1] < 5:
        raise ValueError(
            "RTMDet ONNX detections must have shape [N, 5] with xyxy boxes and scores"
        )
    if labels.ndim != 1 or len(labels) != len(dets):
        raise ValueError("RTMDet ONNX labels must have one class id per detection")

    image_width, image_height = size
    model_width, model_height = int(shape[3]), int(shape[2])
    scale, padding = _transform_for_output(size, shape, transform)
    mask_array = None if masks is None else np.asarray(masks)
    if mask_array is not None:
        while mask_array.ndim > 3 and mask_array.shape[0] == 1:
            mask_array = mask_array[0]
        if mask_array.ndim == 2:
            mask_array = mask_array[None]
    if task == "segmentation" and mask_array is None:
        raise ValueError(
            "RTMDet segmentation ONNX output must include a masks tensor"
        )
    results = []
    for index, (detection, label) in enumerate(zip(dets, labels)):
        x1, y1, x2, y2, score = [float(value) for value in detection[:5]]
        class_id = int(label)
        if score <= 0 or score < conf_threshold or not 0 <= class_id < len(classes):
            continue
        x1 = round((x1 - padding[0]) / scale)
        y1 = round((y1 - padding[1]) / scale)
        x2 = round((x2 - padding[0]) / scale)
        y2 = round((y2 - padding[1]) / scale)
        x1, x2 = max(0, min(image_width, x1)), max(0, min(image_width, x2))
        y1, y2 = max(0, min(image_height, y1)), max(0, min(image_height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        result = {
            "label": classes[class_id],
            "score": score,
            "box": [x1, y1, x2 - x1, y2 - y1],
            "class_id": class_id,
        }
        if mask_array is not None and index < len(mask_array):
            mask = _rtmdet_mask_to_original(
                mask_array[index],
                (image_height, image_width),
                (model_height, model_width),
                (
                    max(1, round(image_height * scale)),
                    max(1, round(image_width * scale)),
                ),
                padding,
                (x1, y1, x2, y2),
                0.5,
            )
            if mask is not None:
                result["mask"] = mask
        results.append(result)
    if task == "segmentation" and any("mask" not in item for item in results):
        raise ValueError(
            "RTMDet segmentation ONNX output must include per-detection masks"
        )
    return results


def _transform_for_output(size, shape, transform):
    image_width, image_height = size
    model_width, model_height = int(shape[3]), int(shape[2])
    if transform is not None:
        scale, padding = transform
        return float(scale), tuple(map(float, padding))
    scale = min(model_width / image_width, model_height / image_height)
    return scale, (0.0, 0.0)


def _box_from_model_coordinates(xc, yc, width, height, scale, padding):
    pad_x, pad_y = padding
    x1 = round(((xc - width / 2) - pad_x) / scale)
    y1 = round(((yc - height / 2) - pad_y) / scale)
    x2 = round(((xc + width / 2) - pad_x) / scale)
    y2 = round(((yc + height / 2) - pad_y) / scale)
    return [x1, y1, x2 - x1, y2 - y1]


def get_mask(row, box, size, output_size=None, mask_shape=None):
    """Extract a segmentation mask for a model-space object box."""
    x, y, width, height = box
    if mask_shape is None:
        side = round(math.sqrt(row.shape[0]))
        mask_shape = (side, side)
    mask = sigmoid(np.asarray(row).reshape(mask_shape))
    mask_height, mask_width = mask.shape[:2]
    x1 = max(0, round(x / size[0] * mask_width))
    y1 = max(0, round(y / size[1] * mask_height))
    x2 = min(mask_width, round((x + width) / size[0] * mask_width))
    y2 = min(mask_height, round((y + height) / size[1] * mask_height))
    if x2 <= x1 or y2 <= y1:
        target_width, target_height = output_size or (round(width), round(height))
        return np.zeros(
            (max(0, target_height), max(0, target_width)), dtype=np.uint8
        )
    cropped = mask[y1:y2, x1:x2]
    target_width, target_height = output_size or (round(width), round(height))
    resized = cv2.resize(
        cropped, (max(1, target_width), max(1, target_height)), cv2.INTER_NEAREST
    )
    return (resized > 0.5).astype(np.uint8)


def process_output(
    outputs, size, shape, classes, conf_threshold=0.25, iou_threshold=0.7,
    approx=0.001, labels=None, transform=None,
):
    """Decode YOLO detection or instance-segmentation output."""
    del approx
    output0 = np.asarray(outputs[0][0], dtype=float).transpose()
    is_segmentation = len(outputs) == 2
    if is_segmentation:
        output1 = np.asarray(outputs[1][0], dtype=float)
        mask_shape = output1.shape[1:]
        output1 = output1.reshape(output1.shape[0], -1)

    scale, padding = _transform_for_output(size, shape, transform)
    model_width, model_height = int(shape[3]), int(shape[2])
    objects = []
    class_count = len(classes)
    for row in output0:
        xc, yc, width, height = row[:4]
        probs = row[4:4 + class_count]
        class_id = int(probs.argmax())
        score = float(probs[class_id])
        label = classes[class_id]
        if score < conf_threshold or (labels and label not in labels):
            continue
        obj = {
            "label": label,
            "score": score,
            "box": _box_from_model_coordinates(
                xc, yc, width, height, scale, padding
            ),
            "class_id": class_id,
        }
        if is_segmentation:
            obj["mask"] = row[4 + class_count:]
        objects.append(obj)

    results = non_maximum_suppression(objects, iou_threshold)
    if is_segmentation:
        for result in results:
            x, y, width, height = result["box"]
            model_box = [
                round(x * scale + padding[0]),
                round(y * scale + padding[1]),
                round(width * scale),
                round(height * scale),
            ]
            mask = result["mask"] @ output1
            result["mask"] = get_mask(
                mask,
                model_box,
                (model_width, model_height),
                output_size=(max(0, width), max(0, height)),
                mask_shape=mask_shape,
            )
    return results


__all__ = [
    "get_mask",
    "prepare_input",
    "prepare_rtmdet_input",
    "process_rtmdet_output",
    "process_output",
    "validate_model_config",
]
