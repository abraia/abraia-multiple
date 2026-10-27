"""Preprocessing, decoding, and inference for detection models."""

from functools import partial

import math
import cv2
import numpy as np

from ..geometry import non_maximum_suppression, sigmoid
from ..model_config import normalize_task
from ..session import OnnxSessionMixin, load_onnx_model_config


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


def _resize_geometry(image_shape, model_shape):
    """Return the scale and rounded size for fitting an image to a model."""
    image_height, image_width = image_shape
    model_height, model_width = model_shape
    scale = min(model_width / image_width, model_height / image_height)
    return (
        scale,
        max(1, round(image_width * scale)),
        max(1, round(image_height * scale)),
    )


def prepare_input(img, shape, return_transform=False):
    """Letterbox an image and optionally return its coordinate transform."""
    model_height, model_width = int(shape[2]), int(shape[3])
    image_height, image_width = img.shape[:2]
    scale, resized_width, resized_height = _resize_geometry(
        (image_height, image_width), (model_height, model_width)
    )
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
    scale, resized_width, resized_height = _resize_geometry(
        (image_height, image_width), (model_height, model_width)
    )
    resized = cv2.resize(
        img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
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
    outputs, size, shape, classes, output_names, conf_threshold=0.25,
    iou_threshold=0.7, approx=0.001, labels=None, transform=None,
    task="detection", config=None,
):
    """Decode native or deployment-converted RTMDet ONNX outputs."""
    del approx
    config = config or {}
    raw_layout = config.get("rawOutputs")
    if isinstance(raw_layout, dict):
        outputs = _decode_rtmdet_raw(
            outputs,
            output_names,
            raw_layout,
            classes,
            conf_threshold,
            iou_threshold,
            task,
        )
    named_outputs = dict(zip(output_names, outputs))
    dets = next(
        (value for name, value in named_outputs.items() if "det" in name.lower()),
        outputs[0] if outputs else None,
    )
    class_ids = next(
        (value for name, value in named_outputs.items() if "label" in name.lower()),
        outputs[1] if len(outputs) > 1 else None,
    )
    masks = next(
        (value for name, value in named_outputs.items() if "mask" in name.lower()),
        outputs[2] if len(outputs) > 2 else None,
    )
    if dets is None or class_ids is None:
        raise ValueError("RTMDet ONNX output must include detections and labels")

    dets = np.asarray(dets)
    class_ids = np.asarray(class_ids)
    if dets.ndim == 3:
        dets = dets[0]
    if class_ids.ndim > 1:
        class_ids = class_ids[0]
    if dets.ndim != 2 or dets.shape[-1] < 5:
        raise ValueError(
            "RTMDet ONNX detections must have shape [N, 5] with xyxy boxes and scores"
        )
    if class_ids.ndim != 1 or len(class_ids) != len(dets):
        raise ValueError("RTMDet ONNX labels must have one class id per detection")

    image_width, image_height = size
    model_width, model_height = int(shape[3]), int(shape[2])
    scale, padding = _transform_for_output(size, shape, transform)
    mask_array = None if masks is None else np.asarray(masks)
    if mask_array is not None:
        mask_array = np.squeeze(mask_array)
        if mask_array.ndim == 2:
            mask_array = mask_array[None]
    if task == "segmentation" and mask_array is None:
        raise ValueError(
            "RTMDet segmentation ONNX output must include a masks tensor"
        )
    results = []
    for index, (detection, label) in enumerate(zip(dets, class_ids)):
        x1, y1, x2, y2, score = [float(value) for value in detection[:5]]
        class_id = int(label)
        if (
            score <= 0
            or score < conf_threshold
            or not 0 <= class_id < len(classes)
            or (labels and classes[class_id] not in labels)
        ):
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
        if mask_array is not None:
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


def _decode_rtmdet_raw(
    outputs, output_names, layout, classes, conf_threshold, iou_threshold, task
):
    """Decode raw RTMDet head tensors exported by Abraia's PyTorch exporter."""
    named = dict(zip(output_names, outputs))
    class_names = layout.get("classScores", [])
    box_names = layout.get("boxDistances", [])
    kernel_names = layout.get("maskKernels", [])
    strides = layout.get("strides", [])
    if not class_names or len(class_names) != len(box_names) or len(strides) != len(class_names):
        raise ValueError("RTMDet raw ONNX metadata has an invalid output layout")

    max_pre = int(layout.get("nmsPre", 30000))
    score_threshold = max(float(layout.get("scoreThreshold", 0.001)), conf_threshold)
    nms_threshold = float(layout.get("nmsThreshold", iou_threshold))
    candidates = []
    for level, (score_name, box_name, stride) in enumerate(
        zip(class_names, box_names, strides)
    ):
        scores = np.asarray(named[score_name], dtype=np.float32)[0]
        distances = np.asarray(named[box_name], dtype=np.float32)[0]
        if scores.ndim != 3 or distances.ndim != 3:
            raise ValueError("RTMDet raw outputs must use NCHW tensors")
        scores = sigmoid(scores).transpose(1, 2, 0).reshape(-1, len(classes))
        distances = distances.transpose(1, 2, 0).reshape(-1, 4)
        if layout.get("bboxExponent", False):
            distances = np.exp(np.clip(distances, -20, 20)) * float(stride)
        elif not layout.get("bboxDistancesScaled", False):
            distances *= float(stride)
        height, width = np.asarray(named[score_name]).shape[-2:]
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        centers = np.stack((x.reshape(-1), y.reshape(-1)), axis=1).astype(np.float32)
        centers *= float(stride)

        flat_indices = np.flatnonzero(scores.reshape(-1) >= score_threshold)
        if not len(flat_indices):
            continue
        flat_scores = scores.reshape(-1)[flat_indices]
        if max_pre > 0 and len(flat_indices) > max_pre:
            selected = np.argpartition(flat_scores, -max_pre)[-max_pre:]
            flat_indices = flat_indices[selected]
            flat_scores = flat_scores[selected]
        location_indices = flat_indices // len(classes)
        labels = flat_indices % len(classes)
        distances_for_scores = distances[location_indices]
        points = centers[location_indices]
        boxes = np.stack((
            points[:, 0] - distances_for_scores[:, 0],
            points[:, 1] - distances_for_scores[:, 1],
            points[:, 0] + distances_for_scores[:, 2],
            points[:, 1] + distances_for_scores[:, 3],
        ), axis=1)
        candidates.extend(
            (float(score), int(label), box, int(level), int(location))
            for score, label, box, location in zip(
                flat_scores, labels, boxes, location_indices
            )
        )

    if not candidates:
        empty = [
            np.empty((0, 5), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        ]
        if task == "segmentation":
            feature_shape = np.asarray(named[layout["maskFeatures"]]).shape[-2:]
            scale = int(layout["maskFeatureStride"])
            empty.append(np.empty((0, feature_shape[0] * scale,
                                   feature_shape[1] * scale), dtype=np.float32))
        return empty

    boxes = np.asarray([item[2] for item in candidates], dtype=np.float32)
    scores = np.asarray([item[0] for item in candidates], dtype=np.float32)
    labels = np.asarray([item[1] for item in candidates], dtype=np.int64)
    keep = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        class_boxes = boxes[indices]
        class_scores = scores[indices]
        areas = np.maximum(class_boxes[:, 2] - class_boxes[:, 0], 0) * np.maximum(
            class_boxes[:, 3] - class_boxes[:, 1], 0
        )
        order = class_scores.argsort()[::-1]
        while order.size:
            selected = order[0]
            keep.append(int(indices[selected]))
            remaining = order[1:]
            x1 = np.maximum(class_boxes[selected, 0], class_boxes[remaining, 0])
            y1 = np.maximum(class_boxes[selected, 1], class_boxes[remaining, 1])
            x2 = np.minimum(class_boxes[selected, 2], class_boxes[remaining, 2])
            y2 = np.minimum(class_boxes[selected, 3], class_boxes[remaining, 3])
            intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
            union = areas[selected] + areas[remaining] - intersection
            overlap = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[overlap <= nms_threshold]
    keep = sorted(keep, key=lambda index: scores[index], reverse=True)
    max_detections = int(layout.get("maxDetections", 300))
    keep = keep[:max_detections]
    detections = np.column_stack((boxes[keep], scores[keep])).astype(np.float32)
    kept_labels = labels[keep]

    if task != "segmentation":
        return [detections[None], kept_labels[None]]
    if len(kernel_names) != len(class_names):
        raise ValueError("RTMDet segmentation outputs are missing mask kernels")
    mask_feat = np.asarray(named[layout["maskFeatures"]], dtype=np.float32)[0]
    masks = _rtmdet_raw_masks(
        mask_feat,
        named,
        kernel_names,
        candidates,
        keep,
        layout,
        strides,
    )
    return [detections[None], kept_labels[None], masks[None]]


def _rtmdet_raw_masks(
    mask_feat, named, kernel_names, candidates, keep, layout, strides
):
    """Run RTMDet-Ins dynamic mask convolutions for retained detections."""
    import cv2

    prototype_count, height, width = mask_feat.shape
    channels = int(layout["dyconvChannels"])
    layers = int(layout["dyconvLayers"])
    weight_sizes = [(prototype_count + 2) * channels]
    bias_sizes = [channels]
    for _ in range(1, layers - 1):
        weight_sizes.append(channels * channels)
        bias_sizes.append(channels)
    weight_sizes.append(channels)
    bias_sizes.append(1)
    sizes = weight_sizes + bias_sizes
    feature_stride = int(layout["maskFeatureStride"])
    gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack((gx, gy), axis=0).astype(np.float32) * feature_stride
    masks = []
    for candidate_index in keep:
        _, _, _, level, location = candidates[candidate_index]
        stride = float(strides[level])
        score_tensor = np.asarray(named[layout["classScores"][level]])[0]
        feature_width = score_tensor.shape[-1]
        center = np.asarray(
            [(location % feature_width) * stride, (location // feature_width) * stride],
            dtype=np.float32,
        )
        relative = (center[:, None, None] - grid) / (stride * 8.0)
        x = np.concatenate((relative, mask_feat), axis=0)[None]
        kernel = np.asarray(named[kernel_names[level]], dtype=np.float32)[0]
        kernel = kernel.reshape(kernel.shape[0], -1)
        params = kernel[:, location]
        split = np.split(params, np.cumsum(sizes)[:-1])
        weights, biases = split[:layers], split[layers:]
        for layer in range(layers):
            out_channels = channels if layer < layers - 1 else 1
            in_channels = x.shape[1]
            weight = weights[layer].reshape(out_channels, in_channels, 1, 1)
            bias = biases[layer].reshape(out_channels)
            x = np.einsum("oc, bchw -> bohw", weight[:, :, 0, 0], x)
            x += bias[None, :, None, None]
            if layer < layers - 1:
                x = np.maximum(x, 0)
        logits = x[0, 0]
        logits = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
        mask = cv2.resize(
            logits, (mask_feat.shape[-1] * feature_stride,
                     mask_feat.shape[-2] * feature_stride),
            interpolation=cv2.INTER_LINEAR,
        )
        masks.append(mask.astype(np.float32))
    if not masks:
        return np.empty((0, height * feature_stride, width * feature_stride), dtype=np.float32)
    return np.asarray(masks, dtype=np.float32)


def _transform_for_output(size, shape, transform):
    image_width, image_height = size
    model_width, model_height = int(shape[3]), int(shape[2])
    if transform is not None:
        scale, padding = transform
        return float(scale), tuple(map(float, padding))
    scale, _resized_width, _resized_height = _resize_geometry(
        (image_height, image_width), (model_height, model_width)
    )
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


def process_libreyolo_output(
    outputs, size, shape, classes, conf_threshold=0.25, iou_threshold=0.45,
    approx=0.001, labels=None, transform=None,
):
    """Decode LibreYOLO YOLOv9 ONNX output with embedded NMS."""
    del approx
    detections = np.asarray(outputs[0], dtype=np.float32)
    if detections.ndim == 3:
        detections = detections[0]
    if detections.ndim != 2 or detections.shape[1] < 6:
        raise ValueError(
            "LibreYOLO YOLOv9 ONNX output must have shape [N, 6] "
            "with xyxy boxes, score, and class id"
        )

    image_width, image_height = size
    scale, padding = _transform_for_output(size, shape, transform)
    grouped = {}
    for detection in detections:
        x1, y1, x2, y2, score = [float(value) for value in detection[:5]]
        class_id = int(detection[5])
        if (
            score <= 0
            or score < conf_threshold
            or not 0 <= class_id < len(classes)
        ):
            continue
        label = classes[class_id]
        if labels and label not in labels:
            continue
        x1 = max(0, min(image_width, round((x1 - padding[0]) / scale)))
        y1 = max(0, min(image_height, round((y1 - padding[1]) / scale)))
        x2 = max(0, min(image_width, round((x2 - padding[0]) / scale)))
        y2 = max(0, min(image_height, round((y2 - padding[1]) / scale)))
        if x2 <= x1 or y2 <= y1:
            continue
        grouped.setdefault(class_id, []).append({
            "label": label,
            "score": score,
            "box": [x1, y1, x2 - x1, y2 - y1],
            "class_id": class_id,
        })

    results = []
    for objects in grouped.values():
        results.extend(non_maximum_suppression(objects, iou_threshold))
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def process_pose_output(
    outputs, size, shape, classes, conf_threshold=0.25, iou_threshold=0.7,
    approx=0.001, labels=None, transform=None,
):
    """Decode raw YOLO pose-estimation output."""
    del approx
    output = np.asarray(outputs[0][0], dtype=float).transpose()
    scale, padding = _transform_for_output(size, shape, transform)
    class_count = len(classes)
    keypoint_start = 4 + class_count
    keypoint_values = output.shape[1] - keypoint_start
    if keypoint_values <= 0 or keypoint_values % 3:
        raise ValueError(
            "Pose-estimation output does not contain complete keypoint triples"
        )
    keypoint_count = keypoint_values // 3

    objects = []
    pad_x, pad_y = padding
    for row in output:
        xc, yc, width, height = row[:4]
        probs = row[4:keypoint_start]
        class_id = int(probs.argmax())
        score = float(probs[class_id])
        label = classes[class_id]
        if score < conf_threshold or (labels and label not in labels):
            continue
        keypoints = row[keypoint_start:].reshape(keypoint_count, 3).copy()
        keypoints[:, 0] = (keypoints[:, 0] - pad_x) / scale
        keypoints[:, 1] = (keypoints[:, 1] - pad_y) / scale
        joint_scores = keypoints[:, 2]
        if np.any((joint_scores < 0) | (joint_scores > 1)):
            joint_scores = sigmoid(joint_scores)
        objects.append({
            "label": label,
            "score": score,
            "box": _box_from_model_coordinates(
                xc, yc, width, height, scale, padding
            ),
            "class_id": class_id,
            "keypoints": keypoints[:, :2],
            "joint_scores": joint_scores,
        })
    return non_maximum_suppression(objects, iou_threshold)


class Model(OnnxSessionMixin):
    def __init__(self, model_uri, providers=None, accelerator=None, kind=None):
        model_path, config = load_onnx_model_config(model_uri)
        self.task, self.input_shape, self.classes = validate_model_config(
            config
        )
        self.kind = str(kind or config.get("kind", "yolov8")).strip().lower()
        if self.task == "classification":
            raise ValueError(
                "Classification ONNX models must be loaded with ResNetClassifier"
            )
        if self.task not in ("detection", "segmentation", "pose"):
            raise ValueError(
                "Detection ONNX models support detection, segmentation, or pose"
            )
        self._init_onnx_session(
            model_path,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name
        output_names = [item.name for item in self.session.get_outputs()]
        if self.kind == "rtmdet":
            self.decoder = partial(
                process_rtmdet_output,
                output_names=output_names,
                classes=list(self.classes),
                task=self.task,
                config=config,
            )
        elif self.kind == "yolo9":
            self.decoder = partial(
                process_libreyolo_output,
                classes=list(self.classes),
            )
        else:
            decoder = process_pose_output if self.task == "pose" else process_output
            self.decoder = partial(decoder, classes=list(self.classes))

    def run(
        self,
        img,
        conf_threshold=0.35,
        iou_threshold=0.7,
        approx=0.001,
        labels=None,
        top_k=1,
        score_threshold=None,
    ):
        self._ensure_open()
        img_size = img.shape[1], img.shape[0]
        if self.kind == "rtmdet":
            input_tensor, scale, padding = prepare_rtmdet_input(
                img,
                self.input_shape,
                return_transform=True,
            )
        else:
            input_tensor, scale, padding = prepare_input(
                img, self.input_shape, return_transform=True
            )
        inputs = {self.input_name: input_tensor}
        outputs = self.session.run(None, inputs)
        return self.decoder(
            outputs,
            img_size,
            self.input_shape,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            approx=approx,
            labels=labels,
            transform=(scale, padding),
        )

__all__ = [
    "Model", "get_mask", "prepare_input", "prepare_rtmdet_input",
    "process_output", "process_pose_output",
    "process_libreyolo_output", "process_rtmdet_output",
    "validate_model_config",
]
