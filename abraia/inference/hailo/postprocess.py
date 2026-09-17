"""Pure preprocessing and output postprocessing for Hailo models.

The functions in this module do not access HailoRT.  They operate on numpy
arrays and model metadata, which keeps the geometry and decoder behavior
testable on machines without a Hailo device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..postprocess.boxes import nms
from ..postprocess.common import sigmoid, softmax


logger = logging.getLogger(__name__)


SEGMENT_CONFIG = {
    "v5": {
        "arch": "yolov5_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "sizes": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ],
        },
        "input_shape": [640, 640],
        "mask_channels": 32,
        "score_threshold": 0.001,
        "nms_iou_thresh": 0.6,
        "classes": 80,
        "layers": [
            [1, 160, 160, "mask_channels"],
            [1, 20, 20, "detection_channels"],
            [1, 40, 40, "detection_channels"],
            [1, 80, 80, "detection_channels"],
        ],
    },
    "v8": {
        "arch": "yolov8_seg",
        "anchors": {"strides": [8, 16, 32], "regression_length": 15},
        "input_shape": [640, 640],
        "mask_channels": 32,
        "score_threshold": 0.001,
        "nms_iou_thresh": 0.7,
        "meta_arch": "yolov8_seg_postprocess",
        "classes": 80,
        "layers": [
            [1, 20, 20, "detection_output_channels"],
            [1, 20, 20, "classes"],
            [1, 20, 20, "mask_channels"],
            [1, 40, 40, "detection_output_channels"],
            [1, 40, 40, "classes"],
            [1, 40, 40, "mask_channels"],
            [1, 80, 80, "detection_output_channels"],
            [1, 80, 80, "classes"],
            [1, 80, 80, "mask_channels"],
            [1, 160, 160, "mask_channels"],
        ],
    },
}


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
    """Resize and letterbox an image to the model input dimensions."""
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


def resize_mask_to_unpadded_box(mask_1d, box_on_input_image, box_on_padded_image):
    """Validate and resize a Hailo byte-mask ROI to its original-image box."""
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
        return cv2.resize(
            mask_2d.astype(np.uint8),
            (x2_u - x1_u, y2_u - y1_u),
            interpolation=cv2.INTER_NEAREST,
        )
    except Exception:
        return None


def convert_nms_box_from_normalized(normalized_box, model_dim, original_dim):
    """Map a Hailo byte-mask box to original and model-space coordinates."""
    model_height, model_width = model_dim
    normalized_box = np.clip(np.asarray(normalized_box, dtype=np.float32), 0.0, 1.0)
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
    """Convert a normalized box to original and padded-image coordinates."""
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
    return transform.box_to_original(box_on_padded_image), box_on_padded_image


def segment_xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def segment_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                                max_det=300, nm=32, multi_label=True):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    mi = 5 + nc
    output = []
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
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
            x = np.concatenate(
                (boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]),
                1,
            )

        finite = np.isfinite(x[:, :5]).all(axis=1)
        if not finite.all():
            logger.debug(
                "Discarding %d non-finite segmentation candidates",
                np.count_nonzero(~finite),
            )
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
            "detection_scores": out[:, 4],
        })
    return output


def segment_process_mask_optimized(protos, masks_in, bboxes, shape,
                                   upsample=True, downsample=False):
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
    _, height, width = masks.shape
    output = np.zeros_like(masks)
    boxes = np.round(boxes).astype(int)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
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
    batch_size, height, width = output.shape[0:3]
    stride = stride_list[branch_idx]
    anchors = anchor_list[branch_idx] / stride
    num_anchors = len(anchors) // 2
    grid, anchor_grid = segment_make_grid(anchors, stride, batch_size, width, height)
    output = output.transpose((0, 3, 1, 2)).reshape(
        (batch_size, num_anchors, -1, height, width)
    ).transpose((0, 1, 3, 4, 2))
    xy, wh, conf, mask = np.array_split(
        output, [2, 4, 4 + num_classes + 1], axis=4
    )
    xy = (sigmoid(xy) * 2 + grid) * stride
    wh = (sigmoid(wh) * 2) ** 2 * anchor_grid
    out = np.concatenate((xy, wh, sigmoid(conf), mask), 4)
    return out.reshape((batch_size, num_anchors * height * width, -1)).astype(np.float32)


def normalize_yolov8_scores(scores):
    """Return YOLOv8 class scores as probabilities."""
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
        grid_x, grid_y = np.meshgrid(
            np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5
        )
        ct_row, ct_col = grid_y.flatten() * stride, grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(
            box_distribute,
            (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1),
        )
        finite_distribution = np.isfinite(box_distribute).all(axis=(2, 3))
        safe_distribution = np.where(
            finite_distribution[..., None, None], box_distribute, 0
        )
        box_distance = softmax(safe_distribution)
        box_distance[~finite_distribution] = np.nan
        box_distance = np.sum(
            box_distance * np.reshape(reg_range, (1, 1, 1, -1)), axis=-1
        ) * stride
        box_distance = np.concatenate(
            [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1
        )
        decode_box = np.expand_dims(center, axis=0) + box_distance
        xmin, ymin = decode_box[:, :, 0], decode_box[:, :, 1]
        xmax, ymax = decode_box[:, :, 2], decode_box[:, :, 3]
        xywh_box = np.transpose(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin],
            [1, 2, 0],
        )
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes


def segment_yolov5_postprocess(endnodes, **kwargs):
    img_dims = tuple(kwargs["input_shape"])
    protos = endnodes[0]
    anchor_list = np.array(kwargs["anchors"]["sizes"][::-1])
    stride_list = kwargs["anchors"]["strides"][::-1]
    num_classes = kwargs["classes"]
    outputs = [
        segment_yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes)
        for branch_idx, output in enumerate(endnodes[1:])
    ]
    outputs = segment_non_max_suppression(
        np.concatenate(outputs, 1),
        kwargs["score_threshold"],
        kwargs["nms_iou_thresh"],
        nm=protos.shape[-1],
    )
    for batch_idx, output in enumerate(outputs):
        output["mask"] = segment_process_mask_optimized(
            protos[batch_idx].astype(np.float32, copy=False),
            output["mask"].astype(np.float32, copy=False),
            output["detection_boxes"],
            img_dims,
            upsample=True,
        )
        output["detection_boxes"][:, [0, 2]] /= img_dims[1]
        output["detection_boxes"][:, [1, 3]] /= img_dims[0]
    return outputs


def segment_yolov8_postprocess(endnodes, **kwargs):
    num_classes = kwargs["classes"]
    strides = kwargs["anchors"]["strides"][::-1]
    image_dims = tuple(kwargs["input_shape"])
    reg_max = kwargs["anchors"]["regression_length"]
    raw_boxes = endnodes[:7:3]
    scores = np.concatenate(
        [np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes))
         for s in endnodes[1:8:3]],
        axis=1,
    )
    scores = normalize_yolov8_scores(scores)
    decoded_boxes = segment_yolov8_decoding(raw_boxes, strides, image_dims, reg_max)
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape
    scores_obj = np.concatenate(
        [np.ones((scores.shape[0], scores.shape[1], 1)), scores], axis=-1
    )
    coeffs = np.concatenate(
        [np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks))
         for c in endnodes[2:9:3]],
        axis=1,
    )
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    nms_res = segment_non_max_suppression(
        predictions,
        conf_thres=kwargs["score_threshold"],
        iou_thres=kwargs["nms_iou_thresh"],
        multi_label=False,
    )
    outputs = []
    for batch_idx in range(batch_size):
        masks = segment_process_mask_optimized(
            proto_data[batch_idx].astype(np.float32, copy=False),
            nms_res[batch_idx]["mask"].astype(np.float32, copy=False),
            nms_res[batch_idx]["detection_boxes"],
            image_dims,
        )
        outputs.append({
            "detection_boxes": np.array(nms_res[batch_idx]["detection_boxes"]) / np.tile(image_dims, 2),
            "mask": masks,
            "detection_scores": np.array(nms_res[batch_idx]["detection_scores"]),
            "detection_classes": np.array(nms_res[batch_idx]["detection_classes"]).astype(int),
        })
    return outputs


def decode_pose_results(raw_boxes: np.ndarray, raw_kpts: np.ndarray,
                        strides: List[int], image_dims: Tuple[int, int],
                        reg_max: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes = None
    decoded_kpts = None
    for box_distribute, kpts, stride in zip(raw_boxes, raw_kpts, strides):
        shape = [int(x / stride) for x in image_dims]
        grid_x, grid_y = np.meshgrid(
            np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5
        )
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
        box_distance[~finite_distribution] = np.nan
        box_distance = np.sum(
            box_distance * np.reshape(np.arange(reg_max + 1), (1, 1, 1, -1)),
            axis=-1,
        ) * stride
        decode_box = np.expand_dims(center, axis=0) + np.concatenate(
            [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1
        )
        xmin, ymin = decode_box[:, :, 0], decode_box[:, :, 1]
        xmax, ymax = decode_box[:, :, 2], decode_box[:, :, 3]
        xywh_box = np.transpose(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin],
            [1, 2, 0],
        )
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

        decoded_kpts_for_layer = np.array(kpts, copy=True)
        decoded_kpts_for_layer[..., :2] = (
            stride * (decoded_kpts_for_layer[..., :2] * 2 - 0.5)
            + center[None, :, None, :2]
        )
        decoded_kpts = (
            decoded_kpts_for_layer
            if decoded_kpts is None
            else np.concatenate([decoded_kpts, decoded_kpts_for_layer], axis=1)
        )
    return boxes, decoded_kpts


def map_box_to_orig(box: list, orig_dim: Tuple[int, int], model_dim: Tuple[int, int]) -> list:
    return LetterboxTransform.from_dimensions(orig_dim, model_dim).box_to_original(box)


def map_keypoints_to_orig(keypoints: np.ndarray, orig_dim: Tuple[int, int],
                          model_dim: Tuple[int, int]) -> np.ndarray:
    return LetterboxTransform.from_dimensions(
        orig_dim, model_dim
    ).keypoints_to_original(keypoints)


def map_mask_to_orig(mask: np.ndarray, orig_dim: Tuple[int, int],
                     model_dim: Tuple[int, int]) -> np.ndarray:
    return LetterboxTransform.from_dimensions(
        orig_dim, model_dim
    ).mask_to_original(mask)


def resolve_shape(layer, model_type, arch_cfg):
    batch, height, width, channel_tag = layer
    mask_channels = arch_cfg["mask_channels"]
    if isinstance(channel_tag, str):
        if channel_tag == "mask_channels":
            channels = mask_channels
        elif channel_tag == "detection_channels":
            channels = (arch_cfg["classes"] + 4 + 1 + mask_channels) * len(arch_cfg["anchors"]["strides"])
        elif channel_tag == "detection_output_channels":
            channels = (
                (arch_cfg["classes"] + 4 + 1 + mask_channels)
                * len(arch_cfg["anchors"]["strides"])
                if model_type == "v5"
                else (arch_cfg["anchors"]["regression_length"] + 1) * 4
            )
        elif channel_tag == "classes":
            channels = arch_cfg["classes"]
        else:
            raise ValueError(f"Unsupported channel tag: {channel_tag}")
    else:
        channels = channel_tag
    return batch, height, width, channels


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

    missing = [
        tuple(shape) for shape in expected_shapes
        if tuple(shape) not in layers_by_shape
    ]
    if missing:
        available = ", ".join(map(str, layers_by_shape)) or "none"
        raise ValueError(
            f"Hailo model is missing output shapes {missing}; available: {available}"
        )
    return [outputs[layers_by_shape[tuple(shape)]] for shape in expected_shapes]


__all__ = [
    "SEGMENT_CONFIG",
    "LetterboxTransform",
    "default_preprocess",
    "resize_mask_to_unpadded_box",
    "convert_nms_box_from_normalized",
    "convert_box_from_normalized",
    "segment_xywh2xyxy",
    "segment_non_max_suppression",
    "segment_process_mask_optimized",
    "segment_crop_mask_roi_vectorized",
    "segment_make_grid",
    "segment_yolov5_decoding",
    "normalize_yolov8_scores",
    "segment_yolov8_decoding",
    "segment_yolov5_postprocess",
    "segment_yolov8_postprocess",
    "decode_pose_results",
    "map_box_to_orig",
    "map_keypoints_to_orig",
    "map_mask_to_orig",
    "resolve_shape",
    "select_output_layers",
]
