import cv2
import queue
import threading
import collections
import numpy as np

from pathlib import Path
from functools import partial
from types import SimpleNamespace
from scipy.special import expit
from concurrent.futures import ThreadPoolExecutor

import logging
logger = logging.getLogger(__name__)

from .tracker.byte_tracker import BYTETracker
from .tracker.matching import find_best_matching_mask_index
from .toolbox import (
    VideoInput,
    VideoVisualizer,
    get_labels,
    resolve_hef_path,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS,
    HailoInfer,
)

from ..inference.ops import nms, sigmoid, softmax
from ..utils.draw import (
    draw_rectangle, draw_text, calculate_optimal_thickness, calculate_optimal_text_scale,
    draw_overlay_mask, get_color, hex_to_rgb, render_results
)

APP_NAME = Path(__file__).stem

DEFAULT_OPTIONS = {
    "input": None,
    "hef_path": None,
    "batch_size": 1,
    "frame_rate": None,
    "model_type": "v5",
    "track": False,
    "labels": None,
    "camera_resolution": None,
    "video_unpaced": False,
    "output_dir": None,
    "save_output": False,
}

CONFIG_DATA = {
    "v5": {
        "arch": "yolov5_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "sizes": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]
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
            [1, 80, 80, "detection_channels"]
        ]
    },
    "v8": {
        "arch": "yolov8_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "regression_length": 15
        },
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
            [1, 160, 160, "mask_channels"]
        ]
    },
    "visualization_params": {
        "score_thres": 0.25,
        "mask_thresh": 0.4,
        "mask_alpha": 0.5,
        "max_boxes_to_draw": 50,
        "tracker": {
            "track_thresh": 0.01,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "aspect_ratio_thresh": 2.0,
            "min_box_area": 500,
            "mot20": False
        }
    }
}


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=32, multi_label=True):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Args:
        prediction: numpy.ndarray with shape (batch_size, num_proposals, 351)
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximal number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals
    Returns:
         A list of per image detections, where each is a dictionary with the following structure:
         {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
         }
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, 32)),
                    "detection_classes": np.zeros((0, 80)),
                    "detection_scores": np.zeros((0, 80)),
                }
            )
            continue

        # Confidence = Objectness X Class Score
        x[:, 5:] *= x[:, 4:5]

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
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

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        # per-class NMS
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])
        keep = nms(preds, iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        out = x[keep]
        scores = out[:, 4]
        classes = out[:, 5]
        boxes = out[:, :4]
        masks = out[:, 6:]

        out = {"detection_boxes": boxes, "mask": masks, "detection_classes": classes, "detection_scores": scores}

        output.append(out)

    return output


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def crop_mask(masks, boxes):
    """
    Zeroing out mask region outside of the predicted bbox.
    Args:
        masks: numpy array of masks with shape [n, h, w]
        boxes: numpy array of bbox coords with shape [n, 4]
    """

    n_masks, _, _ = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)
    for k in range(n_masks):
        masks[k, : y1[k, 0], :] = 0
        masks[k, y2[k, 0] :, :] = 0
        masks[k, :, : x1[k, 0]] = 0
        masks[k, :, x2[k, 0] :] = 0
    return masks


def process_mask(protos, masks_in, bboxes, shape, upsample=True, downsample=False):

    mh, mw, c = protos.shape
    ih, iw = shape
    masks = sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape((-1, mh, mw))

    downsampled_bboxes = bboxes.copy()
    if downsample:
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        if not masks.shape[0]:
            return None
        masks = cv2.resize(np.transpose(masks, axes=(1, 2, 0)), shape, interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = np.transpose(masks, axes=(2, 0, 1))  # CHW

    if not downsample:
        masks = crop_mask(masks, downsampled_bboxes)  # CHW

    return masks


def _yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes):
    BS, H, W = output.shape[0:3]
    stride = stride_list[branch_idx]
    anchors = anchor_list[branch_idx] / stride
    num_anchors = len(anchors) // 2

    grid, anchor_grid = _make_grid(anchors, stride, BS, W, H)

    output = output.transpose((0, 3, 1, 2)).reshape((BS, num_anchors, -1, H, W)).transpose((0, 1, 3, 4, 2))
    xy, wh, conf, mask = np.array_split(output, [2, 4, 4 + num_classes + 1], axis=4)

    # decoding
    xy = (sigmoid(xy) * 2 + grid) * stride
    wh = (sigmoid(wh) * 2) ** 2 * anchor_grid

    out = np.concatenate((xy, wh, sigmoid(conf), mask), 4)
    out = out.reshape((BS, num_anchors * H * W, -1)).astype(np.float32)

    return out


def yolov5_seg_postprocess(endnodes, device_pre_post_layers=None, **kwargs):
    """
    endnodes is a list of 4 tensors:
        endnodes[0]:  mask protos with shape (BS, 160, 160, 32)
        endnodes[1]:  stride 32 of input with shape (BS, 20, 20, 351)
        endnodes[2]:  stride 16 of input with shape (BS, 40, 40, 351)
        endnodes[3]:  stride 8 of input with shape (BS, 80, 80, 351)
    Returns:
        A list of per image detections, where each is a dictionary with the following structure:
        {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
        }
    """
    img_dims = tuple(kwargs["input_shape"])
    if kwargs.get("hpp", False):
        # the outputs where decoded by the emulator (as part of the network)
        # organizing the output for evaluation
        return _organize_hpp_yolov5_seg_outputs(endnodes, img_dims=img_dims)

    protos = endnodes[0]
    anchor_list = np.array(kwargs["anchors"]["sizes"][::-1])
    stride_list = kwargs["anchors"]["strides"][::-1]
    num_classes = kwargs["classes"]

    outputs = []
    for branch_idx, output in enumerate(endnodes[1:]):
        decoded_info = _yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes)
        outputs.append(decoded_info)

    outputs = np.concatenate(outputs, 1)  # (BS, num_proposals, 117)

    # NMS
    score_thres = kwargs["score_threshold"]
    iou_thres = kwargs["nms_iou_thresh"]
    outputs = non_max_suppression(outputs, score_thres, iou_thres, nm=protos.shape[-1])
    outputs = _finalize_detections_yolov5_seg(outputs, protos, **kwargs)

    # reorder and normalize bboxes
    for output in outputs:
        output["detection_boxes"] = _normalize_yolov5_seg_bboxes(output, img_dims)
    return outputs


def _organize_hpp_yolov5_seg_outputs(outputs, img_dims):
    # the outputs structure is [-1, num_of_proposals, 6 + h*w]
    # were the mask information is ordered as follows
    # [x_min, y_min, x_max, y_max, score, class, flattened mask] of each detection
    # this function separates the structure to informative dict
    predictions = []
    batch_size, num_of_proposals = outputs.shape[0], outputs.shape[-1]
    outputs = np.transpose(np.squeeze(outputs, axis=1), [0, 2, 1])
    for i in range(batch_size):
        predictions.append(
            {
                "detection_boxes": outputs[i, :, :4][:, [1, 0, 3, 2]],
                "detection_scores": outputs[i, :, 4],
                "detection_classes": outputs[i, :, 5],
                "mask": outputs[i, :, 6:].reshape((num_of_proposals, *img_dims)),
            }
        )
    return predictions


def _normalize_yolov5_seg_bboxes(output, img_dims):
    # normalizes bboxes and change the bboxes format to y_min, x_min, y_max, x_max
    bboxes = output["detection_boxes"]
    bboxes[:, [0, 2]] /= img_dims[1]
    bboxes[:, [1, 3]] /= img_dims[0]

    return bboxes


def _finalize_detections_yolov5_seg(outputs, protos, **kwargs):
    for batch_idx, output in enumerate(outputs):
        shape = tuple(kwargs["input_shape"])
        boxes = output["detection_boxes"]
        masks = output["mask"]
        proto = protos[batch_idx]
        masks = process_mask_optimized(proto.astype(np.float32, copy=False), masks.astype(np.float32, copy=False), boxes, shape, upsample=True)
        output["mask"] = masks

    return outputs


def _make_grid(anchors, stride, bs=8, nx=20, ny=20):
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


def _yolov8_decoding(raw_boxes, strides, image_dims, reg_max):
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        # create grid
        shape = [int(x / stride) for x in image_dims]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        # box distribution to distance
        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(
            box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)
        )
        box_distance = softmax(box_distribute)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1)
        box_distance = box_distance * stride

        # decode box
        box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        decode_box = np.expand_dims(center, axis=0) + box_distance

        xmin = decode_box[:, :, 0]
        ymin = decode_box[:, :, 1]
        xmax = decode_box[:, :, 2]
        ymax = decode_box[:, :, 3]
        decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])

        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes  # tf.expand_dims(boxes, axis=2)


def yolov8_seg_postprocess(endnodes, device_pre_post_layers=None, **kwargs):
    """
    endnodes is a list of 10 tensors:
        endnodes[0]:  bbox output with shapes (BS, 20, 20, 64)
        endnodes[1]:  scores output with shapes (BS, 20, 20, 80)
        endnodes[2]:  mask coeff output with shapes (BS, 20, 20, 32)
        endnodes[3]:  bbox output with shapes (BS, 40, 40, 64)
        endnodes[4]:  scores output with shapes (BS, 40, 40, 80)
        endnodes[5]:  mask coeff output with shapes (BS, 40, 40, 32)
        endnodes[6]:  bbox output with shapes (BS, 80, 80, 64)
        endnodes[7]:  scores output with shapes (BS, 80, 80, 80)
        endnodes[8]:  mask coeff output with shapes (BS, 80, 80, 32)
        endnodes[9]:  mask protos with shape (BS, 160, 160, 32)
    Returns:
        A list of per image detections, where each is a dictionary with the following structure:
        {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 160, 160),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
        }
    """

    num_classes = kwargs["classes"]
    strides = kwargs["anchors"]["strides"][::-1]
    image_dims = tuple(kwargs["input_shape"])
    reg_max = kwargs["anchors"]["regression_length"]
    raw_boxes = endnodes[:7:3]
    scores = [np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes)) for s in endnodes[1:8:3]]
    scores = np.concatenate(scores, axis=1)
    outputs = []

    decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, reg_max)

    score_thres = kwargs["score_threshold"]
    iou_thres = kwargs["nms_iou_thresh"]
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape

    # add objectness=1 for working with yolov5_nms
    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1))
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)

    coeffs = [np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]]
    coeffs = np.concatenate(coeffs, axis=1)

    # re-arrange predictions for yolov5_nms
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)
    nms_res = non_max_suppression(predictions, conf_thres=score_thres, iou_thres=iou_thres, multi_label=True)

    outputs = []

    for b in range(batch_size):
        protos = proto_data[b]
        protos = protos.astype(np.float32, copy=False)
        masks_in = nms_res[b]["mask"].astype(np.float32, copy=False)
        masks = process_mask_optimized(protos, masks_in, nms_res[b]["detection_boxes"], image_dims)
        output = {}
        output["detection_boxes"] = np.array(nms_res[b]["detection_boxes"]) / np.tile(image_dims, 2)
        output["mask"] = masks
        output["detection_scores"] = np.array(nms_res[b]["detection_scores"])
        output["detection_classes"] = np.array(nms_res[b]["detection_classes"]).astype(int)
        outputs.append(output)

    return outputs


def inference_result_handler(frame, detections, config_data, model_type, labels, tracker=None, nms_postprocess_enabled=False):
    """
    This function performs post-processing on the raw model output to extract
    detection results (bounding boxes, masks, classes, scores), applies tracking
    using BYTETracker, and renders the visualized results (boxes, masks, IDs)
    on top of the original input frame.

    Args:
        frame: The original input image or video frame (as a NumPy array).
        detections: The extracted detections or raw output tensors from the model inference.
        tracker: An instance of BYTETracker used for object tracking across frames.
        config_data: A dictionary containing model-specific configuration parameters.
        model_type: A string identifier for the model architecture (e.g., "v5", "v8", "fast").

    Returns:
        np.ndarray: The frame with visualized detection, segmentation, and tracking overlays.
    """
    if nms_postprocess_enabled:
        if not detections:
            return frame

        if tracker:
            detections = track_detections(detections, tracker)
        return render_results(frame, detections)

    else:
        infer_results = detections
        decoded_detections = decode_and_postprocess(infer_results, config_data, model_type)
        return draw_detections_no_nms(decoded_detections, np.expand_dims(np.array(frame), axis=0), config_data, labels, model_type, tracker=tracker)

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
        # Step 1: Get the shape of the padded box
        x1_p, y1_p, x2_p, y2_p = box_on_padded_image
        h_p = y2_p - y1_p
        w_p = x2_p - x1_p

        # Step 2: Reshape the mask to original (padded) box shape
        try:
            mask_2d = mask_1d.reshape((h_p, w_p))
        except ValueError:
            closest_shape = find_shape_closest_to_target(mask_1d.size, h_p, w_p)
            if not closest_shape:
                return None

            h, w = closest_shape
            mask_2d = mask_1d.reshape((h, w))

        # Step 3: Get new shape after unpadding
        x1_u, y1_u, x2_u, y2_u = box_on_input_image
        h_u = y2_u - y1_u
        w_u = x2_u - x1_u

        # Step 4: Resize the mask to the unpadded box shape
        resized_mask = cv2.resize(mask_2d.astype(np.uint8), (w_u, h_u), interpolation=cv2.INTER_NEAREST)

    except Exception:
        return None

    return resized_mask


def track_detections(detections: list, tracker: BYTETracker) -> list:
    """
    Perform tracking on the detections.

    Args:
        detections (list): List of detection dictionaries.
        tracker (BYTETracker): ByteTrack tracker instance.

    Returns:
        list: List of tracked objects (dictionaries with 'label', 'score', 'box', 'mask', and 'track_id').
    """
    dets_for_tracker = []
    for det in detections:
        x, y, w, h = det['box']
        dets_for_tracker.append([x, y, x + w, y + h, det['score']])

    if not dets_for_tracker:
        return []

    online_targets = tracker.update(np.array(dets_for_tracker))
    tracked_detections = []

    for track in online_targets:
        x1, y1, x2, y2 = track.tlbr
        xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
        
        # Use the format for boxes when matching
        det_boxes = [[d['box'][0], d['box'][1], d['box'][0]+d['box'][2], d['box'][1]+d['box'][3]] for d in detections]
        masks = [d['mask'] for d in detections]
        best_idx = find_best_matching_mask_index(track.tlbr, det_boxes, masks)
        
        if best_idx is not None:
            tracked_detections.append({
                'label': detections[best_idx]['label'],
                'score': float(track.score),
                'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                'mask': detections[best_idx]['mask'],
                'track_id': track.track_id
            })
    
    return tracked_detections


def mask_to_polygons(mask):
    """
    Convert a binary mask to a list of flattened polygons.

    Args:
        mask (np.ndarray): 2D binary mask.

    Returns:
        list: List of polygons (as flattened arrays).
        bool: True if the mask has holes.
    """
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    contours = res[-2]
    polygons = [c.flatten() + 0.5 for c in contours if len(c) >= 6]
    return polygons, has_holes


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


def decode_and_postprocess(raw_detections, config_data, arch_key):
    """
    Dispatches post-processing for segmentation based on model architecture and JSON config.

    Args:
        raw_detections (Dict): Raw output from model inference.
        config_data (Dict): Loaded JSON config containing layer structure and post-processing metadata.
        arch_key (str): One of ['v5', 'v8', 'fast'] to select the architecture.

    Returns:
        Dict: Post-processed detection results including masks, boxes, scores, and classes.
    """
    arch_cfg = config_data[arch_key]
    layers = arch_cfg["layers"]
    mask_channels = arch_cfg["mask_channels"]
    raw_detections_keys = list(raw_detections.keys())
    layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}

    def resolve_shape(layer):
        b, h, w, c_tag = layer
        if isinstance(c_tag, str):
            if c_tag == "mask_channels":
                c = mask_channels
            elif c_tag == "detection_channels":
                c = (arch_cfg['classes'] + 4 + 1 + mask_channels) * len(arch_cfg['anchors']['strides'])
            elif c_tag == "detection_output_channels":
                if arch_key == 'v5':
                    c = (arch_cfg["classes"] + 4 + 1 + mask_channels) * len(arch_cfg['anchors']['strides'])
                else:
                    c = (arch_cfg['anchors']['regression_length'] + 1) * 4
            elif c_tag == "classes":
                c = arch_cfg["classes"]
            else:
                raise ValueError(f"Unsupported channel tag: {c_tag}")
        else:
            c = c_tag
        return (b, h, w, c)

    # Build endnodes based on resolved layer shapes
    endnodes = [raw_detections[layer_from_shape[resolve_shape(layer)]] for layer in layers]

    if arch_key == "v5":
        return yolov5_seg_postprocess(endnodes, **arch_cfg)[0]

    elif arch_key in ["v8", "fast"]:
        return yolov8_seg_postprocess(endnodes, **arch_cfg)[0]

    else:
        raise ValueError(f"Unsupported architecture key: {arch_key}")


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

    Example
    -------
    >>> find_shape_closest_to_target(30, 6, 6)
    (5, 6)

    Explanation:
        6×6 = 36 → not valid.
        5×6 = 30 → valid and closest to (6, 6).
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


def process_mask_optimized(protos, masks_in, bboxes, shape, upsample=True, downsample=False):
    mh, mw, c = protos.shape
    ih, iw = shape

    # Matrix multiplication (N, 32) x (32, 160*160) -> (N, 160*160)
    protos_flat = protos.reshape(-1, c).T  # (32, H*W)
    masks = masks_in @ protos_flat  # (N, H*W)
    masks = expit(masks).reshape(-1, mh, mw)  # (N, H, W)

    # Downsample-based crop
    bboxes = bboxes.copy()
    if downsample:
        bboxes[:, [0, 2]] *= mw / iw
        bboxes[:, [1, 3]] *= mh / ih
        masks = crop_mask_roi_vectorized(masks, bboxes)

    # Resize
    if upsample:
        masks = fast_resize_masks(masks, (ih, iw))

    # Final crop if not downsample
    if not downsample:
        masks = crop_mask_roi_vectorized(masks, bboxes)

    return masks


def crop_mask_roi_vectorized(masks, boxes):
    """
    Vectorized cropping of masks with zero-padding outside boxes.
    Args:
        masks: (N, H, W)
        boxes: (N, 4) in pixels
    Returns:
        (N, H, W) cropped masks
    """
    N, H, W = masks.shape
    output = np.zeros_like(masks)

    boxes = np.round(boxes).astype(int)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

    for i in range(N):
        x1, y1, x2, y2 = boxes[i]
        output[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]

    return output


def fast_resize_masks(masks, out_shape):
    """
    Resize N masks from (H, W) to out_shape (ih, iw)
    using OpenCV. Input: (N, H, W), Output: (N, ih, iw)
    """
    ih, iw = out_shape
    resized = np.empty((masks.shape[0], ih, iw), dtype=np.float32)
    for i in range(masks.shape[0]):
        resized[i] = cv2.resize(masks[i], (iw, ih), interpolation=cv2.INTER_LINEAR)
    return resized


def draw_single_detection(img_out, box, mask, score, labels, color, config_data, original_size, input_size, pad, skip_boxes, track=False):

    original_h, original_w = original_size
    input_h, input_w = input_size
    pad_h, pad_w = pad

    overlay = np.zeros((original_h, original_w, 3), dtype=np.uint8)

    # === Prepare mask ===
    mask = mask[pad_h:input_h - pad_h, pad_w:input_w - pad_w]
    if mask.shape != (original_h, original_w):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    alpha = config_data["visualization_params"]["mask_alpha"]
    mask_255 = (mask * 255).astype(np.uint8)
    overlay = draw_overlay_mask(overlay, mask_255, color=color, opacity=alpha)

    if not skip_boxes:
        # Inline box drawing on img_out (main image)
        thickness = calculate_optimal_thickness(img_out.shape[:2])
        text_scale = calculate_optimal_text_scale(img_out.shape[:2])
        xmin, ymin, xmax, ymax = map(int, box)
        draw_rectangle(img_out, [xmin, ymin, xmax - xmin, ymax - ymin], color, thickness)
        top_text = f"{labels[0]}: {score * 100.0:.1f}%"
        draw_text(img_out, top_text, (xmin, ymin), background_color=color, text_scale=text_scale)
        if track:
            draw_text(img_out, labels[1], (xmax - 50, ymax), background_color=color, text_scale=text_scale)

    return overlay


def draw_detections_no_nms(detections, img, config_data, labels, arch, tracker=None):
    visualization_params = config_data["visualization_params"]
    input_h, input_w = config_data[arch]["input_shape"]
    img_out = img[0]
    original_h, original_w = img_out.shape[:2]

    # --- Compute scale and padding used in letterbox ---
    scale_ratio = min(input_w / original_w, input_h / original_h)
    resized_w = int(round(original_w * scale_ratio))
    resized_h = int(round(original_h * scale_ratio))
    pad_w = (input_w - resized_w) // 2
    pad_h = (input_h - resized_h) // 2

    # --- Prepare detection data ---
    boxes = detections["detection_boxes"].copy()
    masks = detections["mask"] > visualization_params["mask_thresh"]
    scores = detections["detection_scores"]
    classes = detections["detection_classes"]
    classes = np.array(classes, dtype=int)

    skip_boxes = config_data[arch].get("meta_arch", "") == "yolov8_seg_postprocess" and config_data[arch].get("classes", "") == 1

    keep = scores > visualization_params["score_thres"]
    boxes, masks, scores, classes = boxes[keep], masks[keep], scores[keep], classes[keep]

    max_draw = min(visualization_params["max_boxes_to_draw"], len(boxes))
    boxes, masks, scores, classes = boxes[:max_draw], masks[:max_draw], scores[:max_draw], classes[:max_draw]

    # === Decode boxes back to original image space ===
    def decode_boxes(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * input_w - pad_w) / scale_ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * input_h - pad_h) / scale_ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)
        return boxes

    original_boxes = decode_boxes(boxes)

    # === Prepare for parallel mask drawing ===
    input_size = (input_h, input_w)
    pad = (pad_h, pad_w)
    original_size = (original_h, original_w)
    combined_overlay = np.zeros_like(img_out, dtype=np.uint8)
    futures = []
    executor = ThreadPoolExecutor(max_workers=8)

    if tracker:
        dets_for_tracker = np.concatenate([original_boxes, scores[:, None]], axis=1)
        online_targets = tracker.update(np.array(dets_for_tracker))
        for track in online_targets:
            best_idx = find_best_matching_mask_index(track.tlbr, original_boxes, masks)
            if best_idx is None:
                continue

            track_id = f"ID {track.track_id}"
            color = hex_to_rgb(get_color(track.track_id))
            args = (img_out,
                original_boxes[best_idx],
                masks[best_idx].astype(np.uint8),
                track.score,
                [labels[classes[best_idx]], track_id],
                color,
                config_data,
                original_size,
                input_size,
                pad,
                skip_boxes,
                True
            )
            futures.append(executor.submit(draw_single_detection, *args))
    else:
        for idx, box in enumerate(original_boxes):
            label = f"{labels[classes[idx]]}"
            if not skip_boxes:
                color = hex_to_rgb(get_color(classes[idx]))
            else:
                color = tuple(np.random.randint(low=0, high=255, size=3, dtype=np.uint8).tolist())

            args = (img_out,
                box,
                masks[idx].astype(np.uint8),
                scores[idx],
                [label],
                color,
                config_data,
                original_size,
                input_size,
                pad,
                skip_boxes
            )
            futures.append(executor.submit(draw_single_detection, *args))

    # === Merge overlays ===
    for future in futures:
        overlay = future.result()
        combined_overlay = cv2.add(combined_overlay, overlay)

    img_out = cv2.addWeighted(img_out, 1.0, combined_overlay, 1.0, 0)
    return img_out


class ModelInference(HailoInfer):
    def __init__(self, hef_path: str, batch_size: int = 1, labels: list = None, config_data: dict = None):
        super().__init__(hef_path, batch_size)
        self.labels = labels
        self.config_data = config_data

    def _inference_callback(self, completion_info, bindings_list: list, input_batch: list, output_queue: queue.Queue) -> None:
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                
                if self.is_nms_postprocess_enabled():
                    image = input_batch[i]
                    infer_results = result if isinstance(result, list) else [result]
                    
                    visualization_params = self.config_data["visualization_params"]
                    score_threshold = visualization_params.get("score_thres", 0.2)
                    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

                    # Values used for scaling coords and removing padding
                    img_height, img_width = image.shape[:2]
                    size = max(img_height, img_width)
                    padding_length = int(abs(img_height - img_width) / 2)

                    detections = []
                    counter = 0

                    for det in infer_results:
                        if det.score < score_threshold or counter >= max_boxes:
                            break

                        box_on_input_image, box_on_padded_image = convert_box_from_normalized([det.x_min, det.y_min, det.x_max, det.y_max], size, padding_length, img_height, img_width)
                        mask = resize_mask_to_unpadded_box(det.mask, box_on_input_image, box_on_padded_image)

                        if mask is not None:
                            xmin, ymin, xmax, ymax = box_on_input_image
                            detections.append({
                                'label': self.labels[det.class_id] if self.labels else str(det.class_id),
                                'score': float(det.score),
                                'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                                'mask': mask,
                                'class_id': det.class_id
                            })
                            counter += 1
                    output_queue.put((input_batch[i], detections))
                else:
                    output_queue.put((input_batch[i], result))

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


def run_inference_pipeline(
    net,
    labels,
    model_type,
    input_data: VideoInput,
    visualizer: VideoVisualizer,
    enable_tracking=False,
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    config_data = CONFIG_DATA
    labels = get_labels(labels)

    tracker = None

    if enable_tracking:
        # Load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    model_inference = ModelInference(
        net,
        input_data.batch_size,
        labels=labels,
        config_data=config_data)

    post_process_callback_fn = partial(
        inference_result_handler,
        tracker=tracker,
        config_data=config_data,
        model_type=model_type,
        labels=labels,
        nms_postprocess_enabled=model_inference.is_nms_postprocess_enabled()
    )

    height, width, _ = model_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=input_data.preprocess,
        args=(input_queue, width, height),
        name="preprocess-thread",
    )

    infer_thread = threading.Thread(
        target=model_inference.infer,
        args=(input_queue, output_queue, input_data.stop_event),
        name="infer-thread",
    )

    preprocess_thread.start()
    infer_thread.start()

    try:
        visualizer.visualize(
            output_queue,
            post_process_callback_fn,
            is_capture=input_data.has_capture
        )
    finally:
        input_data.stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    logger.info(visualizer.frame_rate_summary())

    logger.info("Processing completed successfully.")

    if visualizer.save_output or input_data.has_images:
        logger.info(f"Saved outputs to '{visualizer.output_dir}'.")


def main(**kwargs) -> None:
    """
    Main entry point for the instance segmentation application.

    Args:
        **kwargs: Programmatic arguments to override defaults.
    """
    options = DEFAULT_OPTIONS.copy()
    options.update(kwargs)
    args = SimpleNamespace(**options)
    
    logging.basicConfig(level=logging.INFO)
    args.hef_path = resolve_hef_path(args.hef_path, APP_NAME)

    stop_event = threading.Event()
    
    input_data = VideoInput(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        video_unpaced=args.video_unpaced,
        stop_event=stop_event,
    )

    visualizer = VideoVisualizer(
        output_dir=args.output_dir,
        save_output=args.save_output,
        source_fps=input_data.source_fps,
        stop_event=stop_event,
    )

    run_inference_pipeline(
        net=args.hef_path,
        labels=args.labels,
        model_type=args.model_type,
        input_data=input_data,
        visualizer=visualizer,
        enable_tracking=args.track,
    )


if __name__ == "__main__":
    main()
