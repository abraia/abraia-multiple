"""Bounding-box overlap and suppression helpers."""

import numpy as np


def iou(box1, box2):
    """Calculate intersection-over-union for two ``xywh`` boxes."""
    top_left1 = [box1[0], box1[1]]
    size1 = [box1[2], box1[3]]
    bottom_right1 = [box1[0] + box1[2], box1[1] + box1[3]]
    top_left2 = [box2[0], box2[1]]
    size2 = [box2[2], box2[3]]
    bottom_right2 = [box2[0] + box2[2], box2[1] + box2[3]]
    intersection = np.prod(
        np.maximum(np.minimum(bottom_right1, bottom_right2) - np.maximum(top_left1, top_left2), 0)
    )
    union = np.prod(size1) + np.prod(size2) - intersection
    return intersection / union


def nms(dets, thresh):
    """Run vectorized NMS on ``[x1, y1, x2, y2, score]`` detections."""
    if dets.shape[0] == 0:
        return np.array([], dtype=np.int64)
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        index = order[0]
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = width * height
        ratios = overlap / (areas[index] + areas[order[1:]] - overlap)
        order = order[np.where(ratios <= thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


def non_maximum_suppression(objects, iou_threshold):
    """Apply NMS to result dictionaries containing ``box`` and ``score``."""
    detections = []
    for obj in objects:
        x, y, width, height = obj["box"]
        detections.append([x, y, x + width, y + height, obj["score"]])
    if not detections:
        return []
    indices = nms(np.array(detections), iou_threshold)
    return [objects[index] for index in indices]


__all__ = ["iou", "nms", "non_maximum_suppression"]
