"""Geometry and numerical helpers shared by inference and runtime components."""

import cv2
import numpy as np


def normalize(img, mean, std):
    """Normalize an image using mean and standard deviation."""
    return ((img / 255 - np.array(mean)) / np.array(std)).astype(np.float32)


def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Compute softmax values along an axis."""
    values = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def iou(box1, box2):
    """Calculate intersection-over-union for two ``xywh`` boxes."""
    top_left1 = [box1[0], box1[1]]
    size1 = [box1[2], box1[3]]
    bottom_right1 = [box1[0] + box1[2], box1[1] + box1[3]]
    top_left2 = [box2[0], box2[1]]
    size2 = [box2[2], box2[3]]
    bottom_right2 = [box2[0] + box2[2], box2[1] + box2[3]]
    intersection = np.prod(
        np.maximum(
            np.minimum(bottom_right1, bottom_right2)
            - np.maximum(top_left1, top_left2),
            0,
        )
    )
    union = np.prod(size1) + np.prod(size2) - intersection
    return intersection / union if union else 0.0


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


def is_clockwise(contour):
    """Check if a contour is oriented clockwise."""
    value = 0
    for index, point in enumerate(contour):
        next_point = contour[index + 1] if index < len(contour) - 1 else contour[0]
        value += (next_point[0] - point[0]) * (next_point[1] + point[1])
    return value < 0


def get_merge_point_idx(contour1, contour2):
    """Find the closest point indices between two contours."""
    idx1, idx2 = 0, 0
    distance_min = -1
    for i, point1 in enumerate(contour1):
        for j, point2 in enumerate(contour2):
            distance = (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
            if distance_min < 0 or distance < distance_min:
                distance_min = distance
                idx1, idx2 = i, j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    """Merge two contours at given point indices."""
    contour = list(contour1[:idx1 + 1])
    contour.extend(contour2[idx2:])
    contour.extend(contour2[:idx2 + 1])
    contour.extend(contour1[idx1:])
    return np.array(contour)


def merge_with_parent(contour_parent, contour):
    """Merge a child contour into a parent contour."""
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def approx_contour(contour, approx=0.001):
    """Approximate a contour using the Douglas-Peucker algorithm."""
    epsilon = approx * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def approximate_polygon(polygon, approx=0.02):
    """Approximate a polygon using the Douglas-Peucker algorithm."""
    return approx_contour(np.array([polygon]).astype(np.int32), approx).tolist()


def triplet_orientation(point_a, point_b, point_c):
    """Return the orientation of a point triplet."""
    value = (point_c[1] - point_a[1]) * (point_b[0] - point_a[0])
    value -= (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    return 1 if value > 0 else -1 if value < 0 else 0


def segments_intersect(point_a, point_b, point_c, point_d):
    """Check if line segments AB and CD intersect."""
    orientations = (
        triplet_orientation(point_a, point_b, point_c),
        triplet_orientation(point_a, point_b, point_d),
        triplet_orientation(point_c, point_d, point_a),
        triplet_orientation(point_c, point_d, point_b),
    )
    o1, o2, o3, o4 = orientations
    return (1 if o1 > 0 else -1) if o1 != o2 and o3 != o4 else 0


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    x, y = point
    inside = False
    for index in range(len(polygon)):
        (x1, y1), (x2, y2) = polygon[index - 1], polygon[index]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        )
        if intersects:
            inside = not inside
    return inside


def polygon_to_box(polygon):
    """Calculate the bounding box from a polygon."""
    return cv2.boundingRect(np.array(polygon, dtype=np.int32))


def mask_to_polygon(mask, origin=(0, 0), approx=0.001):
    """Return the largest polygon extracted from a segmentation mask."""
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("Segmentation masks must be two-dimensional")
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    mask = np.ascontiguousarray(mask)
    contours, hierarchies = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    outer_areas = [
        cv2.contourArea(contour) if hierarchy[3] < 0 else 0.0
        for contour, hierarchy in zip(contours, hierarchies[0])
    ]
    contours = [approx_contour(contour, approx) for contour in contours]
    parent_indexes = [int(hierarchy[3]) for hierarchy in hierarchies[0]]
    parents = [
        contour if parent < 0 and len(contour) >= 3 else []
        for contour, parent in zip(contours, parent_indexes)
    ]
    for contour, parent in zip(contours, parent_indexes):
        if parent >= 0 and len(contour) >= 3 and len(parents[parent]):
            parents[parent] = merge_with_parent(parents[parent], contour)
    if max(outer_areas) <= 0:
        return []
    selected = int(np.argmax(outer_areas))
    if len(parents[selected]) < 3:
        return []
    return (parents[selected] + np.array(origin)).tolist()


def mask_to_box(mask):
    """Calculate an ``xywh`` bounding box from a segmentation mask."""
    polygon = mask_to_polygon(mask)
    if not polygon:
        return (0, 0, 0, 0)
    return cv2.boundingRect(np.array(polygon, dtype=np.int32))


__all__ = [
    "iou", "mask_to_box", "mask_to_polygon", "nms",
    "non_maximum_suppression", "normalize", "sigmoid", "softmax",
    "approx_contour", "approximate_polygon", "get_merge_point_idx",
    "is_clockwise", "merge_contours", "merge_with_parent", "point_in_polygon",
    "polygon_to_box", "segments_intersect", "triplet_orientation",
]
