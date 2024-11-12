import cv2
import numpy as np


def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i in range(num):
        p1 = contour[i]
        p2 = contour[i + 1] if i < num - 1 else contour[0]
        value += (p2[0] - p1[0]) * (p2[1] + p1[1]);
    return value < 0


def get_merge_point_idx(contour1, contour2):                   
    idx1, idx2 = 0, 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2);
            if distance_min < 0 or distance < distance_min:
                distance_min = distance
                idx1, idx2 = i, j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    return np.array(contour)


def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def approx_contour(contour, approx=0.001):
    epsilon = approx * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    return contour.reshape(-1, 2)


def mask_to_polygon(mask, origin=[0, 0], approx=0.001):
    """Returns the largest bounding polygon based on the segmentation mask."""
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = [approx_contour(contour, approx) for contour in contours]
    parent_idxs = [int(hierarchy[3]) for hierarchy in hierarchies[0]]
    contours_parent = []
    for contour, parent_idx in zip(contours, parent_idxs):
        contours_parent.append(contour if parent_idx < 0 and len(contour) >= 3 else [])
    for contour, parent_idx in zip(contours, parent_idxs):
        if parent_idx >= 0 and len(contour) >= 3 and len(contours_parent[parent_idx]):
            contours_parent[parent_idx] = merge_with_parent(contours_parent[parent_idx], contour)
    lengths = [len(contour) for contour in contours_parent]
    polygon = contours_parent[np.argmax(lengths)] + np.array(origin)
    return polygon.tolist()


# def mask_to_polygon(mask, origin=[0, 0], approx=0.001):
#     """Returns the largest bounding polygon based on the segmentation mask."""
#     contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#     contours = [approx_contour(contour, approx) for contour in contours]
#     lengths = [len(contour) for contour in contours]
#     polygon = contours[np.argmax(lengths)].reshape(-1, 2)
#     polygon = polygon + np.array(origin)
#     return polygon.tolist()


def approximate_polygon(polygon, approx=0.02):
    contour = np.array([polygon]).astype(np.int32)
    contour = approx_contour(contour, approx)
    return contour.tolist()


def normalize(img, mean, std):
    img = (img / 255 - np.array(mean)) / np.array(std)
    return img.astype(np.float32)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline.

    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    """
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def non_maximum_suppression(objects, iou_threshold):
    dets = []
    for obj in objects:
        s = obj['confidence']
        x, y, w, h = obj['box']
        dets.append([x, y, x + w, y + h, s])
    if dets:
        idxs = py_cpu_nms(np.array(dets), iou_threshold)
        return [objects[idx] for idx in idxs]
    return []
