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


# def iou(box1, box2):
#     """Calculates the intersection-over-union of two boxes."""
#     tl1, wh1, br1 = [box1[0], box1[1]], [box1[2], box1[3]], [box1[0] + box1[2], box1[1] + box1[3]]
#     tl2, wh2, br2 = [box2[0], box2[1]], [box2[2], box2[3]], [box2[0] + box2[2], box2[1] + box2[3]]
#     intersection_area = np.prod(np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0))
#     union_area = np.prod(wh1) + np.prod(wh2) - intersection_area;
#     return intersection_area / union_area


# def non_maximum_suppression(objects, iou_threshold):
#     results = []
#     objects.sort(key=lambda obj: obj['score'], reverse=True)
#     while len(objects) > 0:
#         results.append(objects[0])
#         objects = [obj for obj in objects if iou(obj['box'], objects[0]['box']) < iou_threshold]
    # return results


def non_maximum_suppression(objects, iou_threshold):
    dets = []
    for obj in objects:
        s = obj['score']
        x, y, w, h = obj['box']
        dets.append([x, y, x + w, y + h, s])
    if dets:
        idxs = py_cpu_nms(np.array(dets), iou_threshold)
        return [objects[idx] for idx in idxs]
    return []


def softmax(x):
    """Computes softmax values for each sets of scores in x.
    This ensures the output sums to 1 for each image (along axis 1)."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def count_objects(results):
    counts, colors = {}, {}
    for result in results:
        label, color = result['label'], result['color']
        counts[label] = counts.get(label, 0) + 1
        colors[label] = color
    objects = [{'label': label, 'count': counts[label], 'color': colors[label]} for label in counts.keys()]
    return objects


def triplet_orientation(A, B, C):
    """Return the orientation of the triplet (A, B, C)."""
    val = (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
    return 1 if val > 0 else -1 if val < 0 else 0


def segments_intersect(A, B, C, D):
    """Check if line segments AB and CD intersect."""
    o1, o2 = triplet_orientation(A, B, C), triplet_orientation(A, B, D)
    o3, o4 = triplet_orientation(C, D, A), triplet_orientation(C, D, B)
    return (1 if o1 > 0 else -1) if (o1 != o2 and o3 != o4) else 0


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using the ray-casting algorithm."""
    x, y = point
    inside = False
    for i in range(len(polygon)):
        (x1, y1), (x2, y2) = polygon[i-1], polygon[i]
        intersect = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        if intersect:
            inside = not inside
    return inside


def crop_box(img, box):
    x, y, w, h = box
    return img[y:y+w, x:x+w]


def euclidean_distance(feat1, feat2):
    return float(np.linalg.norm(feat1 - feat2))


def cosine_similarity(feat1, feat2):
    """Compute pairwise similarity scores between two arrays of embeddings."""
    return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


def search_vector(vector, index):
    distances = [cosine_similarity(vector, row['embeddings']) for row in index]
    idx = np.argmax(distances)
    return idx, distances


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)
