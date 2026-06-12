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


# def mask_to_polygons(mask):
#     """
#     Convert a binary mask to a list of flattened polygons.

#     Args:
#         mask (np.ndarray): 2D binary mask.

#     Returns:
#         list: List of polygons (as flattened arrays).
#         bool: True if the mask has holes.
#     """
#     mask = np.ascontiguousarray(mask)
#     res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
#     hierarchy = res[-1]
#     if hierarchy is None:
#         return [], False
#     has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
#     contours = res[-2]
#     polygons = [c.flatten() + 0.5 for c in contours if len(c) >= 6]
#     return polygons, has_holes


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


def nms(dets, thresh):
    """
    Pure Python NMS implementation using NumPy.
    dets: (N, 5) - [x1, y1, x2, y2, score]
    thresh: IoU threshold
    """
    if dets.shape[0] == 0:
        return np.array([], dtype=np.int64)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int64)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return np.where(suppressed == 0)[0]


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Computes softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def count_objects(results):
    counts = {}
    for result in results:
        label= result['label']
        counts[label] = counts.get(label, 0) + 1
    objects = [{'label': label, 'count': counts[label]} for label in counts.keys()]
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
    """Compute pairwise similarity scores between two vectors."""
    return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


def search_vector(vector, index, max_results=1):
    distances = [cosine_similarity(vector, row['vector']) for row in index]
    idxs = np.argsort(distances)[-max_results:][::-1]
    return idxs, [distances[idx] for idx in idxs]


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


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
        box_on_input_image (list): [xmin, ymin, xmax, ymax] in original input image.
        box_on_padded_image (list): [xmin, ymin, xmax, ymax] in padded model output image.

    Returns:
        np.ndarray: Resized 2D mask for the unpadded box size.
    """
    x1_p, y1_p, x2_p, y2_p = box_on_padded_image
    w_p, h_p = x2_p - x1_p, y2_p - y1_p
    mask_2d = mask_1d.reshape((h_p, w_p))
    # try:
    #     mask_2d = mask_1d.reshape((h_p, w_p))
    # except ValueError:
    #     closest_shape = find_shape_closest_to_target(mask_1d.size, h_p, w_p)
    #     if not closest_shape:
    #         return None
    #     h, w = closest_shape
    #     mask_2d = mask_1d.reshape((h, w))
    x1_u, y1_u, x2_u, y2_u = box_on_input_image
    resized_mask = cv2.resize(mask_2d.astype(np.uint8), (x2_u - x1_u, y2_u - y1_u), interpolation=cv2.INTER_NEAREST)
    return resized_mask
