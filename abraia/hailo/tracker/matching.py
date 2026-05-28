from __future__ import annotations
import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist


def bbox_ious(bboxes1, bboxes2):
    """
    Compute IoU between two sets of bounding boxes.
    bboxes1: (N, 4) - [x1, y1, x2, y2]
    bboxes2: (M, 4) - [x1, y1, x2, y2]
    """
    bboxes1 = np.ascontiguousarray(bboxes1, dtype=np.float64)
    bboxes2 = np.ascontiguousarray(bboxes2, dtype=np.float64)
    if bboxes1.ndim == 1:
        bboxes1 = bboxes1[None, :]
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2[None, :]
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=np.float64)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            x1 = max(bboxes1[i, 0], bboxes2[j, 0])
            y1 = max(bboxes1[i, 1], bboxes2[j, 1])
            x2 = min(bboxes1[i, 2], bboxes2[j, 2])
            y2 = min(bboxes1[i, 3], bboxes2[j, 3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            inter = w * h
            
            area1 = (bboxes1[i, 2] - bboxes1[i, 0]) * (bboxes1[i, 3] - bboxes1[i, 1])
            area2 = (bboxes2[j, 2] - bboxes2[j, 0]) * (bboxes2[j, 3] - bboxes2[j, 1])
            
            union = area1 + area2 - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious

from .kalman_filter import chi2inv95
import time


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def find_best_matching_mask_index(track_box, original_boxes, masks):
    """
    Finds the index of the mask whose corresponding box has the highest IoU with the given track box.

    Args:
        track_box (list or tuple): The tracking box [x_min, y_min, x_max, y_max].
        original_boxes (list): List of boxes corresponding to the masks.
        masks (list): List of masks.

    Returns:
        int or None: Index of the best matching mask, or None if no suitable match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, box in enumerate(original_boxes):
        iou = compute_iou(track_box, box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx == -1 or best_idx >= len(masks):
        return None
    return best_idx


class Matching:

    @staticmethod
    def merge_matches(m1, m2, shape):
        O,P,Q = shape
        m1 = np.asarray(m1)
        m2 = np.asarray(m2)

        M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
        M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

        mask = M1*M2
        match = mask.nonzero()
        match = list(zip(match[0], match[1]))
        unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
        unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

        return match, unmatched_O, unmatched_Q

    @staticmethod
    def _indices_to_matches(cost_matrix, indices, thresh):
        matched_cost = cost_matrix[tuple(zip(*indices))]
        matched_mask = (matched_cost <= thresh)

        matches = indices[matched_mask]
        unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
        unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

        return matches, unmatched_a, unmatched_b

    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    @staticmethod
    def ious(atlbrs, btlbrs):
        """
        Compute cost based on IoU
        :type atlbrs: list[tlbr] | np.ndarray
        :type atlbrs: list[tlbr] | np.ndarray

        :rtype ious np.ndarray
        """
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
        if ious.size == 0:
            return ious

        ious = bbox_ious(
            np.ascontiguousarray(atlbrs, dtype=np.float64),
            np.ascontiguousarray(btlbrs, dtype=np.float64)
        )

        return ious

    @staticmethod
    def iou_distance(atracks, btracks):
        """
        Compute cost based on IoU
        :type atracks: list[STrack]
        :type btracks: list[STrack]

        :rtype cost_matrix np.ndarray
        """

        if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]
        _ious = Matching.ious(atlbrs, btlbrs)
        cost_matrix = 1 - _ious

        return cost_matrix

    @staticmethod
    def v_iou_distance(atracks, btracks):
        """
        Compute cost based on IoU
        :type atracks: list[STrack]
        :type btracks: list[STrack]

        :rtype cost_matrix np.ndarray
        """

        if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
            btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
        _ious = Matching.ious(atlbrs, btlbrs)
        cost_matrix = 1 - _ious

        return cost_matrix


    @staticmethod
    def embedding_distance(tracks, detections, metric='cosine'):
        """
        :param tracks: list[STrack]
        :param detections: list[BaseTrack]
        :param metric:
        :return: cost_matrix np.ndarray
        """

        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
        if cost_matrix.size == 0:
            return cost_matrix
        det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
        #for i, track in enumerate(tracks):
            #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
        track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
        cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
        return cost_matrix


    @staticmethod
    def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
        if cost_matrix.size == 0:
            return cost_matrix
        gating_dim = 2 if only_position else 4
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position)
            cost_matrix[row, gating_distance > gating_threshold] = np.inf
        return cost_matrix


    @staticmethod
    def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
        if cost_matrix.size == 0:
            return cost_matrix
        gating_dim = 2 if only_position else 4
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position, metric='maha')
            cost_matrix[row, gating_distance > gating_threshold] = np.inf
            cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
        return cost_matrix


    @staticmethod
    def fuse_iou(cost_matrix, tracks, detections):
        if cost_matrix.size == 0:
            return cost_matrix
        reid_sim = 1 - cost_matrix
        iou_dist = Matching.iou_distance(tracks, detections)
        iou_sim = 1 - iou_dist
        fuse_sim = reid_sim * (1 + iou_sim) / 2
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        #fuse_sim = fuse_sim * (1 + det_scores) / 2
        fuse_cost = 1 - fuse_sim
        return fuse_cost


    @staticmethod
    def fuse_score(cost_matrix, detections):
        if cost_matrix.size == 0:
            return cost_matrix
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        fuse_sim = iou_sim * det_scores
        fuse_cost = 1 - fuse_sim
        return fuse_cost