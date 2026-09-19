"""Output decoding for YOLO pose-estimation models."""

import numpy as np

from .common import non_maximum_suppression, sigmoid
from .detection import _box_from_model_coordinates, _transform_for_output


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


class PoseDecoder:
    """Decode pose-estimation outputs for one model."""

    def __init__(self, classes):
        self.classes = list(classes)

    def __call__(self, outputs, size, shape, **kwargs):
        return process_pose_output(outputs, size, shape, self.classes, **kwargs)


__all__ = ["PoseDecoder", "process_pose_output"]
