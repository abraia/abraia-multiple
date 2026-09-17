"""Segmentation mask conversion and comparison helpers."""

import cv2
import numpy as np

from ..geometry import approx_contour, merge_with_parent


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
    if not contours or hierarchies is None:
        return []
    contours = [approx_contour(contour, approx) for contour in contours]
    parent_indexes = [int(hierarchy[3]) for hierarchy in hierarchies[0]]
    parents = [
        contour if parent < 0 and len(contour) >= 3 else []
        for contour, parent in zip(contours, parent_indexes)
    ]
    for contour, parent in zip(contours, parent_indexes):
        if parent >= 0 and len(contour) >= 3 and len(parents[parent]):
            parents[parent] = merge_with_parent(parents[parent], contour)
    lengths = [len(contour) for contour in parents]
    if not lengths or max(lengths) == 0:
        return []
    return (parents[np.argmax(lengths)] + np.array(origin)).tolist()


def mask_to_box(mask):
    """Calculate an ``xywh`` bounding box from a segmentation mask."""
    polygon = mask_to_polygon(mask)
    if not polygon:
        return (0, 0, 0, 0)
    return cv2.boundingRect(np.array(polygon, dtype=np.int32))


def colored_prediction_layers(prediction, class_names, color_for_class):
    """Return colored mask layers for integer model predictions."""
    labels = np.asarray(prediction)
    if labels.ndim != 2 or not labels.size:
        raise ValueError("model predictions must be a non-empty 2D mask")
    layers = []
    for class_id, name in enumerate(class_names or [], start=1):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == class_id] = 1
        if mask.any():
            layers.append((mask.astype(bool), color_for_class(name)))
    return layers


def _mask_pixels(mask, shape):
    mask = np.asarray(mask) > 0
    if mask.shape != shape:
        return None
    return set(map(tuple, np.argwhere(mask)))


def compare_mask_layers(reference_layers, prediction_layers, image_shape):
    """Return foreground and label agreement metrics for two mask layers."""
    if not image_shape or len(image_shape) < 2:
        raise ValueError("image_shape must contain image height and width")
    shape = (int(image_shape[0]), int(image_shape[1]))
    reference, prediction = set(), set()
    reference_colors, prediction_colors = {}, {}
    for mask, color in reference_layers or []:
        pixels = _mask_pixels(mask, shape)
        if pixels is None:
            continue
        reference.update(pixels)
        rgb = tuple(int(channel) for channel in color[:3])
        for pixel in pixels:
            reference_colors[pixel] = rgb
    for mask, color in prediction_layers or []:
        pixels = _mask_pixels(mask, shape)
        if pixels is None:
            continue
        prediction.update(pixels)
        rgb = tuple(int(channel) for channel in color[:3])
        for pixel in pixels:
            prediction_colors[pixel] = rgb

    overlap = reference & prediction
    union = reference | prediction
    annotated_pixels, predicted_pixels = len(reference), len(prediction)
    union_pixels = len(union)
    total_pixels = shape[0] * shape[1]
    label_matches = sum(
        reference_colors[pixel] == prediction_colors[pixel] for pixel in overlap
    )
    return {
        "annotated_pixels": annotated_pixels,
        "predicted_pixels": predicted_pixels,
        "overlap_pixels": len(overlap),
        "union_pixels": union_pixels,
        "label_matches": label_matches,
        "iou": len(overlap) / union_pixels if union_pixels else 1.0,
        "precision": len(overlap) / predicted_pixels if predicted_pixels else 0.0,
        "recall": len(overlap) / annotated_pixels if annotated_pixels else 0.0,
        "pixel_accuracy": (
            (total_pixels - union_pixels + len(overlap)) / total_pixels
            if total_pixels else 1.0
        ),
        "label_accuracy": label_matches / len(overlap) if overlap else 0.0,
    }


__all__ = [
    "colored_prediction_layers",
    "compare_mask_layers",
    "mask_to_box",
    "mask_to_polygon",
]
