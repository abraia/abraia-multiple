"""Reusable mask-layer operations."""

from __future__ import annotations

import numpy as np


def colored_prediction_layers(prediction, class_names, color_for_class):
    """Return colored mask layers for integer model predictions.

    ``color_for_class`` receives each class name and returns its display RGB
    color. Keeping color resolution as a callback lets callers use their own
    palette without coupling this array operation to a UI or annotation
    format.
    """
    labels = np.asarray(prediction)
    if labels.ndim != 2 or not labels.size:
        raise ValueError("model predictions must be a non-empty 2D mask")

    layers = []
    for class_id, name in enumerate(class_names or [], start=1):
        mask = labels == class_id
        if np.any(mask):
            layers.append((mask, color_for_class(name)))
    return layers


def compare_mask_layers(reference_layers, prediction_layers, image_shape):
    """Return foreground and label agreement metrics for two mask layers."""
    if not image_shape or len(image_shape) < 2:
        raise ValueError("image_shape must contain image height and width")
    shape = (int(image_shape[0]), int(image_shape[1]))
    reference = np.zeros(shape, dtype=bool)
    prediction = np.zeros(shape, dtype=bool)
    reference_colors = np.zeros((*shape, 3), dtype=np.uint8)
    prediction_colors = np.zeros((*shape, 3), dtype=np.uint8)

    for mask, color in reference_layers or []:
        mask = np.asarray(mask) > 0
        if mask.shape != shape:
            continue
        np.logical_or(reference, mask, out=reference)
        reference_colors[mask] = tuple(int(channel) for channel in color[:3])
    for mask, color in prediction_layers or []:
        mask = np.asarray(mask) > 0
        if mask.shape != shape:
            continue
        np.logical_or(prediction, mask, out=prediction)
        prediction_colors[mask] = tuple(int(channel) for channel in color[:3])

    overlap = reference & prediction
    union = reference | prediction
    true_positive = int(np.count_nonzero(overlap))
    annotated_pixels = int(np.count_nonzero(reference))
    predicted_pixels = int(np.count_nonzero(prediction))
    union_pixels = int(np.count_nonzero(union))
    total_pixels = int(reference.size)
    label_matches = int(
        np.count_nonzero(
            overlap & np.all(reference_colors == prediction_colors, axis=-1)
        )
    )
    return {
        "annotated_pixels": annotated_pixels,
        "predicted_pixels": predicted_pixels,
        "overlap_pixels": true_positive,
        "union_pixels": union_pixels,
        "label_matches": label_matches,
        "iou": true_positive / union_pixels if union_pixels else 1.0,
        "precision": true_positive / predicted_pixels if predicted_pixels else 0.0,
        "recall": true_positive / annotated_pixels if annotated_pixels else 0.0,
        "pixel_accuracy": (
            (total_pixels - union_pixels + true_positive) / total_pixels
            if total_pixels else 1.0
        ),
        "label_accuracy": label_matches / true_positive if true_positive else 0.0,
    }
