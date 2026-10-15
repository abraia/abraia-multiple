"""Validation-based confidence threshold analysis for detector models."""

import numpy as np


def detection_threshold_analysis(model_trainer, *, is_cancelled=None):
    """Sweep validation confidence and recommend the macro-F1 peak."""
    model = model_trainer.model.model
    callback = None
    if is_cancelled:
        def callback(validator):
            if is_cancelled():
                raise RuntimeError("Training canceled")

        model.add_callback("on_val_batch_end", callback)
    try:
        result = model.val(
            split="val",
            conf=0.001,
            plots=False,
            verbose=False,
        )
    finally:
        if callback:
            callbacks = model.callbacks.get("on_val_batch_end", [])
            try:
                callbacks.remove(callback)
            except ValueError:
                pass

    curves = result.box.curves_results
    if len(curves) < 4:
        return None
    confidence = np.asarray(curves[1][0], dtype=float).reshape(-1)
    f1 = _mean_curve(curves[1][1])
    precision = _mean_curve(curves[2][1])
    recall = _mean_curve(curves[3][1])
    if not len(confidence) or not len(f1) or not np.isfinite(f1).any():
        return None

    size = min(101, len(confidence))
    sample_indices = np.linspace(0, len(confidence) - 1, size).round().astype(int)
    recommended_index = int(np.nanargmax(f1))
    sample_indices = np.unique(np.append(sample_indices, recommended_index))
    return {
        "split": "validation",
        "recommended_threshold": float(confidence[recommended_index]),
        "recommended_f1": float(f1[recommended_index]),
        "confidence": confidence[sample_indices].tolist(),
        "precision": precision[sample_indices].tolist(),
        "recall": recall[sample_indices].tolist(),
        "f1": f1[sample_indices].tolist(),
    }


def _mean_curve(values):
    curve = np.asarray(values, dtype=float)
    if curve.ndim == 1:
        return curve
    return np.nanmean(curve, axis=0)


__all__ = ["detection_threshold_analysis"]
