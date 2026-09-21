"""Normalization helpers for model evaluation payloads."""

from ..tasks import normalize_task


def metric_average(value):
    """Return a scalar metric, averaging per-class numeric values when needed."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value if isinstance(item, (int, float))]
        return sum(values) / len(values) if values else None
    return None


def normalize_model_record(model):
    """Normalize one model catalog record for application clients."""
    if isinstance(model, str):
        return {"name": model, "metrics": {}}
    if not isinstance(model, dict):
        return None
    return {
        "name": str(model.get("name") or model.get("model") or "Model"),
        "metrics": model.get("metrics") or {},
        "classes": model.get("classes") or [],
        "task": normalize_task(model.get("task")),
    }


__all__ = ["metric_average", "normalize_model_record"]
