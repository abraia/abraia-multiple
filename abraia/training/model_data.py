"""Normalize model catalog and evaluation data for training clients."""

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


def normalize_custom_models(models, project="", userid="", task=""):
    """Return project-scoped custom model records with normalized metadata."""
    default_task = normalize_task(task, default="detection")
    prefix = "/".join(
        part for part in (str(userid).strip(), str(project).strip()) if part
    )
    normalized = []
    for model in models or []:
        record = dict(model) if isinstance(model, dict) else {"name": str(model)}
        name = str(record.get("name", "") or "").strip()
        if not name:
            continue
        uri = str(record.get("uri", "") or "").strip()
        if not uri:
            uri = f"{prefix}/{name}" if prefix else name
        model_task = normalize_task(record.get("task"))
        model_kind = record.get("kind")
        name_lower = name.lower()
        if not model_task:
            if any(marker in name_lower for marker in ("_seg", "-seg")):
                model_task = "segmentation"
            elif model_kind == "resnet":
                model_task = "classification"
            else:
                model_task = default_task
        if model_kind not in ("yolov5", "yolov8", "yolo11", "rtmdet", "resnet"):
            if str(record.get("backend", "")).lower() == "rtmdet":
                model_kind = "rtmdet"
            elif model_task == "classification":
                model_kind = "resnet"
            else:
                model_kind = next(
                    (
                        candidate
                        for candidate in ("yolov5", "yolov8", "yolo11")
                        if name_lower.startswith(candidate)
                    ),
                    "yolov8",
                )
        record.update({
            "name": name,
            "uri": uri,
            "task": model_task,
            "kind": model_kind,
        })
        normalized.append(record)
    return normalized


def normalize_confusion_matrix(matrix, labels=None):
    """Return rectangular numeric rows and matching class labels."""
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    rows = []
    for row in matrix if matrix is not None else []:
        if not isinstance(row, (list, tuple)):
            continue
        values = []
        for value in row:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
        rows.append(values)
    if not rows or not any(any(value != 0 for value in row) for row in rows):
        return [], []

    columns = max(len(row) for row in rows)
    normalized_rows = [row + [0.0] * (columns - len(row)) for row in rows]
    normalized_labels = [str(label) for label in labels or []]
    while len(normalized_labels) < columns:
        normalized_labels.append(
            "background"
            if len(normalized_labels) == columns - 1
            else str(len(normalized_labels) + 1)
        )
    return normalized_rows, normalized_labels


__all__ = [
    "metric_average",
    "normalize_confusion_matrix",
    "normalize_custom_models",
    "normalize_model_record",
]
