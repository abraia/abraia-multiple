"""Model catalog normalization for training clients."""

from ..tasks import normalize_task


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
        model_task = normalize_task(record.get("task"), default=default_task)
        if not model_task:
            model_task = "detection"
        record.update({
            "name": name,
            "uri": uri,
            "task": model_task,
            "kind": record.get(
                "kind",
                "resnet" if model_task == "classification" else "yolov8",
            ),
        })
        normalized.append(record)
    return normalized


__all__ = ["normalize_custom_models"]
