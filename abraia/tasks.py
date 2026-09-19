"""Canonical task names shared by pipeline, inference, and training APIs."""


PIPELINE_TASKS = (
    "classification",
    "detection",
    "segmentation",
    "pose",
    "recognition",
)
TRAINING_TASKS = ("classification", "detection", "segmentation")
DEFAULT_TRAINING_EPOCHS = {
    "classification": 30,
    "detection": 300,
    "segmentation": 300,
}
MODEL_SIZES = ("small", "medium", "large")
HAILO_TASKS = ("detection", "segmentation", "pose")

# Accepted input spellings. Callers normalize these at API boundaries and
# serialize only the canonical values above.
TASK_ALIASES = {
    "classify": "classification",
    "detect": "detection",
    "segment": "segmentation",
    "recognize": "recognition",
}

HAILO_BACKEND_TASKS = {
    "detection": "detect",
    "segmentation": "segment",
    "pose": "pose",
}
ULTRALYTICS_BACKEND_TASKS = {
    "classification": "classify",
    "detection": "detect",
    "segmentation": "segment",
}


def normalize_task(value, default=None):
    """Normalize accepted task spellings to one canonical task value."""
    if value is None:
        value = default
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def normalize_model_size(value, default="small"):
    """Normalize a supported model size and reject unknown values."""
    size = str(value or default).strip().lower()
    if size not in MODEL_SIZES:
        raise ValueError(
            f"Unsupported model size: {size}. Use {', '.join(MODEL_SIZES)}"
        )
    return size


def to_hailo_task(value):
    """Translate a canonical task to Hailo's backend task name."""
    task = normalize_task(value)
    try:
        return HAILO_BACKEND_TASKS[task]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Hailo task: {task}. "
            f"Use {', '.join(HAILO_TASKS)}"
        ) from error


def to_ultralytics_task(value):
    """Translate a canonical task to Ultralytics' backend task name."""
    task = normalize_task(value)
    try:
        return ULTRALYTICS_BACKEND_TASKS[task]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Ultralytics training task: {task}. "
            f"Use {', '.join(TRAINING_TASKS)}"
        ) from error


__all__ = [
    "HAILO_BACKEND_TASKS",
    "HAILO_TASKS",
    "DEFAULT_TRAINING_EPOCHS",
    "MODEL_SIZES",
    "PIPELINE_TASKS",
    "TASK_ALIASES",
    "TRAINING_TASKS",
    "ULTRALYTICS_BACKEND_TASKS",
    "normalize_task",
    "normalize_model_size",
    "to_hailo_task",
    "to_ultralytics_task",
]
