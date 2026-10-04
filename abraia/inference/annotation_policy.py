"""Task-aware routing for the built-in annotation inference models."""

from dataclasses import dataclass


MOBILE_SAM_MODEL = "MobileSAM"
GROUNDING_DINO_MODEL = "Grounding DINO"
MOBILE_SAM_DISABLED_TASKS = frozenset(("classification",))


@dataclass(frozen=True)
class AnnotationModelRoute:
    """Describe the inference operation selected for an annotation model."""

    operation: str
    allowed: bool = True


def model_route(model_name, task):
    """Return the backend inference route for a model/task pair."""
    if model_name == MOBILE_SAM_MODEL:
        return AnnotationModelRoute("sam", task not in MOBILE_SAM_DISABLED_TASKS)
    if model_name == GROUNDING_DINO_MODEL:
        operation = (
            "grounding_dino_segmentation"
            if task == "segmentation"
            else "grounding_dino"
        )
        return AnnotationModelRoute(operation)
    return AnnotationModelRoute("custom")


__all__ = [
    "AnnotationModelRoute",
    "GROUNDING_DINO_MODEL",
    "MOBILE_SAM_DISABLED_TASKS",
    "MOBILE_SAM_MODEL",
    "model_route",
]
