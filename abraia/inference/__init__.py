"""Public inference API."""

from .registry import create_model as _create_model, register_model_factory
from .model_config import ModelDescriptor, ModelSpec, normalize_task
from .session import get_model_accelerator


def load_model(config):
    """Load a configured model using its kind, task, and model URI."""
    return _create_model(config)


__all__ = [
    "ModelDescriptor", "ModelSpec", "get_model_accelerator",
    "load_model", "normalize_task", "register_model_factory",
]
