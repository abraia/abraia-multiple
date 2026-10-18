"""Public inference API."""

from .registry import create_model as _create_model, register_model_factory


def load_model(config):
    """Load a configured model using its kind, task, and model URI."""
    return _create_model(config)


__all__ = ["load_model", "register_model_factory"]
