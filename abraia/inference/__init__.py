"""Public inference API with lazy model imports.

Importing :mod:`abraia.inference` should not initialize every model backend.
Each public class is loaded only when it is first accessed.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "Model": (".models.detection", "Model"),
    "ResNetClassifier": (".models.classification", "ResNetClassifier"),
    "GroundingDINOModel": (".models.grounding_dino", "GroundingDINOModel"),
    "Tracker": (".tracker", "Tracker"),
    "FaceRecognizer": (".models.faces", "FaceRecognizer"),
    "FaceAttribute": (".models.faces", "FaceAttribute"),
    "PlateDetector": (".models.plates", "PlateDetector"),
    "PlateRecognizer": (".models.plates", "PlateRecognizer"),
    "TextSystem": (".models.ocr", "TextSystem"),
    "ImageSearch": (".search", "ImageSearch"),
    "SAM": (".models.sam", "SAM"),
    "InteractiveSAM": (".models.sam", "InteractiveSAM"),
    "Clip": (".models.clip", "Clip"),
    "InferenceService": (".service", "InferenceService"),
    "ModelSession": (".service", "ModelSession"),
    "InferenceModel": (".contracts", "InferenceModel"),
    "create_model": (".registry", "create_model"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Model",
    "ResNetClassifier",
    "GroundingDINOModel",
    "Tracker",
    "FaceRecognizer",
    "FaceAttribute",
    "PlateDetector",
    "PlateRecognizer",
    "TextSystem",
    "ImageSearch",
    "SAM",
    "InteractiveSAM",
    "Clip",
    "InferenceService",
    "ModelSession",
    "InferenceModel",
    "create_model",
]
