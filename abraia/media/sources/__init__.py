"""Media input sources for files, OpenCV devices, and GenICam cameras."""

from importlib import import_module

from .types import (
    IMAGE_SUFFIXES,
    SOURCE_TYPE_ALIASES,
    VIDEO_SUFFIXES,
    infer_source_type,
    normalize_source_type,
)

_EXPORTS = {
    "CameraSource": (".opencv", "CameraSource"),
    "FrameSource": (".opencv", "FrameSource"),
    "GenICamSource": (".genicam", "GenICamSource"),
    "ImageSequenceSource": (".opencv", "ImageSequenceSource"),
    "ImageSource": (".opencv", "ImageSource"),
    "StreamSource": (".opencv", "StreamSource"),
    "VideoSource": (".opencv", "VideoSource"),
}
__all__ = [
    *_EXPORTS,
    "IMAGE_SUFFIXES",
    "SOURCE_TYPE_ALIASES",
    "VIDEO_SUFFIXES",
    "infer_source_type",
    "normalize_source_type",
]


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module, __name__), attribute)
