"""Reusable image helpers, media sources, display, and output sinks."""

from importlib import import_module

_EXPORTS = {
    "CameraSource": (".sources", "CameraSource"),
    "encode_image": (".images", "encode_image"),
    "FrameSource": (".sources", "FrameSource"),
    "GenICamSource": (".sources", "GenICamSource"),
    "image_base64": (".images", "image_base64"),
    "ImageSequenceSource": (".sources", "ImageSequenceSource"),
    "ImageSource": (".sources", "ImageSource"),
    "load_image": (".images", "load_image"),
    "RTSPOutput": (".outputs", "RTSPOutput"),
    "save_image": (".images", "save_image"),
    "StreamSource": (".sources", "StreamSource"),
    "VideoSource": (".sources", "VideoSource"),
    "VideoWriter": (".outputs", "VideoWriter"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module, __name__), attribute)
