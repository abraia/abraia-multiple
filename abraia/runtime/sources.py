"""Source-type detection and frame-source construction for runtime clients."""

import os
from pathlib import Path
from typing import Any, Optional


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
SOURCE_TYPE_ALIASES = {
    "images": "image",
    "usb_camera": "camera",
    "rpi_camera": "camera",
}


def normalize_source_type(source_type: Any) -> str:
    """Return the canonical editor/runtime name for a source type."""
    normalized = str(source_type or "").strip().lower()
    return SOURCE_TYPE_ALIASES.get(normalized, normalized)


def infer_source_type(
    source: Any,
    fallback: Optional[str] = None,
    *,
    require_exists: bool = False,
    image_type: str = "image",
) -> Optional[str]:
    """Infer a source type without importing media or model libraries.

    Editors can classify paths before they exist. Runtime callers can pass
    ``require_exists=True`` to require local files and choose ``"images"``
    as their historical directory-source value.
    """
    value = "" if source is None else str(source).strip()
    lowered = value.lower()
    if value.isdigit():
        return "camera"
    if lowered.startswith(("rtsp://", "http://", "https://")):
        return "stream"
    if lowered.endswith(IMAGE_SUFFIXES + tuple(_SOURCE_SUFFIXES)):
        if not require_exists or os.path.isfile(value):
            return image_type
    if lowered.endswith(VIDEO_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return "video"
    if value and Path(value).is_dir():
        return image_type
    return fallback


_SOURCE_FACTORIES = {}
_SOURCE_SUFFIXES = {}


def register_source_factory(source_type, factory, *, suffixes=()):
    """Register a source constructor and optional file suffixes.

    The factory receives ``(path, **source_options)``. Sources yield images or
    mappings with ``frame`` and ``data`` keys. Data is an opaque mapping passed
    to ``FrameContext.data``. Sources implement iteration, ``frame_rate``,
    ``set_display_enabled(enabled)`` and ``close()``; displayed sources also
    implement ``show(frame)``. Register extensions in each worker process.
    Re-registering a type replaces its factory and suffixes.
    """
    if not isinstance(source_type, str) or not source_type:
        raise ValueError("Source type must be a non-empty string")
    if not callable(factory):
        raise TypeError("Source factory must be callable")
    for suffix, registered_type in tuple(_SOURCE_SUFFIXES.items()):
        if registered_type == source_type:
            del _SOURCE_SUFFIXES[suffix]
    _SOURCE_FACTORIES[source_type] = factory
    for suffix in suffixes:
        _SOURCE_SUFFIXES[suffix.lower()] = source_type


def create_source(source, video_kwargs):
    """Create a built-in source or a registered source by type/file suffix."""
    source_type = video_kwargs.get("source_type")
    factory = _SOURCE_FACTORIES.get(source_type)
    if factory is None and isinstance(source, str):
        registered_type = _SOURCE_SUFFIXES.get(Path(source).suffix.lower())
        factory = _SOURCE_FACTORIES.get(registered_type)
    if factory is not None:
        return factory(source, **video_kwargs)
    from .video import Video

    return Video(source, **video_kwargs)


__all__ = [
    "IMAGE_SUFFIXES",
    "SOURCE_TYPE_ALIASES",
    "VIDEO_SUFFIXES",
    "create_source",
    "infer_source_type",
    "normalize_source_type",
    "register_source_factory",
]
