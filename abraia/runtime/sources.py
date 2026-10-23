"""Source-type detection and frame-source construction for runtime clients."""

import os
from pathlib import Path
from typing import Any, Optional


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
SPECTRAL_SUFFIXES = (".tif", ".tiff", ".hdr", ".raw", ".img")
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
    if lowered.endswith(IMAGE_SUFFIXES + SPECTRAL_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return image_type
    if lowered.endswith(VIDEO_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return "video"
    if value and Path(value).is_dir():
        return image_type
    return fallback


_SPECTRAL_SOURCE_FACTORY = None


def register_spectral_source(factory):
    """Register the application supplied source for spectral data."""
    if not callable(factory):
        raise TypeError("Spectral source factory must be callable")
    global _SPECTRAL_SOURCE_FACTORY
    _SPECTRAL_SOURCE_FACTORY = factory


def create_source(source, video_kwargs, spectral=False):
    """Create a standard or application supplied spectral frame source."""
    if spectral:
        if _SPECTRAL_SOURCE_FACTORY is None:
            raise ValueError("Spectral pipeline support has not been registered")
        return _SPECTRAL_SOURCE_FACTORY(source, **video_kwargs)
    from .video import Video

    return Video(source, **video_kwargs)


__all__ = [
    "IMAGE_SUFFIXES",
    "SPECTRAL_SUFFIXES",
    "SOURCE_TYPE_ALIASES",
    "VIDEO_SUFFIXES",
    "create_source",
    "infer_source_type",
    "normalize_source_type",
    "register_spectral_source",
]
