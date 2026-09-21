"""Dependency-light source-type detection shared by SDK consumers."""

import os
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class ImageSource(Protocol):
    """Minimal image-loading surface shared by local and remote clients."""

    def load_image(self, path: str):
        """Load an image or image cube from ``path``."""
        ...

    def load_metadata(self, path: str):
        """Load source metadata when available."""
        ...


@runtime_checkable
class PreviewSource(ImageSource, Protocol):
    """Image source that can produce a display-sized image preview."""

    def load_image_preview(self, path: str, size=144, bands=(0, 1, 2)):
        """Return ``(preview, metadata)`` for a source path."""
        ...


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
    if lowered.endswith(IMAGE_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return image_type
    if lowered.endswith(VIDEO_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return "video"
    if value and Path(value).is_dir():
        return image_type
    return fallback


__all__ = [
    "IMAGE_SUFFIXES",
    "SOURCE_TYPE_ALIASES",
    "VIDEO_SUFFIXES",
    "ImageSource",
    "PreviewSource",
    "infer_source_type",
    "normalize_source_type",
]
