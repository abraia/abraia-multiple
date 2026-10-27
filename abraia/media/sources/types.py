"""Media source type names and path detection."""

import os
from pathlib import Path
from typing import Any, Optional


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
SOURCE_TYPE_ALIASES = {
    "images": "image",
    "sequence": "image",
    "image_sequence": "image",
    "usb_camera": "camera",
    "rpi_camera": "camera",
}


def normalize_source_type(source_type: Any) -> str:
    """Return the canonical name for a media source type."""
    normalized = str(source_type or "").strip().lower()
    return SOURCE_TYPE_ALIASES.get(normalized, normalized)


def infer_source_type(
    source: Any,
    fallback: Optional[str] = None,
    *,
    require_exists: bool = False,
    image_type: str = "image",
) -> Optional[str]:
    """Infer a media source type from a device index, URL, path, or suffix."""
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
    "infer_source_type",
    "normalize_source_type",
]
