"""Dependency-light source-type detection shared by SDK consumers."""

import os
from pathlib import Path
from typing import Any, Optional


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


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


__all__ = ["IMAGE_SUFFIXES", "VIDEO_SUFFIXES", "infer_source_type"]
