"""Runtime source factory registration and built-in source construction."""

import os
from pathlib import Path
from typing import Any, Optional

from ..media.sources.types import (
    IMAGE_SUFFIXES,
    SOURCE_TYPE_ALIASES,
    VIDEO_SUFFIXES,
    infer_source_type as infer_media_source_type,
    normalize_source_type,
)


_SOURCE_FACTORIES = {}
_SOURCE_SUFFIXES = {}


def infer_source_type(
    source: Any,
    fallback: Optional[str] = None,
    *,
    require_exists: bool = False,
    image_type: str = "image",
) -> Optional[str]:
    """Infer a media type, including file suffixes registered by runtime."""
    value = "" if source is None else str(source).strip()
    lowered = value.lower()
    if any(lowered.endswith(suffix) for suffix in _SOURCE_SUFFIXES):
        if not require_exists or os.path.isfile(value):
            return image_type
    return infer_media_source_type(
        source,
        fallback=fallback,
        require_exists=require_exists,
        image_type=image_type,
    )


def register_source_factory(source_type, factory, *, suffixes=()):
    """Register a source constructor and optional file suffixes.

    The factory receives ``(path, **source_kwargs)`` with common runtime
    settings and optional ``source_options``. Sources yield images or mappings
    with ``frame`` and optional ``data`` keys. Data is an opaque mapping passed
    to ``FrameContext.data``. Sources implement iteration and ``close()``;
    ``frame_rate`` and ``stop()`` are optional. Display is owned by the
    pipeline and is not part of the source contract. Register extensions in
    each worker process. Re-registering a type replaces its factory and
    suffixes.
    """
    if not isinstance(source_type, str) or not source_type:
        raise ValueError("Source type must be a non-empty string")
    source_type = normalize_source_type(source_type)
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
    source_type = normalize_source_type(video_kwargs.get("source_type"))
    factory = _SOURCE_FACTORIES.get(source_type)
    if factory is None and isinstance(source, str):
        registered_type = _SOURCE_SUFFIXES.get(Path(source).suffix.lower())
        factory = _SOURCE_FACTORIES.get(registered_type)
    if factory is not None:
        return factory(source, **video_kwargs)
    if source_type == "genicam":
        from ..media.sources.genicam import GenICamSource

        return GenICamSource(
            source,
            fps=video_kwargs.get("fps", 30),
            options=video_kwargs.get("source_options"),
        )
    if video_kwargs.get("source_options"):
        raise ValueError(
            f"Source type {source_type!r} does not accept source options"
        )
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
