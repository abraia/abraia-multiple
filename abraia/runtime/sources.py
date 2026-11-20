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


def _registered_source_factory(source, source_type):
    factory = _SOURCE_FACTORIES.get(source_type)
    if factory is None and isinstance(source, str):
        registered_type = _SOURCE_SUFFIXES.get(Path(source).suffix.lower())
        factory = _SOURCE_FACTORIES.get(registered_type)
    return factory


def create_frame_source(source, source_kwargs):
    """Create an input source without coupling it to a display."""
    source_type = normalize_source_type(source_kwargs.get("source_type"))
    factory = _registered_source_factory(source, source_type)
    if factory is not None:
        return factory(source, **source_kwargs)
    if source_type == "genicam":
        if source_kwargs.get("source_options"):
            raise ValueError(
                "GenICam source options are not supported. Set source.src to "
                "a camera index or serial number, and configure the vendor "
                "GenTL Producer through GENICAM_GENTL64_PATH."
            )
        from ..media.sources.genicam import GenICamSource

        return GenICamSource(source, fps=source_kwargs.get("fps", 30))
    if source_kwargs.get("source_options"):
        raise ValueError(
            f"Source type {source_type!r} does not accept source options"
        )
    from ..media.sources.opencv import _create_input_source

    return _create_input_source(
        source,
        source_type,
        source_kwargs.get("resolution", (1920, 1080)),
        source_kwargs.get("fps", 30),
        source_kwargs.get("video_unpaced", True),
    )


def create_source(source, video_kwargs):
    """Create a compatibility video facade or a registered source."""
    source_type = normalize_source_type(video_kwargs.get("source_type"))
    if (
        _registered_source_factory(source, source_type) is not None
        or source_type == "genicam"
    ):
        return create_frame_source(source, video_kwargs)
    from .video import Video

    return Video(source, **video_kwargs)


__all__ = [
    "IMAGE_SUFFIXES",
    "SOURCE_TYPE_ALIASES",
    "VIDEO_SUFFIXES",
    "create_frame_source",
    "create_source",
    "infer_source_type",
    "normalize_source_type",
    "register_source_factory",
]
