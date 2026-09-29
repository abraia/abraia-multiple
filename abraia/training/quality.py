"""RGB training-dataset quality checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .quality_cache import (
    _load_cached_analysis,
    _record_file_size,
    _save_cached_analysis,
)
from .quality_metrics import (
    DEFAULT_NEAREST_NEIGHBORS,
    ImageAnalysis,
    _features,
    _image_stats,
    _mark_outliers,
    _mark_similar,
    _percentile,
    _prepare_quality_image,
    _prepare_rgb,
    _sharpness_image,
)


DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_DUPLICATE_THRESHOLD = 0.995
DEFAULT_BLUR_THRESHOLD = 0.0015
DEFAULT_DARK_THRESHOLD = 0.15
DEFAULT_BRIGHT_THRESHOLD = 0.85
_CACHE_SCHEMA_VERSION = 3

_RGB_MIME_TYPES = frozenset({
    "image/jpeg", "image/png", "image/bmp", "image/webp",
})
_RGB_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
})


def _is_rgb_record(image):
    image = image or {}
    image_type = str(image.get("type", "") or "").lower()
    if image_type:
        return image_type in _RGB_MIME_TYPES
    path = str(image.get("path") or image.get("name") or "")
    return os.path.splitext(path.split("?", 1)[0].lower())[1] in _RGB_SUFFIXES


def is_rgb_dataset(dataset):
    """Return whether a non-empty dataset contains only standard RGB images."""
    images = getattr(dataset, "images", []) or []
    return bool(images) and all(_is_rgb_record(image) for image in images)


def analyze_rgb_quality(
    images,
    image_loader,
    *,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    blur_percentile=None,
    dark_threshold=DEFAULT_DARK_THRESHOLD,
    bright_threshold=DEFAULT_BRIGHT_THRESHOLD,
    outlier_percentile=None,
    nearest_neighbors_k=DEFAULT_NEAREST_NEIGHBORS,
    outlier_mode="one",
    include_all=False,
    feature_extractor=None,
    cache_dir=None,
    cache_namespace=None,
    cache_context=None,
    candidate_mode="all",
    compute_stats=True,
    progress_callback=None,
    is_cancelled=None,
):
    """Find RGB images that are candidates for dataset removal."""
    records = list(images or [])
    if any(not _is_rgb_record(image) for image in records):
        raise ValueError("Dataset curation is available only for RGB image datasets")

    is_cancelled = is_cancelled or (lambda: False)
    if nearest_neighbors_k < 1:
        raise ValueError("nearest_neighbors_k must be at least 1")
    if outlier_mode not in {"one", "all"}:
        raise ValueError("outlier_mode must be 'one' or 'all'")
    if candidate_mode not in {"all", "hash"}:
        raise ValueError("candidate_mode must be 'all' or 'hash'")
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_context = dict(cache_context or {})
    cache_context.update({
        "schema_version": _CACHE_SCHEMA_VERSION,
        "compute_stats": bool(compute_stats),
    })
    analyses: List[ImageAnalysis] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if is_cancelled():
            raise RuntimeError("Operation canceled")
        path = record.get("path") or record.get("name") or ""
        name = record.get("name") or path
        if progress_callback:
            progress_callback(index - 1, max(1, total), name, False)
        try:
            cached = None
            if feature_extractor is None and cache_dir is not None:
                cached = _load_cached_analysis(
                    cache_dir,
                    record,
                    cache_namespace=cache_namespace,
                    cache_context=cache_context,
                )
            if cached is not None:
                cached.record = record
                analyses.append(cached)
                if progress_callback:
                    progress_callback(index, max(1, total), name, True)
                continue

            image = _prepare_rgb(image_loader(path))
            feature, pixel_feature, gray, perceptual_hash, texture = _features(
                image,
                feature_extractor=feature_extractor,
            )
            height, width = image.shape[:2]
            quality_image = _prepare_quality_image(image)
            brightness = float(np.mean(quality_image))
            sharpness = _sharpness_image(quality_image)
            exposure_quality = max(0.0, 1.0 - 2.0 * abs(brightness - 0.5))
            stats = _image_stats(image, gray) if compute_stats else {}
            stats.update({
                "brightness": brightness,
                "sharpness": sharpness,
                "width": int(record.get("width") or width),
                "height": int(record.get("height") or height),
                "resolution": int(
                    (record.get("width") or width)
                    * (record.get("height") or height)
                ),
                "file_size": _record_file_size(record),
                "quality_score": float(
                    sharpness * (0.5 + 0.5 * exposure_quality)
                ),
            })
            analyses.append(ImageAnalysis(
                record=record,
                path=path,
                name=name,
                feature=feature,
                feature_is_custom=feature_extractor is not None,
                pixel_feature=pixel_feature,
                gray=gray,
                perceptual_hash=perceptual_hash,
                texture=texture,
                brightness=brightness,
                sharpness=sharpness,
                metrics=stats,
            ))
            if feature_extractor is None and cache_dir is not None:
                _save_cached_analysis(
                    cache_dir,
                    record,
                    analyses[-1],
                    cache_namespace=cache_namespace,
                    cache_context=cache_context,
                )
        except Exception as error:
            analyses.append(ImageAnalysis(
                record=record,
                path=path,
                name=name,
                feature=None,
                feature_is_custom=False,
                pixel_feature=None,
                gray=None,
                perceptual_hash=None,
                texture=None,
                brightness=None,
                sharpness=None,
                reasons=["unreadable"],
                metrics={"error": str(error)},
            ))
        if progress_callback:
            progress_callback(index, max(1, total), name, True)

    readable = [item for item in analyses if item.feature is not None]
    _mark_similar(
        readable,
        similarity_threshold,
        duplicate_threshold,
        nearest_neighbors_k,
        candidate_mode=candidate_mode,
    )
    _mark_outliers(
        readable,
        percentile=outlier_percentile,
        nearest_neighbors_k=nearest_neighbors_k,
        mode=outlier_mode,
        candidate_mode=candidate_mode,
    )
    if blur_threshold is None:
        blur_threshold = _percentile(
            [item.sharpness for item in readable],
            blur_percentile,
        )
    for item in readable:
        if blur_threshold is not None and item.sharpness <= blur_threshold:
            item.reasons.append("blurry")
        if item.brightness < dark_threshold:
            item.reasons.append("dark")
        elif item.brightness > bright_threshold:
            item.reasons.append("bright")

    return [
        {
            "path": item.path,
            "name": item.name,
            "reasons": tuple(dict.fromkeys(item.reasons)),
            "metrics": dict(item.metrics),
        }
        for item in analyses
        if include_all or item.reasons
    ]


__all__ = [
    "DEFAULT_BLUR_THRESHOLD",
    "DEFAULT_BRIGHT_THRESHOLD",
    "DEFAULT_DARK_THRESHOLD",
    "DEFAULT_DUPLICATE_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_NEAREST_NEIGHBORS",
    "analyze_rgb_quality",
    "is_rgb_dataset",
]
