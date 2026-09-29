"""Image feature extraction and quality scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

DEFAULT_NEAREST_NEIGHBORS = 2
_FEATURE_SIZE = 24
_QUALITY_MAX_DIMENSION = 1024
_PHASH_SIZE = 32
_PHASH_LOW_FREQUENCIES = 8
_FAST_MAX_CANDIDATES = 256
_DCT_BASIS = np.cos(
    np.pi
    / _PHASH_SIZE
    * (np.arange(_PHASH_SIZE)[:, np.newaxis] + 0.5)
    * np.arange(_PHASH_SIZE)[np.newaxis, :]
).T
_DCT_BASIS[0] *= 1 / np.sqrt(2)
_DCT_BASIS *= np.sqrt(2 / _PHASH_SIZE)


@dataclass
class ImageAnalysis:
    """Intermediate image features and mutable quality findings."""

    record: Dict[str, Any]
    path: str
    name: str
    feature: Optional[np.ndarray]
    feature_is_custom: bool
    pixel_feature: Optional[np.ndarray]
    gray: Optional[np.ndarray]
    perceptual_hash: Optional[np.ndarray]
    texture: Optional[float]
    brightness: Optional[float]
    sharpness: Optional[float]
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _prepare_rgb(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("Expected a grayscale or RGB image")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[..., :3]
    values = array.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite pixels")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(values.max())
    if np.issubdtype(array.dtype, np.integer):
        values /= float(np.iinfo(array.dtype).max)
    elif maximum > 1.0:
        values /= 255.0 if maximum <= 255.0 else maximum
    return np.clip(values, 0.0, 1.0)


def _features(image, feature_extractor=None):
    thumbnail = _resize(image, _FEATURE_SIZE)
    gray = (
        thumbnail[..., 0] * 0.299
        + thumbnail[..., 1] * 0.587
        + thumbnail[..., 2] * 0.114
    )
    texture = float(np.std(gray))
    perceptual_hash = _perceptual_hash(gray)
    pixel_feature = thumbnail.reshape(-1).astype(np.float32)
    feature = pixel_feature
    if feature_extractor is not None:
        feature = np.asarray(feature_extractor(image), dtype=np.float32).reshape(-1)
        if not feature.size or not np.isfinite(feature).all():
            raise ValueError("feature_extractor returned an invalid feature vector")
    return feature, pixel_feature, gray.astype(np.float32), perceptual_hash, texture


def _image_stats(image, gray):
    """Return inexpensive statistics useful for quality ranking and review."""
    quantized = np.uint8(np.clip(image, 0.0, 1.0) * 255.0)
    gradients = np.concatenate((
        np.abs(np.diff(gray, axis=0)).reshape(-1),
        np.abs(np.diff(gray, axis=1)).reshape(-1),
    ))
    return {
        "contrast": float(np.std(gray)),
        "unique_colors": int(np.unique(quantized.reshape(-1, 3), axis=0).shape[0]),
        "edge_density": float(np.mean(gradients > 0.08)) if gradients.size else 0.0,
    }


def _resize(image, size):
    values = np.uint8(np.clip(image, 0.0, 1.0) * 255.0)
    return np.asarray(
        Image.fromarray(values).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def _perceptual_hash(gray):
    resized = np.asarray(
        Image.fromarray(np.uint8(np.clip(gray, 0.0, 1.0) * 255.0)).resize(
            (_PHASH_SIZE, _PHASH_SIZE), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    coefficients = _DCT_BASIS @ resized @ _DCT_BASIS.T
    low = coefficients[:_PHASH_LOW_FREQUENCIES, :_PHASH_LOW_FREQUENCIES]
    median = np.median(low.reshape(-1)[1:])
    return low > median


def _correlation(first, second):
    first = first.reshape(-1).astype(np.float32)
    second = second.reshape(-1).astype(np.float32)
    first -= np.mean(first)
    second -= np.mean(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-8:
        return None
    return float((np.dot(first, second) / denominator + 1.0) / 2.0)


def _translation_correlation(first, second, radius=2):
    height, width = first.shape
    best = 0.0
    for row_shift in range(-radius, radius + 1):
        for column_shift in range(-radius, radius + 1):
            first_row = max(0, row_shift)
            second_row = max(0, -row_shift)
            first_column = max(0, column_shift)
            second_column = max(0, -column_shift)
            overlap_height = height - abs(row_shift)
            overlap_width = width - abs(column_shift)
            if overlap_height < 4 or overlap_width < 4:
                continue
            score = _correlation(
                first[
                    first_row:first_row + overlap_height,
                    first_column:first_column + overlap_width,
                ],
                second[
                    second_row:second_row + overlap_height,
                    second_column:second_column + overlap_width,
                ],
            )
            if score is not None:
                best = max(best, score)
    return best


def _similarity(first, second, fast=False, translated=True):
    if first.feature_is_custom or second.feature_is_custom:
        return _cosine_similarity(first.feature, second.feature)
    if first.texture <= 0.01 or second.texture <= 0.01:
        brightness = 1.0 - abs(first.brightness - second.brightness)
        return max(0.0, float(brightness))
    hash_similarity = 1.0 - np.mean(
        np.logical_xor(first.perceptual_hash, second.perceptual_hash)
    )
    if translated:
        correlation = _translation_correlation(
            first.gray,
            second.gray,
            radius=1 if fast else 2,
        )
    else:
        correlation = _correlation(first.gray, second.gray) or 0.0
    return float(0.15 * hash_similarity + 0.85 * correlation)


def _cosine_similarity(first, second):
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-8:
        return 0.0
    return float((np.dot(first, second) / denominator + 1.0) / 2.0)


def _mark_similar(
    items,
    similarity_threshold,
    duplicate_threshold,
    nearest_neighbors_k=DEFAULT_NEAREST_NEIGHBORS,
    candidate_mode="all",
):
    buckets = _build_hash_buckets(items) if candidate_mode == "hash" else None
    for index, item in enumerate(items):
        candidates = []
        for previous_index in _candidate_indices(
            items, index, candidate_mode, buckets,
            max_candidates=_FAST_MAX_CANDIDATES,
        ):
            score = _similarity(
                item,
                items[previous_index],
                fast=candidate_mode == "hash",
            )
            if score < similarity_threshold:
                continue
            pixel_similarity = 1.0 - np.mean(
                np.abs(
                    item.pixel_feature
                    - items[previous_index].pixel_feature
                )
            )
            candidates.append({
                "name": items[previous_index].name,
                "similarity": round(score, 4),
                "reason": (
                    "duplicate"
                    if pixel_similarity >= duplicate_threshold
                    else "highly similar"
                ),
            })
        candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
        candidates = candidates[:nearest_neighbors_k]
        if candidates:
            item.reasons.append(
                "duplicate"
                if any(candidate["reason"] == "duplicate" for candidate in candidates)
                else "highly similar"
            )
            item.metrics["similar_neighbors"] = candidates
            item.metrics["similarity"] = candidates[0]["similarity"]
            item.metrics["similar_to"] = candidates[0]["name"]


def _mark_outliers(
    items,
    percentile=None,
    nearest_neighbors_k=DEFAULT_NEAREST_NEIGHBORS,
    mode="one",
    candidate_mode="all",
):
    # Percentile-based outlier selection is not meaningful on tiny datasets;
    # treat outliers as a population-level signal.
    if len(items) < 4:
        return
    if percentile is None:
        # Keep the original absolute mode for callers of the low-level API.
        # Curation uses percentile-based nearest-neighbor scores below.
        features = np.stack([item.feature for item in items])
        center = np.median(features, axis=0)
        scores = np.mean(np.abs(features - center), axis=1).tolist()
        threshold = max(
            0.35,
            float(np.mean(scores)) + 1.5 * float(np.std(scores)),
        )
    else:
        scores = []
        buckets = _build_hash_buckets(items) if candidate_mode == "hash" else None
        for index, item in enumerate(items):
            distances = sorted(
                1.0 - _similarity(
                    item,
                    other,
                    fast=candidate_mode == "hash",
                    translated=False,
                )
                for other_index in _candidate_indices(
                    items, index, candidate_mode, buckets,
                    max_candidates=_FAST_MAX_CANDIDATES,
                )
                for other in [items[other_index]]
            )
            neighbors = distances[:nearest_neighbors_k]
            score = (
                neighbors[0]
                if neighbors and mode == "one"
                else float(np.mean(neighbors))
                if neighbors
                else 1.0
            )
            scores.append(score)
        threshold = _percentile(scores, 1.0 - percentile)
    for item, score in zip(items, scores):
        item.metrics["outlier_score"] = round(score, 4)
        item.metrics["outlier_mode"] = mode
        if score > threshold:
            item.reasons.append("outlier")


def _build_hash_buckets(items):
    """Build four-band pHash buckets for fast candidate generation."""
    buckets = {}
    for index, item in enumerate(items):
        bits = np.asarray(item.perceptual_hash, dtype=np.uint8).reshape(-1)
        for band in range(4):
            start = band * 16
            key = (band, tuple(bits[start:start + 16].tolist()))
            buckets.setdefault(key, set()).add(index)
        coarse_key = (
            "coarse",
            int(np.clip(item.brightness * 8.0, 0, 7)),
            int(np.clip(item.texture * 8.0, 0, 7)),
        )
        buckets.setdefault(coarse_key, set()).add(index)
    return buckets


def _candidate_indices(
    items,
    index,
    candidate_mode,
    buckets,
    max_candidates=None,
):
    if candidate_mode == "all":
        return range(index)
    bits = np.asarray(items[index].perceptual_hash, dtype=np.uint8).reshape(-1)
    candidates = set()
    for band in range(4):
        start = band * 16
        key = (band, tuple(bits[start:start + 16].tolist()))
        candidates.update(candidate for candidate in buckets.get(key, ()) if candidate < index)
    coarse_key = (
        "coarse",
        int(np.clip(items[index].brightness * 8.0, 0, 7)),
        int(np.clip(items[index].texture * 8.0, 0, 7)),
    )
    candidates.update(
        candidate for candidate in buckets.get(coarse_key, ()) if candidate < index
    )
    candidates = sorted(candidates)
    if max_candidates is not None and len(candidates) > max_candidates:
        positions = np.linspace(
            0, len(candidates) - 1, max_candidates, dtype=int
        )
        candidates = [candidates[position] for position in positions]
    return candidates


def _prepare_quality_image(image):
    """Bound image size once for consistent brightness and blur metrics."""
    height, width = image.shape[:2]
    scale = min(1.0, _QUALITY_MAX_DIMENSION / max(height, width))
    if scale >= 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    values = np.uint8(np.clip(image, 0.0, 1.0) * 255.0)
    return np.asarray(
        Image.fromarray(values).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def _sharpness_image(image):
    """Return variance of the grayscale Laplacian for an RGB image."""
    gray = (
        image[..., 0] * 0.299
        + image[..., 1] * 0.587
        + image[..., 2] * 0.114
    )
    return _sharpness(gray)


def _sharpness(gray):
    if gray.size < 9:
        return 0.0
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(np.var(laplacian))


def _percentile(values, percentile):
    if not values or percentile is None:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered)) - 1))
    return ordered[index]
