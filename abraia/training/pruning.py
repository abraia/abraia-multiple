"""RGB training-dataset quality checks."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_DUPLICATE_THRESHOLD = 0.995
DEFAULT_BLUR_THRESHOLD = 0.0015
DEFAULT_DARK_THRESHOLD = 0.15
DEFAULT_BRIGHT_THRESHOLD = 0.85
_FEATURE_SIZE = 24
_PHASH_SIZE = 32
_PHASH_LOW_FREQUENCIES = 8
_DCT_BASIS = np.cos(
    np.pi
    / _PHASH_SIZE
    * (np.arange(_PHASH_SIZE)[:, np.newaxis] + 0.5)
    * np.arange(_PHASH_SIZE)[np.newaxis, :]
).T
_DCT_BASIS[0] *= 1 / np.sqrt(2)
_DCT_BASIS *= np.sqrt(2 / _PHASH_SIZE)

_RGB_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/bmp", "image/webp"})
_RGB_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


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


def analyze_rgb_images(
    images,
    image_loader,
    *,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    dark_threshold=DEFAULT_DARK_THRESHOLD,
    bright_threshold=DEFAULT_BRIGHT_THRESHOLD,
    progress_callback=None,
    is_cancelled=None,
):
    """Find RGB images that are candidates for dataset removal."""
    records = list(images or [])
    if any(not _is_rgb_record(image) for image in records):
        raise ValueError("Dataset pruning is available only for RGB image datasets")

    is_cancelled = is_cancelled or (lambda: False)
    analyses = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if is_cancelled():
            raise RuntimeError("Operation canceled")
        path = record.get("path") or record.get("name") or ""
        name = record.get("name") or path
        if progress_callback:
            progress_callback(index - 1, max(1, total), name, False)
        try:
            image = _prepare_rgb(image_loader(path))
            feature, gray, perceptual_hash, texture = _features(image)
            analyses.append(
                {
                    "record": record,
                    "path": path,
                    "name": name,
                    "feature": feature,
                    "gray": gray,
                    "perceptual_hash": perceptual_hash,
                    "texture": texture,
                    "brightness": float(np.mean(gray)),
                    "sharpness": _sharpness(gray),
                    "reasons": [],
                    "metrics": {},
                }
            )
        except Exception as error:
            analyses.append(
                {
                    "record": record,
                    "path": path,
                    "name": name,
                    "feature": None,
                    "gray": None,
                    "perceptual_hash": None,
                    "texture": None,
                    "brightness": None,
                    "sharpness": None,
                    "reasons": ["unreadable"],
                    "metrics": {"error": str(error)},
                }
            )
        if progress_callback:
            progress_callback(index, max(1, total), name, True)

    readable = [item for item in analyses if item["feature"] is not None]
    _mark_similar(readable, similarity_threshold, duplicate_threshold)
    _mark_outliers(readable)
    for item in readable:
        if item["sharpness"] < blur_threshold:
            item["reasons"].append("blurry")
        if item["brightness"] < dark_threshold:
            item["reasons"].append("dark")
        elif item["brightness"] > bright_threshold:
            item["reasons"].append("bright")

    return [
        {
            "path": item["path"],
            "name": item["name"],
            "reasons": tuple(dict.fromkeys(item["reasons"])),
            "metrics": dict(item["metrics"]),
        }
        for item in analyses
        if item["reasons"]
    ]


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


def _features(image):
    thumbnail = _resize(image, _FEATURE_SIZE)
    gray = (
        thumbnail[..., 0] * 0.299
        + thumbnail[..., 1] * 0.587
        + thumbnail[..., 2] * 0.114
    )
    texture = float(np.std(gray))
    perceptual_hash = _perceptual_hash(gray)
    feature = thumbnail.reshape(-1).astype(np.float32)
    return feature, gray.astype(np.float32), perceptual_hash, texture


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


def _similarity(first, second):
    if first["texture"] <= 0.01 or second["texture"] <= 0.01:
        brightness = 1.0 - abs(first["brightness"] - second["brightness"])
        return max(0.0, float(brightness))
    hash_similarity = 1.0 - np.mean(
        np.logical_xor(first["perceptual_hash"], second["perceptual_hash"])
    )
    correlation = _translation_correlation(first["gray"], second["gray"])
    return float(0.15 * hash_similarity + 0.85 * correlation)


def _mark_similar(items, similarity_threshold, duplicate_threshold):
    for index, item in enumerate(items):
        best_score = 0.0
        best_index = None
        for previous_index in range(index):
            score = _similarity(item, items[previous_index])
            if score > best_score:
                best_score = score
                best_index = previous_index
        if best_index is not None and best_score >= similarity_threshold:
            pixel_similarity = 1.0 - np.mean(
                np.abs(item["feature"] - items[best_index]["feature"])
            )
            reason = "duplicate" if pixel_similarity >= duplicate_threshold else "highly similar"
            item["reasons"].append(reason)
            item["metrics"]["similarity"] = round(best_score, 4)
            item["metrics"]["similar_to"] = items[best_index]["name"]


def _mark_outliers(items):
    if len(items) < 4:
        return
    features = np.stack([item["feature"] for item in items])
    center = np.median(features, axis=0)
    distances = np.mean(np.abs(features - center), axis=1)
    threshold = max(0.35, float(np.mean(distances)) + 1.5 * float(np.std(distances)))
    for item, distance in zip(items, distances):
        score = float(distance)
        item["metrics"]["outlier_score"] = round(score, 4)
        if score > threshold:
            item["reasons"].append("outlier")


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


__all__ = [
    "DEFAULT_BLUR_THRESHOLD",
    "DEFAULT_BRIGHT_THRESHOLD",
    "DEFAULT_DARK_THRESHOLD",
    "DEFAULT_DUPLICATE_THRESHOLD",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "analyze_rgb_images",
    "is_rgb_dataset",
]
