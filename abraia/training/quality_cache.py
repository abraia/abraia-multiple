"""Persistent cache helpers for RGB image quality analysis."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _record_file_size(record):
    value = record.get("file_size")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    path = Path(str(record.get("path") or ""))
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _cache_identity(record, cache_namespace=None, cache_context=None):
    identity = {
        "namespace": str(cache_namespace or ""),
        "context": dict(cache_context or {}),
        "path": str(record.get("path") or record.get("name") or ""),
        "cache_key": record.get("cache_key"),
        "file_size": _record_file_size(record),
    }
    path = Path(identity["path"])
    try:
        identity["mtime_ns"] = path.stat().st_mtime_ns
    except OSError:
        identity["mtime_ns"] = None
    return identity


def _cache_path(cache_dir, record, cache_namespace=None, cache_context=None):
    identity = json.dumps(
        _cache_identity(
            record,
            cache_namespace=cache_namespace,
            cache_context=cache_context,
        ),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return Path(cache_dir) / (digest + ".npz")


def _load_cached_analysis(
    cache_dir,
    record,
    *,
    cache_namespace=None,
    cache_context=None,
):
    filename = _cache_path(
        cache_dir,
        record,
        cache_namespace=cache_namespace,
        cache_context=cache_context,
    )
    if not filename.is_file():
        return None
    try:
        with np.load(filename, allow_pickle=False) as cached:
            from .quality_metrics import ImageAnalysis

            return ImageAnalysis(
                record=record,
                path=str(cached["path"]),
                name=str(cached["name"]),
                feature=cached["feature"],
                feature_is_custom=False,
                pixel_feature=cached["pixel_feature"],
                gray=cached["gray"],
                perceptual_hash=cached["perceptual_hash"].astype(bool),
                texture=float(cached["texture"]),
                brightness=float(cached["brightness"]),
                sharpness=float(cached["sharpness"]),
                metrics=json.loads(str(cached["metrics"])),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _save_cached_analysis(
    cache_dir,
    record,
    item,
    *,
    cache_namespace=None,
    cache_context=None,
):
    filename = _cache_path(
        cache_dir,
        record,
        cache_namespace=cache_namespace,
        cache_context=cache_context,
    )
    temporary = filename.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        path=np.asarray(item.path),
        name=np.asarray(item.name),
        feature=item.feature,
        pixel_feature=item.pixel_feature,
        gray=item.gray,
        perceptual_hash=item.perceptual_hash,
        texture=item.texture,
        brightness=item.brightness,
        sharpness=item.sharpness,
        metrics=np.asarray(json.dumps(item.metrics)),
    )
    os.replace(temporary, filename)
