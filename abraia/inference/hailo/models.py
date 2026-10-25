"""Hailo model metadata and explicit HEF resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..model_config import HAILO_TASKS, normalize_task
from ...utils.remote import (
    is_managed_model_path,
    resolve_artifact,
)


logger = logging.getLogger(__name__)

HAILO_FILE_EXTENSION = ".hef"
HAILO_TARGETS = ("hailo8l", "hailo8", "hailo10h", "hailo15h", "hailo15l")


def resolve_model_type(kind, task):
    """Derive the Hailo decoder from the model family and task."""
    task = normalize_task(task)
    if task not in HAILO_TASKS:
        raise ValueError(
            f"Unsupported Hailo task: {task}. Use {', '.join(HAILO_TASKS)}"
        )
    kind = str(kind or "").strip().lower()
    model_type = {
        "yolov5": "v5",
        "yolov8": "v8",
        "yolo11": "v8",
        "rtmdet": "rtmdet",
    }.get(kind)
    if model_type is None:
        raise ValueError(f"Unsupported Hailo model kind: {kind}")
    return "v8" if task == "pose" and model_type == "v5" else model_type


def _model_json_candidates(hef_path):
    """Return JSON sidecar candidates for a local or remote HEF path."""
    path = Path(hef_path)
    if path.is_dir():
        return (path / "abraia.json",)

    stems = [path.stem]
    for target in HAILO_TARGETS:
        suffix = f"_{target}"
        if path.stem.endswith(suffix):
            stems.append(path.stem[: -len(suffix)])
            break
    candidates = [path.with_name(f"{stem}.json") for stem in stems]
    if path.suffix.lower() == HAILO_FILE_EXTENSION:
        candidates.insert(0, path.with_suffix("") / "abraia.json")
    return tuple(candidates)


def load_hailo_model_config(hef_path):
    """Load the JSON sidecar used to configure the corresponding ONNX model."""
    from ...utils.filesystem import load_json
    from ...utils.remote import resolve_model_file

    for candidate in _model_json_candidates(hef_path):
        try:
            config = load_json(resolve_model_file(candidate))
        except Exception:
            continue
        if isinstance(config, dict):
            return config
    return {}


def labels_from_model_config(config):
    """Return the required class labels from ONNX-compatible model JSON."""
    classes = config.get("classes") if isinstance(config, dict) else None
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError(
            "Hailo model metadata must define a non-empty classes list"
        )
    return [str(label) for label in classes]


def _download_artifact(path) -> Path:
    """Resolve a local or remote artifact through the shared resolver."""
    from ...utils.remote import download_file

    return Path(resolve_artifact(path, downloader=download_file))


def resolve_hef_path(hef_path: Optional[str], task: str,
                     arch: Optional[str] = None) -> Optional[Path]:
    """Resolve an explicit local or Abraia-managed HEF artifact."""
    if hef_path is None:
        raise ValueError("A Hailo model URI is required")

    path = Path(hef_path)
    managed = is_managed_model_path(path)
    if path.is_dir() and not managed:
        candidates = sorted(path.rglob(f"*{HAILO_FILE_EXTENSION}"))
        if len(candidates) == 1:
            return candidates[0].resolve()
        if not candidates:
            logger.error("Hailo bundle '%s' contains no HEF file.", path)
        else:
            logger.error("Hailo bundle '%s' contains multiple HEF files.", path)
        return None
    if path.is_file() and not managed:
        return path.resolve()

    # Compiled Abraia models are stored under project paths such as
    # ``project/model_v2_hailo8l.hef``. Download those explicit HEF paths
    # instead of treating them as names in the built-in model catalog.
    if path.suffix.lower() == HAILO_FILE_EXTENSION and path.parent != Path("."):
        try:
            downloaded = _download_artifact(path)
            if downloaded.is_file():
                return downloaded.resolve()
        except Exception as exc:
            logger.warning("Unable to download Hailo model '%s': %s", path, exc)
            return None
    logger.error(
        "Hailo model '%s' is not an explicit HEF artifact. "
        "Provide a local HEF file or an uploaded Abraia HEF URI.",
        path,
    )
    return None


__all__ = [
    "HAILO_FILE_EXTENSION",
    "labels_from_model_config",
    "load_hailo_model_config",
    "resolve_model_type",
    "resolve_hef_path",
]
