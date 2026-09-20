"""Hailo model metadata and explicit HEF resolution."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ...tasks import normalize_task
from ...utils.remote import (
    ARTIFACT_RESOLVER,
    is_managed_model_path,
)


logger = logging.getLogger(__name__)

HAILO_FILE_EXTENSION = ".hef"
HAILO_TARGETS = ("hailo8l", "hailo8", "hailo10h", "hailo15h", "hailo15l")

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def resolve_model_type(model_type, hef_path, task):
    """Resolve the postprocessor architecture for a Hailo model."""
    if model_type:
        return model_type
    normalized_task = normalize_task(task)
    if normalized_task not in ("detection", "segmentation", "pose"):
        raise ValueError(
            f"Unsupported Hailo task: {normalized_task}. "
            "Use detection, segmentation, or pose"
        )
    if normalized_task == "pose":
        return "v8"
    if normalized_task == "segmentation":
        # Native Ultralytics exports use the YOLOv8/YOLO11 raw head layout.
        # Keep the legacy Model Zoo YOLOv5 naming convention on its existing
        # decoder, and use v8 for custom/native bundles without a family name.
        return "v5" if "yolov5" in str(hef_path).lower() else "v8"
    return "v5"


def model_type_from_onnx_uri(onnx_uri):
    """Infer the Hailo postprocessor family from an ONNX model URI."""
    name = Path(os.fspath(onnx_uri)).stem.lower()
    if "yolov5" in name:
        return "v5"
    if "yolov8" in name or "yolo11" in name:
        return "v8"
    return None


def _metadata_bundle_path(path):
    """Return the directory that may contain an exported Hailo manifest."""
    path = Path(path)
    return path if path.is_dir() else path.parent


def load_hailo_metadata(hef_path):
    """Load Ultralytics' optional ``metadata.yaml`` beside a Hailo model."""
    path = Path(hef_path)
    metadata_paths = [_metadata_bundle_path(path) / "metadata.yaml"]
    if path.suffix.lower() == HAILO_FILE_EXTENSION:
        metadata_paths.append(path.with_suffix("") / "metadata.yaml")

    metadata_path = next(
        (candidate for candidate in metadata_paths if candidate.is_file()),
        None,
    )
    if metadata_path is None and path.parent != Path("."):
        # A compiled model may be addressed by its remote Abraia path.  The
        # compiler bundle is stored beside the flat HEF as ``<stem>/``.
        try:
            for candidate in metadata_paths:
                try:
                    downloaded = _download_artifact(candidate)
                except Exception:
                    continue
                if downloaded.is_file():
                    metadata_path = downloaded
                    break
        except Exception:
            return {}
    if metadata_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML is unavailable; ignoring Hailo metadata at %s", metadata_path)
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            value = yaml.safe_load(metadata_file)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        logger.warning("Unable to read Hailo metadata from %s", metadata_path)
        return {}
    return value if isinstance(value, dict) else {}


def labels_from_metadata(metadata):
    """Return ordered class labels from Ultralytics metadata."""
    names = metadata.get("names") if isinstance(metadata, dict) else None
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda key: int(key))]
        except (TypeError, ValueError):
            return [str(value) for value in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(value) for value in names]
    return None


def _model_json_candidates(hef_path):
    """Return JSON sidecar candidates for a local or remote HEF path."""
    path = Path(hef_path)
    if path.is_dir():
        return (path / "abraia.json", path / "model.json")

    stems = [path.stem]
    for target in HAILO_TARGETS:
        suffix = f"_{target}"
        if path.stem.endswith(suffix):
            stems.append(path.stem[: -len(suffix)])
            break
    variants = []
    for stem in stems:
        for variant in (stem, stem.replace("_seg", "-seg")):
            if variant not in variants:
                variants.append(variant)
    return tuple(path.with_name(f"{stem}.json") for stem in variants)


def load_hailo_model_config(hef_path):
    """Load the JSON sidecar used to configure the corresponding ONNX model."""
    from ...utils import load_json, resolve_model_file

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
    from ...utils import download_file

    return Path(ARTIFACT_RESOLVER.resolve(path, downloader=download_file))


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


def get_labels(labels_path: Optional[str]) -> list:
    """Load labels from a file, falling back to COCO labels."""
    if labels_path is None or not os.path.exists(labels_path):
        return COCO_LABELS
    with open(labels_path, "r", encoding="utf-8") as labels_file:
        return labels_file.read().splitlines()


__all__ = [
    "HAILO_FILE_EXTENSION",
    "COCO_LABELS",
    "labels_from_metadata",
    "labels_from_model_config",
    "load_hailo_model_config",
    "load_hailo_metadata",
    "resolve_model_type",
    "model_type_from_onnx_uri",
    "resolve_hef_path",
    "get_labels",
]
