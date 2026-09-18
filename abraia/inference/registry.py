"""Configuration registry for built-in inference models."""

import os
from pathlib import Path
from typing import Any, Dict

from ..tasks import HAILO_TASKS, normalize_task
from .model_config import (
    DEFAULT_MODEL_URIS,
    GROUNDING_DINO_MODEL_KINDS,
    HAILO_MODEL_KINDS,
    MODEL_RUN_OPTIONS,
    MODEL_SIZE_URIS,
    ONNX_MODEL_KINDS,
    RESNET_MODEL_KINDS,
)
from .accelerators import (
    hailo_device_arch,
    hailo_model_available,
    onnx_providers,
    paired_hailo_uri,
)


def supports_runtime_options(kind):
    """Return whether a model kind accepts generic run-time options."""
    normalized = str(kind or "onnx").strip().lower()
    return normalized in ONNX_MODEL_KINDS | GROUNDING_DINO_MODEL_KINDS | RESNET_MODEL_KINDS


def get_model_run_kwargs(config):
    """Extract run-time options accepted by the configured model kind."""
    if not isinstance(config, dict) or not supports_runtime_options(
        config.get("kind", "onnx")
    ):
        return {}
    return {
        key: config[key]
        for key in MODEL_RUN_OPTIONS
        if key in config
    }


def _resolve_model_uri(uri, base_dir=None):
    """Resolve a local model URI relative to a pipeline configuration."""
    if uri is None:
        return uri
    uri = os.fspath(uri)
    if uri.lower().startswith(("http://", "https://")):
        return uri
    candidate = Path(uri)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / candidate
    return str(candidate) if candidate.is_file() else uri


def _hailo_pair_task(kind, task, uri):
    """Return the Hailo task represented by an ONNX model definition."""
    if kind == "instance_segmentation" or "-seg" in str(uri).lower():
        return "segmentation"
    if kind == "pose" or "_pose" in str(uri).lower():
        return "pose"
    return task


def _select_hailo_pair(config, kind, task, uri, accelerator):
    """Build a paired Hailo config when the requested hardware can use it."""
    if accelerator not in ("auto", "hailo") or kind not in ONNX_MODEL_KINDS:
        return None

    hailo_task = _hailo_pair_task(kind, task, uri)
    if hailo_task not in HAILO_TASKS:
        return None
    hailo_uri = paired_hailo_uri(uri, hailo_task)
    if not hailo_uri:
        return None
    architecture = hailo_device_arch()
    candidate = {
        "model": {
            "kind": "hailo",
            "task": hailo_task,
            "uri": hailo_uri,
        }
    }
    if not architecture or not hailo_model_available(candidate, architecture):
        return None

    paired = dict(config)
    paired["kind"] = "hailo"
    paired["task"] = hailo_task
    paired["uri"] = hailo_uri
    paired["params"] = {}
    return paired


def create_model(config: Dict[str, Any], base_dir=None, accelerator=None):
    """Create a configured model using the requested accelerator policy.

    ONNX-backed models receive the resolved provider list. If a requested
    accelerator is unavailable, that policy supplies CPU providers instead.
    """
    if not isinstance(config, dict):
        raise ValueError("Pipeline model must be an object")

    raw_task = normalize_task(config.get("task"), default="detection")
    kind = str(config.get("kind", "onnx")).strip().lower()
    raw_params = config.get("params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("Pipeline model 'params' must be an object")
    params = dict(raw_params)
    task = raw_task
    providers = onnx_providers(accelerator)
    session_options = {} if providers is None else {"providers": providers}

    if kind in RESNET_MODEL_KINDS or (kind == "onnx" and task == "classification"):
        if "task" not in config:
            task = "classification"
        if task != "classification":
            raise ValueError("ResNet models require the classification task")
        if params:
            raise ValueError("ResNet classifier options belong directly in 'model'")
        uri = config.get("uri")
        if not uri:
            raise ValueError("A ResNet classifier requires a model 'uri'")
        from .models.classification import ResNetClassifier

        return ResNetClassifier(
            _resolve_model_uri(uri, base_dir=base_dir),
            **session_options,
        )

    if kind in GROUNDING_DINO_MODEL_KINDS:
        if task != "detection":
            raise ValueError("Grounding DINO models require the detection task")
        from .models.grounding_dino import GroundingDINOModel

        uri = _resolve_model_uri(
            config.get("uri", "multiple/models/grounding_dino_tiny.onnx"),
            base_dir=base_dir,
        )
        return GroundingDINOModel(uri, providers=providers, **params)

    if kind in ONNX_MODEL_KINDS:
        from .models.detection import Model

        if kind == "pose" and "task" not in config:
            task = "pose"
        configured_uri = config.get("uri")
        if configured_uri:
            uri = configured_uri
        else:
            size = str(config.get("size", "small")).strip().lower()
            size_kind = kind
            if kind == "onnx":
                size_kind = {
                    "detection": "object_detection",
                    "segmentation": "instance_segmentation",
                    "pose": "pose",
                }.get(task)
            if size_kind in MODEL_SIZE_URIS:
                try:
                    uri = MODEL_SIZE_URIS[size_kind][size]
                except KeyError as error:
                    available_sizes = ", ".join(MODEL_SIZE_URIS[size_kind])
                    raise ValueError(
                        f"Unsupported model size '{size}'. Use: {available_sizes}"
                    ) from error
            else:
                uri = DEFAULT_MODEL_URIS.get(kind)
        if not uri:
            raise ValueError("An ONNX detector requires a model 'uri'")
        if params:
            raise ValueError("ONNX detector options belong directly in 'model'")
        if kind == "pose" and task != "pose":
            raise ValueError("Pose ONNX models require the pose task")
        if kind == "onnx" and task not in ("detection", "pose"):
            raise ValueError(
                "Generic ONNX pipeline models only support detection or pose"
            )
        if kind in ("object_detection", "instance_segmentation") and task != "detection":
            raise ValueError("This ONNX model kind only supports the detection task")
        paired = _select_hailo_pair(
            config,
            kind,
            task,
            _resolve_model_uri(uri, base_dir=base_dir),
            accelerator,
        )
        if paired is not None:
            return create_model(
                paired,
                base_dir=base_dir,
                accelerator="hailo",
            )
        return Model(
            _resolve_model_uri(uri, base_dir=base_dir),
            **session_options,
        )

    if kind in HAILO_MODEL_KINDS:
        from .hailo.pipeline import HailoPipelineModel

        uri = _resolve_model_uri(config.get("uri"), base_dir=base_dir)
        if not uri:
            raise ValueError("A Hailo pipeline model requires a model 'uri'")
        if raw_task not in HAILO_TASKS:
            raise ValueError(f"Unsupported Hailo pipeline task: {raw_task}")
        params.setdefault("labels", config.get("labels"))
        params.setdefault("score_threshold", config.get("conf_threshold", 0.25))
        return HailoPipelineModel(uri, task=raw_task, **params)

    if task not in ("detection", "recognition"):
        raise ValueError(f"Unsupported pipeline model task: {task}")

    if kind in ("face", "face_detector"):
        if task == "detection":
            from .models.faces import Retinaface

            return Retinaface(**params, **session_options)
        from .models.faces import FaceRecognizer

        index = params.pop("index", None)
        if isinstance(index, (str, os.PathLike)) and base_dir is not None:
            index_path = Path(index)
            if not index_path.is_absolute():
                index = str(Path(base_dir) / index_path)
        return FaceRecognizer(index=index, **params, **session_options)

    if kind in ("license_plate", "license_plate_detector", "plate"):
        if task == "detection":
            from .models.plates import LicensePlateDetector

            params.setdefault("threshold", 0.5)
            params.setdefault("iou_threshold", 0.1)
            params.setdefault("out_size", 300)
            return LicensePlateDetector(**params, **session_options)
        from .models.plates import PlateRecognizer

        params.setdefault("threshold", 0.85)
        params.setdefault("iou_threshold", 0.15)
        params.setdefault("out_size", 300)
        return PlateRecognizer(**params, **session_options)

    if kind in ("ocr", "text", "text_recognition"):
        if task != "recognition":
            raise ValueError("OCR models only support the recognition task")
        from .models.ocr import TextSystem

        return TextSystem(**params, **session_options)

    available = "onnx, object_detection, instance_segmentation, pose, classification, resnet, grounding_dino, hailo, face, license_plate, ocr"
    raise ValueError(f"Unknown detector kind '{kind}'. Available detectors: {available}")


__all__ = [
    "DEFAULT_MODEL_URIS",
    "GROUNDING_DINO_MODEL_KINDS",
    "HAILO_MODEL_KINDS",
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "ONNX_MODEL_KINDS",
    "RESNET_MODEL_KINDS",
    "create_model",
    "get_model_run_kwargs",
    "supports_runtime_options",
]
