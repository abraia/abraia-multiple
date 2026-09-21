"""Configuration registry for built-in inference models."""

import os
from pathlib import Path
from typing import Any, Dict

from ..tasks import HAILO_TASKS
from .model_config import (
    GROUNDING_DINO_MODEL_KINDS,
    MODEL_ARCHITECTURES,
    MODEL_RUN_OPTIONS,
    MODEL_SIZE_URIS,
    MODEL_DESCRIPTORS,
    ModelSpec,
    PIPELINE_MODEL_KINDS,
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
    normalized = str(kind or "").strip().lower()
    descriptor = MODEL_DESCRIPTORS.get(normalized)
    return bool(descriptor and descriptor.runtime_options)


def model_backend(config):
    """Resolve the execution backend from the model URI."""
    if not isinstance(config, dict):
        return "onnx"
    return ModelSpec.from_config(config).backend


def get_model_run_kwargs(config):
    """Extract run-time options accepted by the configured model kind."""
    if not isinstance(config, dict):
        return {}
    spec = ModelSpec.from_config(config)
    if spec.backend == "hailo" or not supports_runtime_options(spec.kind):
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
    model_name = str(uri).lower()
    if any(
        marker in model_name
        for marker in ("-seg", "_seg", "-segment", "_segment")
    ):
        return "segmentation"
    if "_pose" in model_name:
        return "pose"
    return task


def _select_hailo_pair(config, spec, uri, accelerator):
    """Build a paired Hailo config when the requested hardware can use it."""
    if (
        accelerator not in ("auto", "hailo")
        or spec.kind not in MODEL_ARCHITECTURES
    ):
        return None

    hailo_task = _hailo_pair_task(spec.kind, spec.task, uri)
    if hailo_task not in HAILO_TASKS:
        return None
    architecture = hailo_device_arch()
    if not architecture:
        return None
    hailo_uri = paired_hailo_uri(uri, hailo_task, architecture)
    if not hailo_uri:
        return None
    candidate = {
        "model": {
            "kind": spec.kind,
            "task": hailo_task,
            "uri": hailo_uri,
        }
    }
    from .hailo.models import model_type_from_onnx_uri

    model_type = model_type_from_onnx_uri(uri)
    if model_type:
        candidate["model"]["params"] = {"model_type": model_type}
    if not hailo_model_available(candidate, architecture):
        return None

    paired = dict(config)
    paired["kind"] = spec.kind
    paired["task"] = hailo_task
    paired["uri"] = hailo_uri
    paired["params"] = spec.params_copy()
    if model_type:
        paired["params"].setdefault("model_type", model_type)
    return paired


def _create_hailo_model(spec, config, base_dir, session_options):
    """Create a Hailo adapter from a validated model specification."""
    from .hailo.pipeline import HailoPipelineModel

    uri = _resolve_model_uri(spec.uri, base_dir=base_dir)
    if not uri:
        raise ValueError("A Hailo pipeline model requires a model 'uri'")
    params = spec.params_copy()
    # Class names are authoritative in the model JSON sidecar, matching the
    # ONNX adapter. Ignore the former runtime labels override if present.
    params.pop("labels", None)
    params.setdefault("score_threshold", config.get("conf_threshold", 0.25))
    return HailoPipelineModel(uri, task=spec.task, **params)


def _create_resnet_model(spec, _config, base_dir, session_options):
    """Create the ResNet classification adapter."""
    if spec.params:
        raise ValueError("ResNet classifier options belong directly in 'model'")
    from .models.classification import ResNetClassifier

    return ResNetClassifier(
        _resolve_model_uri(spec.uri, base_dir=base_dir),
        **session_options,
    )


def _create_grounding_dino_model(spec, _config, base_dir, session_options):
    """Create the Grounding DINO adapter."""
    from .models.grounding_dino import GroundingDINOModel

    return GroundingDINOModel(
        _resolve_model_uri(spec.resolved_uri, base_dir=base_dir),
        **session_options,
        **spec.params_copy(),
    )


def _create_yolo_model(spec, config, base_dir, session_options, accelerator=None):
    """Create a YOLO-family ONNX adapter, including optional Hailo pairing."""
    from .models.detection import Model

    uri = spec.resolved_uri
    if not uri:
        raise ValueError(f"A {spec.kind} model requires a model 'uri'")
    if spec.params:
        raise ValueError("Model options belong directly in 'model'")

    resolved_uri = _resolve_model_uri(uri, base_dir=base_dir)
    paired = _select_hailo_pair(config, spec, resolved_uri, accelerator)
    if paired is not None:
        return create_model(
            paired,
            base_dir=base_dir,
            accelerator="hailo",
        )
    return Model(resolved_uri, **session_options)


def _create_face_model(spec, _config, base_dir, session_options):
    """Create a face detection or recognition adapter."""
    params = spec.params_copy()
    if spec.task == "detection":
        from .models.faces import Retinaface

        return Retinaface(**params, **session_options)

    from .models.faces import FaceRecognizer

    index = params.pop("index", None)
    if isinstance(index, (str, os.PathLike)) and base_dir is not None:
        index_path = Path(index)
        if not index_path.is_absolute():
            index = str(Path(base_dir) / index_path)
    return FaceRecognizer(index=index, **params, **session_options)


def _create_license_plate_model(spec, _config, _base_dir, session_options):
    """Create a license-plate detection or recognition adapter."""
    params = spec.params_copy()
    if spec.task == "detection":
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


def _create_ocr_model(spec, _config, _base_dir, session_options):
    """Create the OCR recognition adapter."""
    from .models.ocr import TextSystem

    return TextSystem(**spec.params_copy(), **session_options)


_MODEL_FACTORIES = {
    **{kind: _create_yolo_model for kind in MODEL_ARCHITECTURES},
    "resnet": _create_resnet_model,
    "grounding_dino": _create_grounding_dino_model,
    "face": _create_face_model,
    "license_plate": _create_license_plate_model,
    "ocr": _create_ocr_model,
}


def create_model(config: Dict[str, Any], base_dir=None, accelerator=None):
    """Create a configured model using the registered model adapters."""
    if not isinstance(config, dict):
        raise ValueError("Pipeline model must be an object")

    spec = ModelSpec.from_config(config)
    spec.require_runtime_valid()
    providers = onnx_providers(accelerator)
    session_options = {} if providers is None else {"providers": providers}

    if spec.backend == "hailo":
        return _create_hailo_model(spec, config, base_dir, session_options)

    try:
        factory = _MODEL_FACTORIES[spec.kind]
    except KeyError as error:
        raise AssertionError(f"Unhandled supported detector kind '{spec.kind}'") from error
    if spec.kind in MODEL_ARCHITECTURES:
        return factory(
            spec,
            config,
            base_dir,
            session_options,
            accelerator=accelerator,
        )
    return factory(spec, config, base_dir, session_options)


__all__ = [
    "GROUNDING_DINO_MODEL_KINDS",
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "MODEL_ARCHITECTURES",
    "PIPELINE_MODEL_KINDS",
    "RESNET_MODEL_KINDS",
    "create_model",
    "get_model_run_kwargs",
    "model_backend",
    "supports_runtime_options",
]
