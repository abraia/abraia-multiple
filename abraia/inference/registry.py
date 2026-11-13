"""Configuration registry for built-in inference models."""

import os
from pathlib import Path
from typing import Any, Dict

from ..utils.remote import is_url
from .model_config import (
    HAILO_TASKS,
    MODEL_ARCHITECTURES,
    HAILO_MODEL_ARCHITECTURES,
    MODEL_RUN_OPTIONS,
    MODEL_SIZE_URIS,
    MODEL_DESCRIPTORS,
    ModelSpec,
    _MODEL_DESCRIPTORS,
)
from .accelerators import (
    hailo_device_arch,
    hailo_artifact_available,
    onnx_providers,
    paired_hailo_uri,
)


def get_model_run_kwargs(config, spec):
    """Extract run-time options accepted by the configured model kind."""
    if spec.backend == "hailo":
        if spec.kind == "yolo9" and config.get("labels"):
            return {"labels": config["labels"]}
        return {}
    if not MODEL_DESCRIPTORS[spec.kind].runtime_options:
        return {}
    return {
        key: config[key]
        for key in MODEL_RUN_OPTIONS
        if key in config
        and (key != "labels" or config[key])
    }


def _resolve_model_uri(uri, base_dir=None):
    """Resolve a local model URI relative to a pipeline configuration."""
    uri = os.fspath(uri)
    if is_url(uri):
        return uri
    candidate = Path(uri)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir) / candidate
    return str(candidate) if candidate.is_file() else uri


def _select_hailo_pair(config, spec, uri, accelerator):
    """Build a paired Hailo config when the requested hardware can use it."""
    if (
        accelerator not in ("auto", "hailo")
        or spec.kind not in HAILO_MODEL_ARCHITECTURES
    ):
        return None

    hailo_task = spec.task
    if hailo_task not in HAILO_TASKS:
        return None
    architecture = hailo_device_arch()
    if not architecture:
        return None
    hailo_uri = paired_hailo_uri(uri, architecture)
    if not hailo_uri:
        return None
    if not hailo_artifact_available(hailo_uri, architecture):
        return None

    paired = dict(config)
    paired["kind"] = spec.kind
    paired["task"] = hailo_task
    paired["uri"] = hailo_uri
    paired["params"] = spec.params_copy()
    return paired


def _create_hailo_model(spec, config, base_dir):
    """Create a Hailo adapter from a validated model specification."""
    uri = _resolve_model_uri(spec.uri, base_dir=base_dir)
    if spec.kind == "anomaly":
        from .hailo.anomaly import HailoAnomalyModel

        return HailoAnomalyModel(uri)
    if spec.kind == "yolo9":
        from .hailo.yolov9 import HailoYOLOv9Model

        params = spec.params_copy()
        if "conf_threshold" in config:
            params.setdefault("score_threshold", config["conf_threshold"])
        if "iou_threshold" in config:
            params.setdefault("iou_threshold", config["iou_threshold"])
        return HailoYOLOv9Model(uri, **params)

    from .hailo.pipeline import HailoPipelineModel

    params = spec.params_copy()
    if "conf_threshold" in config:
        params.setdefault("score_threshold", config["conf_threshold"])
    if "iou_threshold" in config:
        params.setdefault("iou_threshold", config["iou_threshold"])
    return HailoPipelineModel(uri, task=spec.task, kind=spec.kind, **params)


def _create_onnx_adapter(adapter, uri, base_dir, session_options):
    """Instantiate a URI-based adapter with the shared ONNX session policy."""
    return adapter(
        _resolve_model_uri(uri, base_dir=base_dir),
        **session_options,
    )


def _create_resnet_model(spec, _config, base_dir, session_options):
    """Create the ResNet classification adapter."""
    if spec.params:
        raise ValueError("ResNet classifier options belong directly in 'model'")
    from .models.classification import ResNetClassifier

    return _create_onnx_adapter(
        ResNetClassifier, spec.uri, base_dir, session_options
    )


def _create_grounding_dino_model(spec, _config, base_dir, session_options):
    """Create the Grounding DINO adapter."""
    from .models.grounding_dino import GroundingDINOModel

    return GroundingDINOModel(
        _resolve_model_uri(spec.resolved_uri, base_dir=base_dir),
        **session_options,
        **spec.params_copy(),
    )


def _create_anomaly_model(spec, _config, base_dir, session_options):
    """Create the ONNX anomaly adapter."""
    from .models.anomaly import AnomalyModel

    return _create_onnx_adapter(
        AnomalyModel, spec.uri, base_dir, session_options
    )


def _create_clip_classifier(spec, _config, base_dir, session_options):
    """Create a CLIP image-classification adapter."""
    from .models.clip_classifier import ClipClassifier

    return _create_onnx_adapter(
        ClipClassifier, spec.uri, base_dir, session_options
    )


def _create_rfdetr_model(spec, _config, base_dir, session_options):
    """Create an RF-DETR ONNX adapter."""
    from .models.object_models import RFDETRONNXModel

    return _create_onnx_adapter(
        RFDETRONNXModel, spec.resolved_uri, base_dir, session_options
    )


def _create_dfine_model(spec, _config, base_dir, session_options):
    """Create a D-FINE ONNX adapter."""
    from .models.object_models import DFineONNXModel

    return _create_onnx_adapter(
        DFineONNXModel, spec.resolved_uri, base_dir, session_options
    )


def _create_yolo_model(spec, config, base_dir, session_options, accelerator=None):
    """Create a YOLO-family ONNX adapter, including optional Hailo pairing."""
    from .models.detection import Model

    uri = spec.resolved_uri
    if spec.params:
        raise ValueError("Model options belong directly in 'model'")

    resolved_uri = _resolve_model_uri(uri, base_dir=base_dir)
    paired = _select_hailo_pair(config, spec, resolved_uri, accelerator)
    if paired is not None:
        paired_spec = ModelSpec(
            kind=spec.kind,
            task=paired["task"],
            uri=paired["uri"],
            size=spec.size,
            params=paired["params"],
        )
        return _create_hailo_model(paired_spec, paired, base_dir)
    return Model(resolved_uri, kind=spec.kind, **session_options)


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
    "anomaly": _create_anomaly_model,
    "clip": _create_clip_classifier,
    "rfdetr": _create_rfdetr_model,
    "dfine": _create_dfine_model,
    "resnet": _create_resnet_model,
    "grounding_dino": _create_grounding_dino_model,
    "face": _create_face_model,
    "license_plate": _create_license_plate_model,
    "ocr": _create_ocr_model,
}


def register_model_factory(kind, factory, *, descriptor=None):
    """Register a model adapter and optional capabilities for a new kind.

    Factories receive ``(spec, config, base_dir, session_options)`` and return
    models implementing ``run(input, **options)`` and ``close()``. ``descriptor``
    is a public ``ModelDescriptor`` used by runtime and editor validation.
    Registration is process-local and replacing the same kind is supported.
    Factories may run concurrently with source construction.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("model factory kind must be a non-empty string")
    if not callable(factory):
        raise TypeError("model factory must be callable")
    if descriptor is not None:
        _MODEL_DESCRIPTORS[kind] = descriptor
    _MODEL_FACTORIES[kind] = factory


def create_model(config: Dict[str, Any], base_dir=None, accelerator=None):
    """Create a configured model using the registered model adapters."""
    spec = ModelSpec.from_config(config)
    spec.require_runtime_valid()
    return _create_model_from_spec(
        spec,
        config,
        base_dir=base_dir,
        accelerator=accelerator,
    )


def _create_model_from_spec(spec, config, base_dir=None, accelerator=None):
    """Create an adapter from a model spec validated by its caller."""
    providers = onnx_providers(accelerator)
    session_options = {} if providers is None else {"providers": providers}

    if spec.backend == "hailo":
        return _create_hailo_model(spec, config, base_dir)

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
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "MODEL_ARCHITECTURES",
    "create_model",
    "register_model_factory",
    "get_model_run_kwargs",
]
