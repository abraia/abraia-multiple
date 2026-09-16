"""Configuration registry for built-in inference models."""

import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_MODEL_URIS = {
    "object_detection": "multiple/models/yolov8n.onnx",
    "instance_segmentation": "multiple/models/yolov8n-seg.onnx",
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


def create_model(config: Dict[str, Any], base_dir=None):
    """Create a configured built-in model from a pipeline model config."""
    if not isinstance(config, dict):
        raise ValueError("Pipeline model must be an object")

    task = str(config.get("task", "detection")).strip().lower()
    task = {"detect": "detection", "recognize": "recognition"}.get(task, task)
    if task not in ("detection", "recognition"):
        raise ValueError(f"Unsupported pipeline model task: {task}")

    kind = str(config.get("kind", "onnx")).strip().lower()
    raw_params = config.get("params", {}) or {}
    if not isinstance(raw_params, dict):
        raise ValueError("Pipeline model 'params' must be an object")
    params = dict(raw_params)

    if kind in ("onnx", "object_detection", "instance_segmentation"):
        from .detect import Model

        uri = config.get("uri") or DEFAULT_MODEL_URIS.get(kind)
        if not uri:
            raise ValueError("An ONNX detector requires a model 'uri'")
        if params:
            raise ValueError("ONNX detector options belong directly in 'model'")
        if task != "detection":
            raise ValueError("ONNX pipeline models only support the detection task")
        return Model(_resolve_model_uri(uri, base_dir=base_dir))

    if kind in ("hailo", "hailo_detection", "hailo_segmentation"):
        from .hailo.pipeline import HailoPipelineModel

        uri = _resolve_model_uri(config.get("uri"), base_dir=base_dir)
        if not uri:
            raise ValueError("A Hailo pipeline model requires a model 'uri'")
        if task != "detection":
            raise ValueError("Hailo pipeline models currently support detection tasks")

        hailo_task = params.pop("hailo_task", None)
        if hailo_task is None:
            hailo_task = "segment" if kind == "hailo_segmentation" else "detect"
        params.setdefault("labels", config.get("labels"))
        params.setdefault("score_threshold", config.get("conf_threshold", 0.25))
        return HailoPipelineModel(uri, task=hailo_task, **params)

    if kind in ("face", "face_detector"):
        if task == "detection":
            from .faces import Retinaface

            return Retinaface(**params)
        from .faces import FaceRecognizer

        index = params.pop("index", None)
        if isinstance(index, (str, os.PathLike)) and base_dir is not None:
            index_path = Path(index)
            if not index_path.is_absolute():
                index = str(Path(base_dir) / index_path)
        return FaceRecognizer(index=index, **params)

    if kind in ("license_plate", "license_plate_detector", "plate"):
        if task == "detection":
            from .plates import LicensePlateDetector

            params.setdefault("threshold", 0.5)
            params.setdefault("iou_threshold", 0.1)
            params.setdefault("out_size", 300)
            return LicensePlateDetector(**params)
        from .plates import PlateRecognizer

        params.setdefault("threshold", 0.85)
        params.setdefault("iou_threshold", 0.15)
        params.setdefault("out_size", 300)
        return PlateRecognizer(**params)

    if kind in ("ocr", "text", "text_recognition"):
        if task != "recognition":
            raise ValueError("OCR models only support the recognition task")
        from .ocr import TextSystem

        return TextSystem(**params)

    available = "onnx, object_detection, instance_segmentation, hailo, face, license_plate, ocr"
    raise ValueError(f"Unknown detector kind '{kind}'. Available detectors: {available}")


__all__ = ["DEFAULT_MODEL_URIS", "create_model"]
