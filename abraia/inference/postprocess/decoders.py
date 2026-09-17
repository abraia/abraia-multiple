"""Decoder factory and public ONNX post-processing interface."""

from .classification import ClassificationDecoder, postprocess, preprocess
from .detection import (
    DetectionDecoder,
    get_mask,
    prepare_input,
    process_output,
    validate_model_config,
)
from .pose import PoseDecoder, process_pose_output


def create_decoder(task, classes):
    """Create the task-specific output decoder used by an ONNX model."""
    if task in ("detection", "segmentation"):
        return DetectionDecoder(classes)
    if task == "pose":
        return PoseDecoder(classes)
    if task == "classification":
        return ClassificationDecoder(classes)
    raise ValueError(f"Unsupported ONNX inference task: {task}")


__all__ = [
    "ClassificationDecoder",
    "DetectionDecoder",
    "PoseDecoder",
    "create_decoder",
    "get_mask",
    "postprocess",
    "preprocess",
    "prepare_input",
    "process_output",
    "process_pose_output",
    "validate_model_config",
]
