"""Tests for the shared pipeline model configuration value object."""

import pytest

from abraia.inference.model_config import ModelSpec


def test_model_spec_normalizes_defaults_and_resolves_builtin_uri():
    spec = ModelSpec.from_config({"kind": "resnet"})

    assert spec.kind == "resnet"
    assert spec.task == "classification"
    assert spec.resolved_uri == "multiple/models/resnet18.onnx"
    assert spec.backend == "onnx"


def test_model_spec_resolves_size_specific_uri_and_backend():
    spec = ModelSpec.from_config({
        "kind": "yolov8",
        "task": "segment",
        "size": "large",
        "uri": "multiple/models/yolov8l_seg.hef",
    })

    assert spec.task == "segmentation"
    assert spec.resolved_uri == "multiple/models/yolov8l_seg.hef"
    assert spec.backend == "hailo"
    assert spec.allowed_tasks == {"detection", "segmentation", "pose"}


def test_model_spec_reports_pipeline_compatibility_errors():
    spec = ModelSpec.from_config({
        "kind": "resnet",
        "task": "detection",
        "uri": "multiple/models/resnet18.onnx",
    })

    assert spec.pipeline_errors() == (
        "Classification models require the classification task.",
    )


def test_model_spec_runtime_validation_rejects_invalid_hailo_kind():
    spec = ModelSpec.from_config({
        "kind": "resnet",
        "task": "classification",
        "uri": "multiple/models/resnet18.hef",
    })

    with pytest.raises(ValueError, match="Hailo models require"):
        spec.require_runtime_valid()


def test_model_spec_does_not_mutate_backend_parameters():
    spec = ModelSpec.from_config({
        "kind": "face",
        "task": "recognition",
        "params": {"index": "faces.json"},
    })

    params = spec.params_copy()
    params["index"] = "other.json"

    assert spec.params["index"] == "faces.json"


def test_model_spec_owns_legacy_parameter_mapping_and_serialization():
    spec = ModelSpec.from_config({
        "kind": "ocr",
        "task": "recognition",
        "params": {"drop_score": 0.6},
    })

    assert spec.editor_values()["conf_threshold"] == 0.6
    assert spec.to_pipeline_config(conf_threshold=0.7) == {
        "kind": "ocr",
        "task": "recognition",
        "params": {"drop_score": 0.7},
    }


def test_model_registry_has_an_adapter_for_every_supported_kind():
    from abraia.inference.model_config import PIPELINE_MODEL_KINDS
    from abraia.inference.registry import _MODEL_FACTORIES

    assert set(_MODEL_FACTORIES) == set(PIPELINE_MODEL_KINDS)
