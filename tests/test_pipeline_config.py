"""Tests for pipeline configuration and ONNX/Hailo parity."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from abraia.demo import (
    DEMO_TASKS,
    demo_model_options,
    resolve_pipeline,
)
from abraia.inference.model_config import (
    HAILO_TASKS,
    ModelSpec,
    PIPELINE_TASKS,
    normalize_task,
)
from abraia.inference.registry import get_model_run_kwargs, resolve_model_backend
from abraia.runtime.config import (
    SUPPORTED_MODEL_OPTIONS,
    model_control_visibility,
    model_defaults,
    model_editor_values,
    model_pipeline_errors,
)


def _primary_model(config):
    steps = config.get("steps", [])
    if steps and steps[0].get("type") == "model":
        return steps[0].get("model", {})
    return {}


@pytest.fixture
def remote_demos(monkeypatch, public_demo_configs):
    monkeypatch.setattr(
        "abraia.demo.load_public_pipeline",
        lambda name: deepcopy(public_demo_configs[name]),
    )
    return public_demo_configs


def test_model_demo_tasks_are_available_for_completion():
    assert DEMO_TASKS["detect"] == "detection"
    assert "yolo9" not in DEMO_TASKS
    assert "rfdetr" not in DEMO_TASKS
    assert "dfine" not in DEMO_TASKS
    assert "yolov8n" in demo_model_options("detect")
    assert "yolo9t" in demo_model_options("detect")
    assert "rfdetr-nano" in demo_model_options("detect")
    assert "dfine-n" in demo_model_options("detect")
    assert "rfdetr-seg-nano" in demo_model_options("segment")
    assert "dfine-seg-n" in demo_model_options("segment")


def test_demo_model_override_uses_a_compatible_builtin_variant(remote_demos):
    _primary_model(remote_demos["apple"])["labels"] = ["apple"]
    config = resolve_pipeline("apple", model="yolov8n-seg")

    model = _primary_model(config)
    assert model["task"] == "segmentation"
    assert model["kind"] == "yolov8"
    assert model["uri"] == "multiple/models/yolov8n-seg.onnx"
    assert model["labels"] == ["apple"]
    assert "rfdetr-seg-nano" in demo_model_options("apple")
    assert "yolo9t" not in demo_model_options("apple")


def test_people_demo_can_select_yolov9_model_variant(remote_demos):
    _primary_model(remote_demos["people"])["labels"] = ["person"]
    config = resolve_pipeline("people", model="yolo9t")

    model = _primary_model(config)
    assert model["kind"] == "yolo9"
    assert model["task"] == "detection"
    assert model["uri"] == "multiple/models/yolo9t.onnx"
    assert model["labels"] == ["person"]


def test_demo_model_override_rejects_incompatible_task_with_choices(remote_demos):
    with pytest.raises(ValueError, match="Compatible models:.*rfdetr-seg-nano"):
        resolve_pipeline("apple", model="yolo9t")


@pytest.mark.parametrize("demo", ["detect", "tomato", "apple", "segment", "pose"])
def test_public_demo_models_use_the_shared_hef_resolver(remote_demos, demo):
    config = resolve_pipeline(demo)
    spec = ModelSpec.from_config(_primary_model(config))
    with patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8"), \
         patch("abraia.inference.registry.hailo_artifact_available", return_value=True):
        selected = resolve_model_backend(spec, accelerator="hailo")

    assert selected.kind == spec.kind and selected.task == spec.task
    assert selected.uri == spec.uri[:-5] + "_hailo8.hef"
    assert config == remote_demos[demo]


@pytest.mark.parametrize("config,architecture,message", [
    ({"kind": "yolov8"}, None, "no compatible Hailo device"),
    ({"kind": "yolov8"}, "hailo8", "no compatible HEF is available for hailo8"),
    ({"kind": "depth_anything"}, "hailo8", "no automatic HEF pairing"),
])
def test_explicit_hailo_requires_an_artifact_while_auto_falls_back(config, architecture, message):
    spec = ModelSpec.from_config(config)
    with patch("abraia.inference.registry.hailo_device_arch", return_value=architecture), \
         patch("abraia.inference.registry.hailo_artifact_available", return_value=False):
        assert resolve_model_backend(spec, accelerator="auto") is spec
        with pytest.raises(RuntimeError, match=spec.kind + ".*" + message):
            resolve_model_backend(spec, accelerator="hailo")


def test_saved_relative_model_can_select_a_local_hef_without_an_onnx_file(tmp_path):
    compiled = tmp_path / "detector_hailo8l.hef"
    compiled.touch()
    spec = ModelSpec.from_config({"kind": "yolov8", "uri": "detector.onnx"})
    with patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8l"):
        selected = resolve_model_backend(spec, base_dir=tmp_path, accelerator="auto")
    assert selected.uri == str(compiled)


@pytest.mark.parametrize("kind,expected", [
    ("grounding_dino", {"prompt": "bird", "conf_threshold": .4}),
    ("yolov8", {"conf_threshold": .4}),
    ("resnet", {"top_k": 3, "conf_threshold": .4}),
    ("yolo26_depth", {}),
])
def test_builtin_model_run_options_filter_irrelevant_fields(kind, expected):
    config = {"kind": kind, "prompt": "bird", "top_k": 3, "conf_threshold": .4, "labels": []}
    assert get_model_run_kwargs(config, ModelSpec.from_config(config)) == expected

def test_hailo_capabilities_are_explicitly_narrower_than_generic_catalog():
    assert set(HAILO_TASKS) == {
        "detection",
        "segmentation",
        "pose",
        "obb",
        "depth",
    }
    assert set(HAILO_TASKS) < set(PIPELINE_TASKS)


@pytest.mark.parametrize("kind,task", [
    ("yolov8", "depth"),
    ("yolo26_depth", "detection"),
])
def test_hailo_depth_requires_yolo26_depth_kind_and_task(kind, task):
    spec = ModelSpec.from_config({"kind": kind, "task": task, "uri": "model.hef"})
    with pytest.raises(ValueError, match="Hailo depth inference requires"):
        spec.require_runtime_valid()


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
    assert spec.allowed_tasks == {"detection", "segmentation", "pose", "obb"}


def test_model_spec_reports_pipeline_compatibility_errors():
    spec = ModelSpec.from_config({
        "kind": "resnet",
        "task": "detection",
        "uri": "multiple/models/resnet18.onnx",
    })

    assert model_pipeline_errors({
        "kind": spec.kind,
        "task": spec.task,
        "uri": spec.uri,
    }) == (
        "Classification models require the classification task.",
    )


def test_model_spec_maps_bound_model_results_to_their_pipeline_field():
    assert ModelSpec.from_config({
        "kind": "ocr",
        "task": "recognition",
    }).result_field == "ocr"
    assert ModelSpec.from_config({
        "kind": "resnet",
        "task": "classification",
    }).result_field == "classification"
    assert ModelSpec.from_config({
        "kind": "face",
        "task": "recognition",
    }).result_field == "recognition"


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


def test_model_editor_values_read_legacy_model_parameters():
    spec = ModelSpec.from_config({
        "kind": "ocr",
        "task": "recognition",
        "params": {"drop_score": 0.6},
    })

    assert model_editor_values({
        "kind": spec.kind,
        "task": spec.task,
        "params": dict(spec.params),
    })["conf_threshold"] == 0.6


def test_model_tasks_normalize_names_and_aliases():
    assert PIPELINE_TASKS == (
        "classification", "detection", "segmentation", "pose", "obb", "recognition",
        "anomaly", "depth",
    )
    assert [normalize_task(value) for value in (
        " detection ", "classify", "detect", "segment", "recognize",
    )] == [
        "detection", "classification", "detection", "segmentation", "recognition",
    ]


def test_new_models_are_available_in_runtime_editor_catalog():
    options = {(option[2], option[3]) for option in SUPPORTED_MODEL_OPTIONS}

    assert {
        ("anomaly", "anomaly"),
        ("rfdetr", "detection"),
        ("rfdetr", "segmentation"),
        ("dfine", "detection"),
        ("dfine", "segmentation"),
        ("depth_anything", "depth"),
        ("yolo26_depth", "depth"),
        ("yolov8", "obb"),
    } <= options
    assert ModelSpec.from_config({"kind": "anomaly"}).task == "anomaly"
    assert model_defaults("rfdetr", "detection") == (0.35, 0.7)
    anomaly_controls = model_control_visibility("anomaly", None)
    assert not anomaly_controls["labels"]
    assert not anomaly_controls["confidence"]
    assert not anomaly_controls["iou"]
    for kind in ("depth_anything", "yolo26_depth"):
        assert ModelSpec.from_config({"kind": kind}).task == "depth"
        controls = model_control_visibility(kind, None)
        assert controls["uri"]
        assert not any(controls[name] for name in ("labels", "confidence", "iou"))
