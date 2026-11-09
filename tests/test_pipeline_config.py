"""Tests for pipeline configuration and ONNX/Hailo parity."""

from unittest.mock import patch

import pytest

from abraia.demo import (
    PIPELINE_DEVICES,
    PIPELINES,
    demo_model_options,
    resolve_pipeline,
)
from abraia.inference.model_config import (
    HAILO_TASKS,
    ModelSpec,
    PIPELINE_TASKS,
    normalize_task,
)
from abraia.runtime.config import (
    SUPPORTED_MODEL_OPTIONS,
    SUPPORTED_SECOND_STEP_MODEL_OPTIONS,
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


@pytest.mark.parametrize(
    "name, kind, task",
    [
        ("yolo9", "yolo9", "detection"),
        ("rfdetr", "rfdetr", "detection"),
        ("rfdetr-seg", "rfdetr", "segmentation"),
        ("dfine", "dfine", "detection"),
        ("dfine-seg", "dfine", "segmentation"),
    ],
)
def test_builtin_model_demos_define_the_expected_model(name, kind, task):
    model = _primary_model(PIPELINES[name])

    assert (model["kind"], model["task"]) == (kind, task)
    assert model["uri"].endswith(".onnx")


def test_hailo_apple_demo_uses_a_bundled_segmentation_model():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        config = resolve_pipeline("apple", accelerator="hailo")

    model = _primary_model(config)
    assert model["task"] == "segmentation"
    assert model["uri"] == "multiple/models/yolov8n-seg_hailo8.hef"


def test_demo_model_override_uses_a_compatible_builtin_variant():
    config = resolve_pipeline(
        "apple", accelerator="cpu", model="yolov8n-seg"
    )

    model = _primary_model(config)
    assert model["task"] == "segmentation"
    assert model["kind"] == "yolov8"
    assert model["uri"] == "multiple/models/yolov8n-seg.onnx"
    assert model["labels"] == ["apple"]
    assert "rfdetr-seg-nano" in demo_model_options("apple")
    assert "yolo9t" not in demo_model_options("apple")


def test_people_demo_can_select_yolov9_model_variant():
    config = resolve_pipeline("people", accelerator="cpu", model="yolo9t")

    model = _primary_model(config)
    assert model["kind"] == "yolo9"
    assert model["task"] == "detection"
    assert model["uri"] == "multiple/models/yolo9t.onnx"
    assert model["labels"] == ["person"]


def test_demo_model_override_rejects_incompatible_task_with_choices():
    with pytest.raises(ValueError, match="Compatible models:.*rfdetr-seg-nano"):
        resolve_pipeline("apple", accelerator="cpu", model="yolo9t")


def test_hailo_video_demos_use_video_metadata_for_resolution_and_fps():
    for name in ("tomato", "apple", "segment"):
        assert PIPELINE_DEVICES[name]["source"]["type"] == "video"
        assert "resolution" not in PIPELINE_DEVICES[name]["source"]
        assert "fps" not in PIPELINE_DEVICES[name]["source"]

    assert "resolution" not in PIPELINE_DEVICES["detect"]["source"]
    assert "fps" not in PIPELINE_DEVICES["detect"]["source"]
    assert "resolution" in PIPELINE_DEVICES["pose"]["source"]
    assert "fps" in PIPELINE_DEVICES["pose"]["source"]


def test_pipeline_devices_pair_existing_definitions_without_rewriting_them():
    assert _primary_model(PIPELINES["tomato"])["uri"].endswith(".onnx")
    assert all("kind" in _primary_model(config) for config in PIPELINES.values())
    assert all(
        "onnx" not in config and "hailo" not in config
        for config in PIPELINE_DEVICES.values()
    )


def test_all_hailo_demos_are_accelerator_variants():
    for name in ("detect", "tomato", "apple", "segment", "pose"):
        with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
             patch("abraia.demo._hailo_model_available", return_value=True):
            selected = resolve_pipeline(name, accelerator="hailo")
        model = _primary_model(selected)
        assert model["kind"] == "yolov8"
        assert model["task"] in ("detection", "segmentation", "pose")
        assert model["uri"].endswith(".hef")


def test_auto_accelerator_selects_hailo_when_device_and_model_are_available():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        selected = resolve_pipeline("tomato", accelerator="auto")

    assert _primary_model(selected)["uri"] == "multiple/tomato/yolov8n_hailo8.hef"
    assert _primary_model(selected)["uri"] != _primary_model(PIPELINE_DEVICES["tomato"])["uri"]
    assert selected is not PIPELINE_DEVICES["tomato"]


def test_auto_accelerator_falls_back_to_onnx_when_hailo_is_unavailable():
    with patch("abraia.demo._hailo_device_arch", return_value=None):
        selected = resolve_pipeline("tomato", accelerator="auto")

    assert selected == PIPELINES["tomato"]
    assert selected is not PIPELINES["tomato"]


def test_default_detect_demo_keeps_the_generic_fallback_without_hailo():
    with patch("abraia.demo._hailo_device_arch", return_value=None):
        selected = resolve_pipeline("detect", accelerator="auto")

    assert _primary_model(selected)["uri"].endswith("yolov8n.onnx")


def test_hailo_demo_resolves_to_bundled_architecture_specific_hef():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        selected = resolve_pipeline("detect", accelerator="hailo")

    assert _primary_model(selected)["uri"].endswith(
        "multiple/models/yolov8n_hailo8.hef"
    )


def test_auto_accelerator_does_not_use_the_unverified_apple_hef():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        selected = resolve_pipeline("apple", accelerator="auto")

    assert selected == PIPELINES["apple"]


def test_explicit_hailo_reports_a_missing_device():
    with patch("abraia.demo._hailo_device_arch", return_value=None):
        with pytest.raises(RuntimeError, match="no compatible Hailo device"):
            resolve_pipeline("tomato", accelerator="hailo")



def test_hailo_capabilities_are_explicitly_narrower_than_generic_catalog():
    assert set(HAILO_TASKS) == {
        "detection",
        "segmentation",
        "pose",
    }
    assert set(HAILO_TASKS) < set(PIPELINE_TASKS)


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
        "classification", "detection", "segmentation", "pose", "recognition",
        "anomaly",
    )
    assert [normalize_task(value) for value in (
        " detection ", "classify", "detect", "segment", "recognize",
    )] == [
        "detection", "classification", "detection", "segmentation", "recognition",
    ]


def test_new_models_are_available_in_runtime_editor_catalog():
    options = {(option[2], option[3]) for option in SUPPORTED_MODEL_OPTIONS}
    second_step_options = {
        (option[2], option[3]) for option in SUPPORTED_SECOND_STEP_MODEL_OPTIONS
    }

    assert {
        ("clip", "classification"),
        ("anomaly", "anomaly"),
        ("rfdetr", "detection"),
        ("rfdetr", "segmentation"),
        ("dfine", "detection"),
        ("dfine", "segmentation"),
    } <= options
    assert ("clip", "classification") in second_step_options
    assert ModelSpec.from_config({"kind": "clip"}).task == "classification"
    assert ModelSpec.from_config({"kind": "anomaly"}).task == "anomaly"
    assert model_defaults("rfdetr", "detection") == (0.35, 0.7)
    assert not model_control_visibility("clip", None)["iou"]
    anomaly_controls = model_control_visibility("anomaly", None)
    assert not anomaly_controls["labels"]
    assert not anomaly_controls["confidence"]
    assert not anomaly_controls["iou"]
