"""Tests for pipeline configuration and ONNX/Hailo parity."""

from dataclasses import asdict, dataclass
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from abraia.demo import (
    PIPELINE_DEVICES,
    PIPELINES,
    resolve_pipeline,
)
from abraia.inference.hailo.pipeline import HailoPipelineModel
from abraia.inference.model_config import (
    MODEL_ARCHITECTURES,
    ModelSpec,
)
from abraia.tasks import HAILO_TASKS, normalize_task


PAIRED_DEMOS = {
    "tomato": "tomato",
    "apple": "apple",
    "detect": "detect",
    "segment": "segment",
    "pose": "pose",
    "people": "people",
    "queue": "queue",
    "escalator": "escalator",
}


@dataclass(frozen=True)
class BackendSnapshot:
    """Comparable facts extracted from one demo configuration."""

    present: bool
    model_kind: Optional[str] = None
    task: Optional[str] = None
    labels: Tuple[str, ...] = ()
    stages: Tuple[str, ...] = ()
    source_type: Optional[str] = None


@dataclass(frozen=True)
class ParitySnapshot:
    """Current parity facts for one logical demo."""

    name: str
    onnx: BackendSnapshot
    hailo: BackendSnapshot
    gaps: Tuple[str, ...]


def _primary_model(config: Dict[str, Any]) -> Dict[str, Any]:
    steps = config.get("steps", [])
    if steps and steps[0].get("type") == "model":
        return steps[0].get("model", {})
    return {}


def _logical_task(config: Dict[str, Any], backend: str) -> Optional[str]:
    """Infer the task represented by a demo config.

    The generic apple demo relies on its model sidecar to identify
    segmentation, so the matrix also recognizes the conventional ``-seg``
    model URI.  This keeps the comparison about behavior rather than whether
    the task happened to be written explicitly in JSON.
    """
    model = _primary_model(config)
    task = model.get("task")
    if task:
        return normalize_task(task)

    uri = str(model.get("uri", "")).lower()
    if backend == "onnx" and ("-seg" in uri or "_seg" in uri):
        return "segmentation"
    if backend == "onnx" and "pose" in uri:
        return "pose"
    return "detection"


def _snapshot(config: Optional[Dict[str, Any]], backend: str) -> BackendSnapshot:
    if config is None:
        return BackendSnapshot(present=False)

    model = _primary_model(config)
    labels = model.get("labels", []) or []
    if isinstance(labels, str):
        labels = (labels,)
    else:
        labels = tuple(str(label) for label in labels)

    return BackendSnapshot(
        present=True,
        model_kind=str(model.get("kind", "")),
        task=_logical_task(config, backend),
        labels=labels,
        stages=tuple(
            step.get("type", "")
            for step in config.get("steps", [])
            if step.get("type") != "model"
        ),
        source_type=config.get("source", {}).get("type"),
    )


def _gaps(onnx: BackendSnapshot, hailo: BackendSnapshot) -> Tuple[str, ...]:
    if not onnx.present or not hailo.present:
        return ("missing_backend_demo",)

    gaps: List[str] = []
    if onnx.task != hailo.task:
        gaps.append("task_mismatch")
    if onnx.labels != hailo.labels:
        gaps.append("labels_mismatch")
    if onnx.stages != hailo.stages:
        gaps.append("stage_sequence_mismatch")
    if onnx.source_type != hailo.source_type:
        gaps.append("source_type_mismatch")
    return tuple(gaps)


def pipeline_parity_matrix() -> Dict[str, Any]:
    """Return a JSON-compatible current-state parity matrix."""
    def resolved_hailo(name):
        with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
             patch("abraia.demo._hailo_model_available", return_value=True):
            return resolve_pipeline(name, accelerator="hailo")

    paired = {
        name: ParitySnapshot(
            name=name,
            onnx=_snapshot(PIPELINES.get(name), "onnx"),
            hailo=_snapshot(resolved_hailo(hailo_name), "hailo"),
            gaps=(),
        )
        for name, hailo_name in PAIRED_DEMOS.items()
    }
    paired = {
        name: ParitySnapshot(
            name=snapshot.name,
            onnx=snapshot.onnx,
            hailo=snapshot.hailo,
            gaps=_gaps(snapshot.onnx, snapshot.hailo),
        )
        for name, snapshot in paired.items()
    }

    matrix = {
        "paired": {name: asdict(snapshot) for name, snapshot in paired.items()},
        "onnx_only_demos": sorted(set(PIPELINES) - set(PAIRED_DEMOS)),
        "hailo_only_demos": [],
        "model_capabilities": {
            "architectures": sorted(MODEL_ARCHITECTURES),
            "hailo_tasks": list(HAILO_TASKS),
            "hailo_constructor_options": sorted(
                name
                for name in inspect.signature(HailoPipelineModel.__init__).parameters
                if name != "self"
            ),
        },
    }
    # Normalize dataclass tuples to lists so callers can compare and persist
    # the result exactly as JSON.
    return json.loads(json.dumps(matrix))


def test_pipeline_parity_matrix_has_expected_current_baseline():
    matrix = pipeline_parity_matrix()

    assert set(matrix["paired"]) == set(PAIRED_DEMOS)
    assert matrix["paired"]["tomato"]["gaps"] == []
    assert matrix["paired"]["apple"]["gaps"] == []
    assert matrix["onnx_only_demos"] == ["grapes", "plates", "strawberry"]
    assert matrix["hailo_only_demos"] == []


def test_hailo_apple_demo_uses_a_bundled_segmentation_model():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        config = resolve_pipeline("apple", accelerator="hailo")

    model = _primary_model(config)
    assert model["task"] == "segmentation"
    assert model["uri"] == "multiple/models/yolov8n_seg_hailo8.hef"


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
    assert PIPELINE_DEVICES["tomato"] is PIPELINES["tomato"]
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
    matrix = pipeline_parity_matrix()
    capabilities = matrix["model_capabilities"]

    assert set(capabilities["hailo_tasks"]) == {
        "detection",
        "segmentation",
        "pose",
    }
    assert "iou_threshold" not in capabilities["hailo_constructor_options"]
    assert "batch_size" in capabilities["hailo_constructor_options"]
    assert "model_type" in capabilities["hailo_constructor_options"]


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


if __name__ == "__main__":
    print(json.dumps(pipeline_parity_matrix(), indent=2))
