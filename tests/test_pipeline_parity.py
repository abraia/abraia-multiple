"""Characterization matrix for ONNX and Hailo pipeline parity.

This module deliberately describes the current state of the two demo
catalogs.  It is the baseline for closing the gaps; later parity changes
should make entries in ``gaps`` disappear rather than changing the matrix
silently.
"""

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


def _logical_task(config: Dict[str, Any], backend: str) -> Optional[str]:
    """Infer the task represented by a demo config.

    The generic apple demo relies on its model sidecar to identify
    segmentation, so the matrix also recognizes the conventional ``-seg``
    model URI.  This keeps the comparison about behavior rather than whether
    the task happened to be written explicitly in JSON.
    """
    model = config.get("model", {})
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

    model = config.get("model", {})
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
        stages=tuple(stage.get("type", "") for stage in config.get("stages", [])),
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

    assert config["model"]["task"] == "segmentation"
    assert config["model"]["uri"] == "multiple/models/yolov8n_seg_hailo8.hef"


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
    assert PIPELINES["tomato"]["model"]["uri"].endswith(".onnx")
    assert all("kind" in config["model"] for config in PIPELINES.values())
    assert all(
        "onnx" not in config and "hailo" not in config
        for config in PIPELINE_DEVICES.values()
    )


def test_all_hailo_demos_are_accelerator_variants():
    for name in ("detect", "tomato", "apple", "segment", "pose"):
        with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
             patch("abraia.demo._hailo_model_available", return_value=True):
            selected = resolve_pipeline(name, accelerator="hailo")
        assert selected["model"]["kind"] == "yolov8"
        assert selected["model"]["task"] in ("detection", "segmentation", "pose")
        assert selected["model"]["uri"].endswith(".hef")


def test_auto_accelerator_selects_hailo_when_device_and_model_are_available():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        selected = resolve_pipeline("tomato", accelerator="auto")

    assert selected["model"]["uri"] == "multiple/tomato/yolov8n_hailo8.hef"
    assert selected["model"]["uri"] != PIPELINE_DEVICES["tomato"]["model"]["uri"]
    assert selected is not PIPELINE_DEVICES["tomato"]


def test_auto_accelerator_falls_back_to_onnx_when_hailo_is_unavailable():
    with patch("abraia.demo._hailo_device_arch", return_value=None):
        selected = resolve_pipeline("tomato", accelerator="auto")

    assert selected == PIPELINES["tomato"]
    assert selected is not PIPELINES["tomato"]


def test_default_detect_demo_keeps_the_generic_fallback_without_hailo():
    with patch("abraia.demo._hailo_device_arch", return_value=None):
        selected = resolve_pipeline("detect", accelerator="auto")

    assert selected["model"]["uri"].endswith("yolov8n.onnx")


def test_hailo_demo_resolves_to_bundled_architecture_specific_hef():
    with patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        selected = resolve_pipeline("detect", accelerator="hailo")

    assert selected["model"]["uri"].endswith(
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


def test_pipeline_parity_matrix_is_json_serializable():
    json.dumps(pipeline_parity_matrix())


if __name__ == "__main__":
    print(json.dumps(pipeline_parity_matrix(), indent=2))
