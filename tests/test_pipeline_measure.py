import numpy as np
import pytest

from abraia.inference.model_config import ModelSpec
from abraia.runtime import FrameContext
from abraia.runtime.config import PipelineStep
from abraia.runtime.pipeline import _build_pipeline_steps
from abraia.runtime.features import FeatureExtractionStep
from abraia.runtime.geometry import result_measurements


def test_measurement_geometry_is_available_from_runtime():
    values = result_measurements({"box": [2, 3, 4, 2]})

    assert values["area_px"] == 8
    assert values["centroid_px"] == [4, 4]


def test_mask_measurements_prefer_segmentation_geometry():
    result = result_measurements({
        "box": [0, 0, 10, 10],
        "mask": np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8),
    })

    assert result["area_px"] == 3
    assert result["centroid"] == pytest.approx([2 / 3, 2 / 3])
    assert result["bounds"] == [0, 0, 2, 2]


def test_anomaly_map_measurements_report_thresholded_area():
    from abraia.runtime.geometry import anomaly_map_measurements

    measurements = anomaly_map_measurements(
        np.array([[0.1, 0.8, 0.0], [0.7, 0.2, 0.9]]), 0.5,
    )

    assert measurements["area_px"] == 3
    assert measurements["fraction"] == pytest.approx(0.5)
    assert measurements["centroid"] == pytest.approx([1.0, 2 / 3])
    assert measurements["bounds"] == [0, 0, 3, 2]
    assert measurements["threshold"] == 0.5


def test_runtime_measure_step_does_not_require_multiple_registration():
    step = FeatureExtractionStep(
        "measure",
        {"features": ["area_px", "centroid_px"]},
        models={"detector": {
            "configuration": {"task": "detection"},
        }},
    )
    record = {"box": [2, 3, 4, 2]}

    class Context:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_time = 0.0
        current = [record]
        results = [record]
        metrics = {}
        artifacts = {}

    context = step(Context())

    assert context.results[0]["measurements"] == {
        "area_px": 8,
        "centroid_px": [4, 4],
    }
    assert context.artifacts["measure.results"] == context.results


def test_configured_pipeline_builds_and_runs_measure_without_multiple():
    class Model:
        def run(self, _frame, **_kwargs):
            return [{"label": "part", "score": 0.9, "box": [2, 3, 4, 2]}]

    model_configuration = {"kind": "yolov8", "task": "detection"}
    model_spec = ModelSpec.from_config(model_configuration)
    steps = [
        PipelineStep.from_config({
            "type": "model", "id": "detector", "model": model_configuration,
        }),
        PipelineStep.from_config({
            "type": "measure", "id": "measure",
            "features": ["area_px", "centroid_px"],
        }),
    ]

    runtime_steps, _components = _build_pipeline_steps(
        steps, [(Model(), {}, model_spec)], video=None, video_kwargs={},
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    context = FrameContext(frame, 0, 0.0, current=frame)
    for step in runtime_steps:
        context = step(context)

    assert context.results[0]["measurements"] == {
        "area_px": 8,
        "centroid_px": [4, 4],
    }


def test_measure_step_measures_masks_and_applies_calibration():
    step = FeatureExtractionStep(
        "measure",
        {
            "features": ["area_px", "width_px"],
            "calibration": {"pixels_per_unit": 2, "unit": "mm"},
        },
        models={"segmenter": {
            "configuration": {"task": "segmentation"},
        }},
    )
    record = {"mask": np.ones((2, 3), dtype=np.uint8)}

    class Context:
        current = [record]
        results = [record]
        metrics = {}
        artifacts = {}

    context = step(Context())

    assert context.results[0]["measurements"] == {
        "area_px": 6, "width_px": 3.0, "width_mm": 1.5,
        "area_mm2": 1.5,
    }


def test_measure_step_tracks_region_time_using_existing_track_ids():
    step = FeatureExtractionStep(
        "measure",
        {
            "features": ["area_px"],
            "time_in_region": True,
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "tracker_max_absent_frames": 3,
        },
        models={"detector": {
            "configuration": {"task": "detection"},
        }},
    )
    records = [{"track_id": 1, "box": [1, 1, 2, 2], "score": 0.9}]

    class Context:
        frame_time = 1.0
        current = records
        results = records
        metrics = {}
        artifacts = {}

    first = step(Context())
    assert first.results[0]["measurements"]["time_in_region_s"] == 0

    class Missing:
        frame_time = 2.0
        current = []
        results = []
        metrics = {}
        artifacts = {}

    step(Missing())

    class Returned:
        frame_time = 3.0
        current = [{"track_id": 1, "box": [1, 1, 2, 2], "score": 0.9}]
        results = current
        metrics = {}
        artifacts = {}

    assert step(Returned()).results[0]["measurements"]["time_in_region_s"] == 2


def test_runtime_measurement_helpers_cover_polygon_and_anomaly_maps():
    from abraia.runtime.geometry import anomaly_map_measurements

    polygon_values = result_measurements({
        "polygon": [[1, 1], [5, 1], [5, 4], [1, 4]],
    })
    anomaly_values = anomaly_map_measurements(
        np.array([[0.1, 0.8], [0.7, 0.2]]), 0.5
    )

    assert polygon_values["area_px"] == pytest.approx(12)
    assert anomaly_values["area_px"] == 2


def test_anomaly_renderer_publishes_json_safe_metrics():
    from types import SimpleNamespace

    from abraia.runtime.anomaly import AnomalyInspectionRenderer

    model = SimpleNamespace(
        last_normalized_anomaly_map=np.full((4, 4), 0.8, dtype=np.float32)
    )
    context = SimpleNamespace(
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        results=[{"label": "anomaly", "score": 0.9}],
        metrics={},
    )
    renderer = AnomalyInspectionRenderer(
        lambda current: current.frame.copy(), model,
        {"steps": [{"type": "model", "model": {"kind": "anomaly"}}]},
    )

    frame = renderer(context)

    assert frame.shape == (8, 8, 3)
    assert context.metrics["anomaly"]["decision"] == "fail"
    assert isinstance(context.metrics["anomaly"]["defect_regions"], list)
