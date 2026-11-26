import numpy as np
import pytest

from abraia.inference.model_config import ModelSpec
from abraia.runtime import FrameContext, Pipeline
from abraia.runtime.config import PipelineStep
from abraia.runtime.pipeline import _build_pipeline_steps
from abraia.runtime.features import FeatureExtractionStep
from abraia.runtime.geometry import result_measurements


def test_measurement_geometry_is_available_from_runtime():
    values = result_measurements({"box": [2, 3, 4, 2]})

    assert values["area"] == 8
    assert values["centroid"] == [4, 4]


def test_mask_measurements_prefer_segmentation_geometry():
    result = result_measurements({
        "box": [0, 0, 10, 10],
        "mask": np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8),
    })

    assert result["area"] == 3
    assert result["centroid"] == pytest.approx([2 / 3, 2 / 3])
    assert result["bounds"] == [0, 0, 2, 2]


def test_anomaly_map_measurements_report_thresholded_area():
    from abraia.runtime.geometry import anomaly_map_measurements

    measurements = anomaly_map_measurements(
        np.array([[0.1, 0.8, 0.0], [0.7, 0.2, 0.9]]), 0.5,
    )

    assert measurements["area"] == 3
    assert measurements["fraction"] == pytest.approx(0.5)
    assert measurements["centroid"] == pytest.approx([1.0, 2 / 3])
    assert measurements["bounds"] == [0, 0, 3, 2]
    assert measurements["threshold"] == 0.5


def test_runtime_measure_step_does_not_require_multiple_registration():
    step = FeatureExtractionStep(
        "measure",
        {"features": ["area", "centroid"]},
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
        "area": 8,
        "centroid": [4, 4],
    }
    assert not any(name.endswith(".results") for name in context.artifacts)


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
            "features": ["area", "centroid"],
        }),
    ]

    runtime_steps, _components = _build_pipeline_steps(
        steps, [(Model(), {}, model_spec)], source=None, video_kwargs={},
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    context = FrameContext(frame, 0, 0.0, current=frame)
    for step in runtime_steps:
        context = step(context)

    assert context.results[0]["measurements"] == {
        "area": 8,
        "centroid": [4, 4],
    }


def test_measure_step_measures_masks_and_applies_calibration():
    step = FeatureExtractionStep(
        "measure",
        {
            "features": ["area", "obb"],
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

    measurements = context.results[0]["measurements"]
    assert measurements["area"] == 6
    *obb_geometry, angle = measurements["obb"]
    *scaled_geometry, scaled_angle = measurements["obb_mm"]
    assert obb_geometry == pytest.approx([1.0, 0.5, 1.0, 2.0])
    assert angle % 180 == pytest.approx(90.0)
    assert scaled_geometry == pytest.approx([0.5, 0.25, 0.5, 1.0])
    assert scaled_angle == angle
    assert measurements["area_mm2"] == 1.5


def test_measure_step_tracks_region_time_using_existing_track_ids():
    step = FeatureExtractionStep(
        "measure",
        {
            "features": ["area"],
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

    assert polygon_values["area"] == pytest.approx(12)
    assert anomaly_values["area"] == 2


def test_anomaly_renderer_does_not_publish_separate_metrics():
    from types import SimpleNamespace

    from abraia.runtime.rendering import AnomalyPipelineRenderer

    model = SimpleNamespace(
        last_normalized_anomaly_map=np.full((4, 4), 0.8, dtype=np.float32)
    )
    context = SimpleNamespace(
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        results=[{"label": "anomaly", "score": 0.9}],
        metrics={},
    )
    renderer = AnomalyPipelineRenderer(lambda current: current.frame, model)

    frame = renderer(context)

    assert frame.shape == (8, 8, 3)
    assert context.metrics == {}


def test_anomaly_measurement_merges_defects_and_decisions_into_results():
    from types import SimpleNamespace

    from abraia.runtime.features import FeatureExtractionStep

    model = SimpleNamespace(
        last_normalized_anomaly_map=np.full((4, 4), 0.9, dtype=np.float32)
    )
    step = FeatureExtractionStep(
        "measure",
        {"features": ["area"]},
        models={"anomaly": {
            "model": model,
            "configuration": {"kind": "anomaly", "task": "anomaly"},
        }},
    )
    context = SimpleNamespace(
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        current=[{"label": "anomaly", "score": 0.9}],
        results=[{"label": "anomaly", "score": 0.9}],
        metrics={}, artifacts={},
    )

    step(context)

    assert context.results[0]["decision_result"] == "FAIL"
    assert len(context.results) == 2
    assert context.results[1]["label"].startswith("Anomaly ")
    assert context.results[1]["decision_result"] == "FAIL"
    assert context.results[1]["measurements"]["area"] > 0
    assert "feature_measurements" not in context.metrics
    assert "anomaly" not in context.metrics


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("score, decision", [(0.2, "FAIL"), (None, "REVIEW")])
def test_pipeline_reuses_anomaly_inspection_for_measurements(monkeypatch, bound, score, decision):
    import cv2

    from abraia.runtime.anomaly import ANOMALY_MEASUREMENT_REGIONS

    class Detector:
        def run(self, _frame):
            return [{"box": [x, 2, 8, 8], "label": "part", "score": 0.9} for x in (2, 12)]

    class Anomaly:
        last_normalized_anomaly_map = np.pad(np.ones((4, 4), dtype=np.float32), 2)

        def run(self, _frame):
            return [{"label": "normal", "score": score}]

    calls = []
    connected_components = cv2.connectedComponentsWithStats

    def inspect(mask, **kwargs):
        calls.append(mask.shape)
        return connected_components(mask, **kwargs)

    monkeypatch.setattr(cv2, "connectedComponentsWithStats", inspect)
    anomaly = Anomaly()
    model_steps = [({"kind": "anomaly", "task": "anomaly"}, anomaly)]
    if bound:
        model_steps.insert(0, ({"kind": "yolov8", "task": "detection"}, Detector()))
    steps = [
        PipelineStep.from_config({"type": "model", "id": config["kind"], "model": config})
        for config, _ in model_steps
    ]
    steps.append(PipelineStep.from_config({
        "type": "measure", "id": "measure", "features": ["area", "centroid", "bounds"],
    }))
    models = [(model, {}, ModelSpec.from_config(config)) for config, model in model_steps]
    runtime_steps, _ = _build_pipeline_steps(steps, models, source=None, video_kwargs={})
    shape = (12, 24) if bound else (8, 8)
    frame = np.zeros((*shape, 3), dtype=np.uint8)

    context = Pipeline(source=[frame], model=models[0][0], steps=runtime_steps).run()

    assert calls == [(8, 8)] * (2 if bound else 1)
    measured = [result for result in context.results if "measurements" in result]
    assert [result["bounds"] for result in measured] == (
        [[4, 4, 4, 4], [14, 4, 4, 4]] if bound else [[2, 2, 4, 4]]
    )
    assert [result["centroid"] for result in measured] == (
        [[5.5, 5.5], [15.5, 5.5]] if bound else [[3.5, 3.5]]
    )
    assert all(
        result["area"] == 16 and "mask" not in result and result["decision_result"] == "FAIL"
        for result in measured
    )
    assert all(
        result["decision_result"] == decision
        for result in context.results if "measurements" not in result
    )
    assert ANOMALY_MEASUREMENT_REGIONS not in context.artifacts


def test_inspect_anomaly_map_preserves_full_frame_masks():
    from abraia.runtime.anomaly import inspect_anomaly_map

    heatmap = np.zeros((12, 16), dtype=np.float32)
    heatmap[2:4, 3:6] = 0.8
    heatmap[8, 11] = 0.9
    summary = inspect_anomaly_map(heatmap, heatmap.shape, [], {}, include_masks=True)
    large, small = summary["defect_regions"]
    assert large["box"] == [3, 2, 3, 2]
    assert small["box"] == [11, 8, 1, 1]
    assert large["area"] == 6
    assert small["score"] == pytest.approx(0.9)
    np.testing.assert_array_equal(large["mask"], heatmap == np.float32(0.8))
    np.testing.assert_array_equal(small["mask"], heatmap == np.float32(0.9))
