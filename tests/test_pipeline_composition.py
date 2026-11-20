import numpy as np
from unittest.mock import patch

from abraia.runtime import Pipeline
from abraia.runtime.pipeline import FrameContext
from abraia.runtime.config import PipelineDraft
from abraia.runtime.composition import (
    BoundModelStep,
    FilterStep,
    ModelStep,
)
from abraia.utils.draw import render_results


class Source:
    frame_rate = 1

    def __init__(self, frame):
        self.frame = frame
        self.closed = False

    def __iter__(self):
        yield self.frame

    def close(self):
        self.closed = True


class Detector:
    def run(self, image):
        assert image.shape == (10, 12, 3)
        return [
            {"box": [1, 2, 4, 3], "label": "plate", "score": 0.95},
            {"box": [7, 7, 3, 3], "label": "person", "score": 0.4},
        ]


class OCR:
    def __init__(self):
        self.crops = []

    def run(self, image):
        self.crops.append(tuple(image.shape[:2]))
        return [{"text": "ABC123", "score": 0.91}]


def test_bound_model_step_filters_crops_and_attaches_automatically():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    detector = Detector()
    ocr = OCR()
    steps = [
        ModelStep("detect", detector),
        BoundModelStep(
            "read_text",
            ocr,
            labels=["plate"],
            min_confidence=0.7,
            crop_padding=0.1,
            field="ocr",
        ),
    ]

    last = Pipeline(source=Source(frame), model=detector, steps=steps).run()

    assert ocr.crops == [(3, 4)]
    assert last.results[0]["ocr"]["text"] == "ABC123"
    assert "ocr" not in last.results[1]


def test_pipeline_results_are_full_table_records_for_all_output_shapes():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    records = [{
        "label": "part",
        "classification": {"label": "healthy", "score": 0.88},
        "regression": {"prediction": 2.4, "target": 2.0, "score": 0.7},
        "recognition": {"text": "AB-123", "text_score": 0.91},
        "measurements": {"area_px": 12},
        "box": [1, 2, 3, 4],
    }]

    class MultiOutput:
        def run(self, _frame):
            return records

    pipeline = Pipeline(source=Source(frame), model=MultiOutput(), steps=[])
    last = pipeline.run()

    assert isinstance(last.results, list)
    assert isinstance(last.results[0], dict)
    row = last.results[0]
    assert row["prediction"] == "part"
    assert row["classification_label"] == "healthy"
    assert row["classification_score"] == 0.88
    assert row["regression_value"] == 2.4
    assert row["regression_target"] == 2.0
    assert row["recognition_text"] == "AB-123"
    assert row["recognition_score"] == 0.91
    assert row["area_px"] == 12
    assert row["classification"] == {"label": "healthy", "score": 0.88}
    assert row["box"] == [1, 2, 3, 4]


def test_pipeline_results_preserve_zero_class_ids():
    from abraia.runtime.results import result_records

    record, = result_records([{"classification": {"class_id": 0}}])

    assert record["prediction"] == 0
    assert record["classification_label"] == 0


def test_decision_step_keeps_result_and_aggregate_decisions():
    from types import SimpleNamespace

    from abraia.runtime.composition import DecisionStep

    result = {"score": 0.8}
    context = SimpleNamespace(
        results=[result], current=[result], metrics={}, artifacts={},
    )

    DecisionStep("quality", "score", "gte", 0.8)(context)

    assert result["decision_result"] == "PASS"
    assert context.metrics["decision"]["status"] == "PASS"


def test_composed_draft_serializes_ordered_follow_up_model_steps():
    draft = PipelineDraft(
        source_type="image",
        source="frame.jpg",
        steps=[
            {
                "type": "model",
                "id": "model",
                "model": {"kind": "yolov8", "task": "detection"},
            },
            {
                "type": "model",
                "id": "read",
                "model": {"kind": "ocr", "task": "recognition"},
            },
        ],
    )

    config = draft.to_dict()

    assert [step["type"] for step in config["steps"]] == [
        "model", "model"
    ]
    assert not draft.validation_errors()


def test_composed_draft_rejects_incompatible_follow_up_model_step():
    draft = PipelineDraft.from_dict({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "task": "detection"},
            },
            {
                "id": "second",
                "type": "model",
                "model": {"kind": "yolov8", "task": "detection"},
            },
        ],
    })

    assert "cannot run as a second-step model" in " ".join(
        draft.validation_errors()
    )


def test_version_two_pipeline_builder_runs_composed_models():
    class Video:
        frame_rate = 1

        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def __iter__(self):
            yield np.zeros((6, 6, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    class Detector:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _image, **_kwargs):
            return [{"box": [1, 1, 3, 3], "label": "plate", "score": 0.9}]

        def close(self):
            pass

    class OCR:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, image, **_kwargs):
            assert tuple(image.shape[:2]) == (3, 3)
            return [{"text": "XYZ", "score": 0.88}]

        def close(self):
            pass

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "detect", "type": "model", "model": {
                "kind": "yolov8", "task": "detection", "uri": "detector.onnx"
            }},
            {"id": "read", "type": "model", "model": {
                "kind": "ocr", "task": "recognition"
            }},
        ],
        "display": {"show": False},
    }

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: Video()), \
         patch("abraia.inference.models.detection.Model", Detector), \
         patch("abraia.inference.models.ocr.TextSystem", OCR):
        pipeline = Pipeline.from_dict(config)
        last = pipeline.run()

    assert last.results[0]["ocr"]["text"] == "XYZ"


def test_version_two_pipeline_builder_runs_automatic_follow_up_model_step():
    class Video:
        frame_rate = 1

        def __init__(self, *_args, **_kwargs):
            pass

        def __iter__(self):
            yield np.zeros((6, 6, 3), dtype=np.uint8)

        def close(self):
            pass

    class Detector:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _image, **_kwargs):
            return [{"box": [1, 1, 3, 3], "label": "plate", "score": 0.9}]

        def close(self):
            pass

    class OCR:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, image, **_kwargs):
            assert tuple(image.shape[:2]) == (3, 3)
            return [{
                "box": [0, 0, 1, 1],
                "points": np.array(
                    [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32
                ),
                "text": "AUTO",
                "score": 0.93,
            }]

        def close(self):
            pass

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "detect", "type": "model", "model": {
                "kind": "yolov8", "task": "detection", "uri": "detector.onnx"
            }},
            {
                "id": "read_text",
                "type": "model",
                "model": {"kind": "ocr", "task": "recognition"},
            },
        ],
        "display": {"show": False},
    }

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: Video()), \
         patch("abraia.inference.models.detection.Model", Detector), \
         patch("abraia.inference.models.ocr.TextSystem", OCR):
        pipeline = Pipeline.from_dict(config)
        last = pipeline.run()

    assert last.results[0]["ocr"]["text"] == "AUTO"
    assert last.results[0]["ocr"]["items"][0]["box"] == [1.0, 1.0, 1.0, 1.0]
    assert last.results[0]["ocr"]["items"][0]["points"][0].tolist() == [1.0, 1.0]


def test_pipeline_renderer_shows_text_attached_by_second_model():
    results = [{
        "label": "plate",
        "score": 0.9,
        "box": [1, 1, 3, 3],
        "ocr": {
            "text": "AUTO",
            "score": 0.93,
            "items": [{"text": "AUTO", "score": 0.93, "box": [2, 2, 1, 1]}],
        },
    }]

    with patch("abraia.utils.draw.render_label") as render_label, \
         patch("abraia.utils.draw.render_box") as render_box:
        render_results(np.zeros((6, 6, 3), dtype=np.uint8), results)

    assert [call.args[1] for call in render_label.call_args_list] == [
        "plate", "AUTO"
    ]
    assert render_box.call_args_list[-1].args[1] == [2, 2, 1, 1]


def test_filter_step_combines_label_confidence_and_region():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    results = [
        {"box": [2, 2, 2, 2], "label": "car", "score": 0.9},
        {"box": [15, 15, 2, 2], "label": "car", "score": 0.9},
        {"box": [2, 2, 2, 2], "label": "truck", "score": 0.9},
        {"box": [2, 2, 2, 2], "label": "car", "score": 0.2},
    ]
    context = FrameContext(
        frame=frame, frame_index=0, frame_time=0, results=results,
    )
    step = FilterStep(
        "filter", labels=["car"], min_confidence=0.5,
        region_enabled=True,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
    )

    step(context)

    assert context.results == [results[0]]
    assert context.current == [results[0]]
