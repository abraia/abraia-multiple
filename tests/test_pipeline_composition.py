import numpy as np
from unittest.mock import patch
from _support import qt_app

from abraia.runtime import FrameContext, Pipeline
from abraia.runtime.config import PipelineDraft
from studio.features.pipeline.view import PipelineView
from abraia.runtime.composition import (
    AttachStep,
    BoundModelStep,
    CropStep,
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


def test_composed_steps_crop_run_and_attach_results():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    detector = Detector()
    ocr = OCR()
    steps = [
        ModelStep("detect", detector),
        FilterStep("plates", "detect.results", labels=["plate"], min_confidence=0.7),
        CropStep("crop", "plates.results"),
        ModelStep("read", ocr, input_ref="crop.items"),
        AttachStep("attach", "read.results", "detect.results", field="ocr"),
    ]

    last = Pipeline(source=Source(frame), model=detector, steps=steps).run()

    assert ocr.crops == [(3, 4)]
    assert last.results[0]["ocr"]["text"] == "ABC123"
    assert last.results[0]["ocr"]["text_score"] == 0.91
    assert "ocr" not in last.results[1]
    assert last.results[0]["id"] == "detect-0"


def test_composed_pipeline_keeps_region_parent_identity():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    context = FrameContext(
        frame=frame,
        frame_index=0,
        frame_time=0,
        results=[{"id": "object-7", "box": [2, 2, 3, 3]}],
    )

    CropStep("crop", "results")(context)

    region = context.artifacts["crop.items"][0]
    assert region.parent_id == "object-7"
    assert region.source_box == (2, 2, 3, 3)
    assert tuple(region.image.shape[:2]) == (3, 3)


def test_bound_model_stage_filters_crops_and_attaches_automatically():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    detector = Detector()
    ocr = OCR()
    steps = [
        ModelStep("detect", detector),
        BoundModelStep(
            "read_text",
            ocr,
            input_ref="detect.results",
            labels=["plate"],
            min_confidence=0.7,
            crop_padding=0.1,
            attach_to="detect.results",
            field="ocr",
        ),
    ]

    last = Pipeline(source=Source(frame), model=detector, steps=steps).run()

    assert ocr.crops == [(3, 4)]
    assert last.results[0]["ocr"]["text"] == "ABC123"
    assert "ocr" not in last.results[1]


def test_composed_draft_serializes_and_validates_step_references():
    draft = PipelineDraft(
        source_type="image",
        source="frame.jpg",
        model_kind="yolov8",
        model_task="detection",
        stages=[
            {"type": "crop", "id": "crop", "input": "model.results", "padding": 0.1},
            {
                "type": "model",
                "id": "read",
                "input": "crop.items",
                "model": {"kind": "ocr", "task": "recognition"},
            },
            {
                "type": "attach",
                "id": "attach",
                "input": "read.results",
                "target": "model.results",
                "field": "ocr",
            },
        ],
    )

    config = draft.to_dict()

    assert config["version"] == 2
    assert [step["id"] for step in config["steps"]] == [
        "model", "crop", "read", "attach"
    ]
    assert not draft.validation_errors()


def test_composed_draft_rejects_missing_intermediate_reference():
    draft = PipelineDraft(
        stages=[
            {
                "type": "model",
                "id": "read",
                "input": "crop.items",
                "model": {"kind": "ocr", "task": "recognition"},
            }
        ]
    )

    assert "unknown input reference 'crop.items'" in " ".join(
        draft.validation_errors()
    )


def test_composed_draft_rejects_incompatible_second_stage_model():
    draft = PipelineDraft.from_dict({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "input": "frame",
                "model": {"kind": "yolov8", "task": "detection"},
            },
            {
                "id": "second",
                "type": "model",
                "model": {"kind": "yolov8", "task": "detection"},
            },
        ],
    })

    assert "cannot run as a second-stage model" in " ".join(
        draft.validation_errors()
    )


def test_composed_draft_rejects_removed_second_stage_routing_fields():
    draft = PipelineDraft.from_dict({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "input": "frame",
                "model": {"kind": "yolov8", "task": "detection"},
            },
            {
                "id": "read_text",
                "type": "model",
                "input": {"from": "model.results"},
                "crop": {"padding": 0.1},
                "output": {"attach_to": "model.results", "field": "ocr"},
                "model": {"kind": "ocr", "task": "recognition"},
            },
        ],
    })

    errors = " ".join(draft.validation_errors())
    assert "removed explicit model field(s)" in errors
    assert all(field in errors for field in ("input", "crop", "output"))


def test_version_two_pipeline_builder_runs_composed_models():
    class Video:
        frame_rate = 1

        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def __iter__(self):
            yield np.zeros((6, 6, 3), dtype=np.uint8)

        def set_display_enabled(self, _enabled):
            pass

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
            {"id": "detect", "type": "model", "input": "frame", "model": {
                "kind": "yolov8", "task": "detection", "uri": "detector.onnx"
            }},
            {"id": "read", "type": "model", "model": {
                "kind": "ocr", "task": "recognition"
            }},
        ],
        "display": {"show": False},
    }

    with patch("abraia.runtime.video.Video", Video), \
         patch("abraia.inference.models.detection.Model", Detector), \
         patch("abraia.inference.models.ocr.TextSystem", OCR):
        pipeline = Pipeline.from_dict(config)
        last = pipeline.run()

    assert last.results[0]["ocr"]["text"] == "XYZ"


def test_version_two_pipeline_builder_runs_automatic_second_model_stage():
    class Video:
        frame_rate = 1

        def __init__(self, *_args, **_kwargs):
            pass

        def __iter__(self):
            yield np.zeros((6, 6, 3), dtype=np.uint8)

        def set_display_enabled(self, _enabled):
            pass

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
            {"id": "detect", "type": "model", "input": "frame", "model": {
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

    with patch("abraia.runtime.video.Video", Video), \
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


def test_version_two_pipeline_builder_accepts_legacy_tracker_stage():
    class Video:
        frame_rate = 1

        def __init__(self, *_args, **_kwargs):
            pass

        def __iter__(self):
            yield np.zeros((6, 6, 3), dtype=np.uint8)

        def set_display_enabled(self, _enabled):
            pass

        def close(self):
            pass

    class Detector:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _image, **_kwargs):
            return [{"box": [1, 1, 3, 3], "label": "plate", "score": 0.9}]

        def close(self):
            pass

    class Tracker:
        def __init__(self, **_kwargs):
            pass

        def update(self, results):
            return [dict(result, track_id=7) for result in results]

        def close(self):
            pass

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "model", "type": "model", "input": "frame", "model": {
                "kind": "yolov8", "task": "detection", "uri": "detector.onnx"
            }},
            {"id": "tracker", "type": "tracker", "enabled": True},
        ],
        "display": {"show": False},
    }

    with patch("abraia.runtime.video.Video", Video), \
         patch("abraia.inference.models.detection.Model", Detector), \
         patch("abraia.inference.Tracker", Tracker):
        pipeline = Pipeline.from_dict(config)
        last = pipeline.run()

    assert last.results[0]["track_id"] == 7


def test_pipeline_view_round_trips_composed_steps():
    qt_app()
    view = PipelineView()
    view.set_configuration({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "model", "type": "model", "input": "frame", "model": {
                "kind": "yolov8", "task": "detection"
            }},
            {"id": "read", "type": "model", "model": {
                "kind": "ocr", "task": "recognition"
            }},
        ],
    })

    config = view.configuration()

    assert config["version"] == 2
    assert [step["id"] for step in config["steps"]] == ["model", "read"]
    assert view.validate() == []
    view.close()


def test_pipeline_view_starts_with_the_detector_as_stage_one():
    qt_app()
    view = PipelineView()

    assert view.stages_card is None
    assert view.stage_cards[0].stage_type == "model"
    assert view.stage_cards[0].objectName() == "pipelineCard"
    assert view.stage_cards[0].header.text() == "MODEL"
    assert view.stage_cards[0].model_toggle.isChecked()
    view.stage_cards[0].model_toggle.click()
    assert view.stage_cards[0].content.isHidden()
    view.stage_cards[0].model_toggle.click()
    assert not view.stage_cards[0].content.isHidden()
    assert view.stage_cards[0].to_config()["input"] == "frame"
    assert view.configuration()["steps"][0]["id"] == "model"
    assert view.configuration()["steps"][0]["input"] == "frame"
    assert "id" not in view.stage_cards[0].fields
    assert "input" not in view.stage_cards[0].fields
    assert view.model_card.isHidden()
    view.close()


def test_pipeline_view_keeps_full_model_configuration_on_first_stage():
    qt_app()
    view = PipelineView()
    stage = view.stage_cards[0]

    stage.fields["model_labels"].setText("car, truck")
    stage.fields["model_confidence"].setValue(0.7)
    stage.fields["model_iou"].setValue(0.6)
    view._editor_changed()

    model = view.configuration()["steps"][0]["model"]
    assert model["labels"] == ["car", "truck"]
    assert model["conf_threshold"] == 0.7
    assert model["iou_threshold"] == 0.6
    view.close()


def test_pipeline_view_limits_trackers_and_models():
    qt_app()
    view = PipelineView()

    model_index = view.stage_selector.findData("model")
    tracker_index = view.stage_selector.findData("tracker")
    view.stage_selector.setCurrentIndex(model_index)
    view._add_stage()
    assert sum(card.stage_type == "model" for card in view.stage_cards) == 2
    assert not view.stage_selector.model().item(model_index).isEnabled()
    view._add_stage()
    assert sum(card.stage_type == "model" for card in view.stage_cards) == 2

    view.stage_selector.setCurrentIndex(tracker_index)
    view._add_stage()
    assert sum(card.stage_type == "tracker" for card in view.stage_cards) == 1
    assert not view.stage_selector.model().item(tracker_index).isEnabled()
    view._add_stage()
    assert sum(card.stage_type == "tracker" for card in view.stage_cards) == 1
    view.close()


def test_pipeline_view_filters_second_stage_models_by_region_compatibility():
    qt_app()
    view = PipelineView()

    model_index = view.stage_selector.findData("model")
    view.stage_selector.setCurrentIndex(model_index)
    view._add_stage()

    first_ids = {
        view.stage_cards[0].model_selector.itemData(index).get("id")
        for index in range(view.stage_cards[0].model_selector.count())
    }
    second_ids = {
        view.stage_cards[1].model_selector.itemData(index).get("id")
        for index in range(view.stage_cards[1].model_selector.count())
    }

    assert "yolov8_detection_small" in first_ids
    assert "yolov8_detection_small" not in second_ids
    assert second_ids == {
        "resnet_classification_small",
        "resnet_classification_medium",
        "resnet_classification_large",
        "face_recognition",
        "license_plate_recognition",
        "ocr_recognition",
        "multispectral_classification",
    }
    view.close()


def test_pipeline_view_serializes_a_model_stage_with_automatic_binding():
    qt_app()
    view = PipelineView()
    view.set_configuration({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "model", "type": "model", "input": "frame", "model": {
                "kind": "yolov8", "task": "detection"
            }},
            {
                "id": "read_text",
                "type": "model",
                "input": {"from": "model.results", "labels": ["plate"]},
                "crop": {"padding": 0.1},
                "output": {"attach_to": "model.results", "field": "ocr"},
                "model": {"kind": "ocr", "task": "recognition"},
            },
        ],
    })

    config = view.configuration()

    assert "id" not in view.stage_cards[1].fields
    assert "input" not in view.stage_cards[1].fields
    assert not {
        "labels",
        "min_confidence",
        "padding",
        "attach_to",
        "field",
    }.intersection(view.stage_cards[1].fields)
    assert view.stage_cards[1]._model_rows["uri"][0].isHidden()
    assert "input" not in config["steps"][1]
    assert "crop" not in config["steps"][1]
    assert "output" not in config["steps"][1]
    assert view.validate() == []
    view.close()
