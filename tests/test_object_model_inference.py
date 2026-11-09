import sys

import numpy as np
import pytest

import abraia.inference.models.object_models as object_models
import abraia.inference.session as inference_session
import abraia.runtime.pipeline as runtime_pipeline
from abraia.inference.models.object_models import (
    DFineONNXModel,
    _decode_rfdetr_outputs,
)
from abraia.runtime.pipeline import Pipeline


def test_rfdetr_decodes_sparse_coco_logits_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    classes = [f"class-{index}" for index in range(80)]
    boxes = np.asarray([[[0.5, 0.5, 0.4, 0.4]]], dtype=np.float32)
    logits = np.full((1, 1, 91), -20.0, dtype=np.float32)
    logits[0, 0, 12] = 20.0
    logits[0, 0, 90] = 10.0

    result = _decode_rfdetr_outputs(
        boxes, logits, None, (100, 200), classes, 0.5, None
    )

    assert [item["label"] for item in result] == ["class-79"]
    assert np.allclose(result[0]["box"], [60.0, 30.0, 80.0, 40.0])


def test_dfine_onnx_detection_and_masks_run_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    class Session:
        def run(self, _outputs, inputs):
            self.input = inputs["image"]
            return [
                np.asarray([[0, 1]], dtype=np.int64),
                np.asarray([[[0, 0, 4, 4], [0, 0, 4, 4]]], dtype=np.float32),
                np.asarray([[0.9, 0.8]], dtype=np.float32),
                np.ones((1, 2, 2, 2), dtype=np.float32),
            ]

    model = DFineONNXModel.__new__(DFineONNXModel)
    model._closed = False
    model.session = Session()
    model.input_name = "image"
    model.input_shape = [1, 3, 4, 4]
    model.labels_index = 0
    model.boxes_index = 1
    model.scores_index = 2
    model.masks_index = 3
    model.classes = ["person", "cat"]

    result = model.run(
        np.full((4, 4, 3), 255, dtype=np.uint8),
        conf_threshold=0.5,
        iou_threshold=0.7,
    )

    assert [item["label"] for item in result] == ["person", "cat"]
    assert np.allclose([item["score"] for item in result], [0.9, 0.8])
    assert all(item["mask"].shape == (4, 4) for item in result)
    assert all(np.all(item["mask"] == 1) for item in result)
    assert np.all(model.session.input == 1.0)


@pytest.mark.parametrize("kind", ["rfdetr", "dfine"])
@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_object_models_run_in_pipeline_without_torch(kind, task, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    metadata = {
        "task": task,
        "classes": ["person"],
        "inputShape": [1, 3, 4, 4],
    }
    monkeypatch.setattr(
        object_models,
        "load_onnx_model_config",
        lambda _uri: ("memory.onnx", metadata),
    )

    class Session:
        def get_inputs(self):
            return [type("Input", (), {"name": "image"})()]

        def get_outputs(self):
            names = (
                ["dets", "labels"]
                if kind == "rfdetr"
                else ["labels", "boxes", "scores"]
            )
            if task == "segmentation":
                names.append("masks")
            return [type("Output", (), {"name": name})() for name in names]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, _inputs):
            if kind == "rfdetr":
                outputs = [
                    np.asarray([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32),
                    np.asarray([[[20.0]]], dtype=np.float32),
                ]
                if task == "segmentation":
                    outputs.append(np.ones((1, 1, 2, 2), dtype=np.float32))
                return outputs
            outputs = [
                np.asarray([[0]], dtype=np.int64),
                np.asarray([[[1, 1, 3, 3]]], dtype=np.float32),
                np.asarray([[0.9]], dtype=np.float32),
            ]
            if task == "segmentation":
                outputs.append(np.ones((1, 1, 2, 2), dtype=np.float32))
            return outputs

    monkeypatch.setattr(
        inference_session,
        "create_onnx_session",
        lambda *_args, **_kwargs: Session(),
    )

    class Source:
        frame_rate = 30

        def __iter__(self):
            return iter([np.full((4, 4, 3), 255, dtype=np.uint8)])

        def set_display_enabled(self, _enabled):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        runtime_pipeline, "create_source", lambda *_args: Source()
    )
    configuration = {
        "version": 2,
        "source": {"type": "camera", "src": "0"},
        "steps": [{
            "type": "model",
            "model": {
                "kind": kind,
                "task": task,
                "uri": "memory.onnx",
                "conf_threshold": 0.5,
                "labels": [],
            },
        }],
        "display": {"show": False},
    }

    with Pipeline.from_dict(configuration, accelerator="cpu") as pipeline:
        context = pipeline.run()

    assert [item["label"] for item in context.results] == ["person"]
    expected_score = 0.9 if kind == "dfine" else 1.0
    assert np.isclose(context.results[0]["score"], expected_score)
    if task == "segmentation":
        assert context.results[0]["box"] == [1.0, 1.0, 2.0, 2.0]
        assert context.results[0]["mask"].shape == (2, 2)
        assert np.all(context.results[0]["mask"] == 1)
