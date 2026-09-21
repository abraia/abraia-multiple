from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from abraia.inference.models.grounding_dino import (
    GroundingDINOModel,
    decode_outputs,
    prepare_image,
)


def test_grounding_dino_does_not_use_coreml_provider(tmp_path, monkeypatch):
    model_path = tmp_path / "grounding_dino.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".json").write_text("{}")
    captured = {}

    class FakeSession:
        def get_inputs(self):
            return []

    def init_session(self, path, providers=None, accelerator=None):
        captured["providers"] = providers
        self.session = FakeSession()
        self._resources = SimpleNamespace(close=lambda: None)
        self._closed = False

    monkeypatch.setattr(
        GroundingDINOModel,
        "_load_tokenizer",
        staticmethod(lambda uri: object()),
    )
    monkeypatch.setattr(GroundingDINOModel, "_init_onnx_session", init_session)

    GroundingDINOModel(
        str(model_path),
        providers=[
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    assert captured["providers"] == ["CPUExecutionProvider"]


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "decoded phrase"


def test_grounding_dino_appends_period_to_single_label_prompt():
    model = GroundingDINOModel.__new__(GroundingDINOModel)
    model._closed = False
    model.input_shape = (1, 3, 100, 100)
    model.letterbox = False
    model.force_full_pixel_mask = True
    model.tokenizer = FakeTokenizer()
    captured = {}

    def tokenize(prompt):
        captured["prompt"] = prompt
        return {"input_ids": np.zeros((1, 4), dtype=np.int64)}, None

    model._tokenize = tokenize
    model._build_inputs = lambda image, token_values, pixel_mask: {}
    model.session = type(
        "FakeSession",
        (),
        {"run": lambda self, output_names, inputs: [
            np.full((1, 1, 4), -10, dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ]},
    )()

    model.run(np.zeros((10, 10, 3), dtype=np.uint8), prompt="cat")

    assert captured["prompt"] == "cat."


def test_prepare_grounding_image_letterboxes_and_marks_valid_pixels():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    pixel_values, pixel_mask, scale, padding = prepare_image(
        image, (1, 3, 100, 100)
    )

    assert pixel_values.shape == (1, 3, 100, 100)
    assert pixel_mask.shape == (1, 100, 100)
    assert pixel_mask[:, :25].sum() == 0
    assert pixel_mask[:, 25:75].all()
    assert pixel_mask[:, 75:].sum() == 0
    assert scale == 0.5
    assert padding == (0, 25)


def test_decode_grounding_outputs_converts_boxes_and_phrases():
    logits = np.full((1, 2, 4), -10, dtype=np.float32)
    logits[0, 0, 1] = 10
    logits[0, 1, 2] = 10
    boxes = np.array(
        [[[0.5, 0.5, 0.5, 0.4], [0.5, 0.5, 0.5, 0.4]]],
        dtype=np.float32,
    )

    results = decode_outputs(
        [logits, boxes],
        input_ids=np.array([101, 200, 201, 102]),
        tokenizer=FakeTokenizer(),
        prompt="cat. dog.",
        image_size=(200, 100),
        input_shape=(1, 3, 100, 100),
        box_threshold=0.5,
        text_threshold=0.5,
        transform=(0.5, (0, 25)),
        iou_threshold=None,
        offsets=[(0, 0), (0, 3), (5, 8), (8, 8)],
    )

    assert len(results) == 2
    assert results[0]["label"] == "cat"
    assert results[0]["box"] == [50, 10, 100, 80]
    assert results[1]["label"] == "dog"


def test_pipeline_factory_supports_grounding_dino_kind():
    class FakeGroundingDINOModel:
        def __init__(self, uri, **kwargs):
            self.uri = uri
            self.kwargs = kwargs

    with patch(
        "abraia.inference.models.grounding_dino.GroundingDINOModel",
        FakeGroundingDINOModel,
    ):
        from abraia.inference.registry import create_model

        model = create_model({
            "kind": "grounding_dino",
            "labels": ["cat", "dog"],
            "text_threshold": 0.2,
        })

    assert model.uri == "multiple/models/grounding_dino_tiny.onnx"
    assert model.kwargs == {}


def test_grounding_dino_runtime_options_are_selected_from_pipeline_config():
    from abraia.inference.registry import get_model_run_kwargs

    assert get_model_run_kwargs({
        "kind": "grounding_dino",
        "labels": ["cat"],
        "conf_threshold": 0.4,
        "text_threshold": 0.3,
    }) == {
        "labels": ["cat"],
        "conf_threshold": 0.4,
        "text_threshold": 0.3,
    }
