from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from abraia.inference.models.clip_classifier import (
    ClipClassifier,
    preprocess_clip,
)
from abraia.inference.session import OnnxSessionMixin


def test_clip_kind_is_registered_for_classification():
    from abraia.inference.model_config import ModelSpec

    spec = ModelSpec.from_config({
        "kind": "clip",
        "task": "classification",
        "uri": "clip-classifier.onnx",
    })

    assert spec.validation_error() is None


@pytest.mark.parametrize("shape", [(28, 40), (40, 28)])
def test_clip_preprocessing_returns_normalized_nchw(shape):
    height, width = shape
    image = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8))

    tensor = preprocess_clip(image, [1, 3, 224, 224])

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert np.allclose(tensor.mean(axis=(0, 2, 3)), [
        (128 / 255 - 0.48145466) / 0.26862954,
        (128 / 255 - 0.4578275) / 0.26130258,
        (128 / 255 - 0.40821073) / 0.27577711,
    ], atol=1e-5)


def test_clip_preprocessing_accepts_grayscale_images():
    tensor = preprocess_clip(
        np.full((16, 20), 64, dtype=np.uint8), [1, 3, 224, 224]
    )

    assert tensor.shape == (1, 3, 224, 224)
    assert np.isfinite(tensor).all()


def test_clip_classifier_runs_onnx_and_decodes_class(monkeypatch):
    import abraia.inference.models.clip_classifier as clip

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, _outputs, feed):
            assert feed["input"].shape == (1, 3, 224, 224)
            return [np.asarray([[0.0, 2.0]], dtype=np.float32)]

    session = FakeSession()

    def initialize(self, _path, providers=None, accelerator=None):
        self.session = session
        self._resources = SimpleNamespace(close=lambda: None)
        self._closed = False

    monkeypatch.setattr(OnnxSessionMixin, "_init_onnx_session", initialize)
    monkeypatch.setattr(
        clip,
        "load_onnx_model_config",
        lambda _uri: ("model.onnx", {
            "task": "classification",
            "inputShape": [1, 3, 224, 224],
            "classes": ["cat", "dog"],
        }),
    )

    model = ClipClassifier("model.onnx")

    assert model.run(np.zeros((32, 48, 3), dtype=np.uint8))[0]["label"] == "dog"
