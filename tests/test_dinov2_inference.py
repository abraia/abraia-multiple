from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import abraia.inference.models.dinov2 as dinov2
from abraia.inference.models.dinov2 import DinoV2ImageEncoder
from abraia.inference.session import OnnxSessionMixin


class FakeSession:
    def __init__(self, input_shape=(None, 3, 448, 448), output_shape=(None, 768)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_sizes = []

    def get_inputs(self):
        return [SimpleNamespace(name="image", shape=self.input_shape)]

    def get_outputs(self):
        return [SimpleNamespace(name="embedding", shape=self.output_shape)]

    def run(self, _outputs, inputs):
        batch = inputs["image"]
        self.batch_sizes.append(len(batch))
        means = batch.mean(axis=(1, 2, 3), keepdims=False)
        return [np.repeat(means[:, None], 768, axis=1)]


def install_fake_session(monkeypatch, session):
    def init(self, path, providers=None, accelerator=None):
        self.session = session
        self._resources = SimpleNamespace(close=lambda: None)
        self._closed = False
        self.model_path = path
        self.providers = providers
        self.accelerator = accelerator

    monkeypatch.setattr(OnnxSessionMixin, "_init_onnx_session", init)
    monkeypatch.setattr(dinov2, "resolve_model_file", lambda uri: uri)


def test_dinov2_encoder_resolves_onnx_and_batches_embeddings(monkeypatch):
    session = FakeSession()
    install_fake_session(monkeypatch, session)
    monkeypatch.setattr(
        dinov2,
        "resolve_model_file",
        lambda uri: f"resolved/{uri}",
    )

    encoder = DinoV2ImageEncoder(batch_size=2)
    embeddings = encoder.get_image_embeddings([
        np.zeros((12, 18, 3), dtype=np.uint8),
        Image.fromarray(np.full((18, 12, 3), 255, dtype=np.uint8)),
        np.full((12, 18), 128, dtype=np.uint8),
    ])

    assert encoder.model_path == (
        "resolved/multiple/models/dinov2/dinov2_vitb14_reg_518.onnx"
    )
    assert session.batch_sizes == [2, 1]
    assert embeddings.shape == (3, 768)
    assert np.allclose(embeddings[0], 0.0)
    assert np.allclose(embeddings[1], 1.0)
    assert np.allclose(embeddings[2], 128 / 255.0)


def test_dinov2_encoder_rejects_invalid_model_input_shape(monkeypatch):
    session = FakeSession(input_shape=(None, 4, 448, 448))
    install_fake_session(monkeypatch, session)

    with pytest.raises(ValueError, match="RGB channels"):
        DinoV2ImageEncoder()


def test_dinov2_encoder_rejects_bad_output_shape(monkeypatch):
    session = FakeSession(output_shape=(None, 384))
    install_fake_session(monkeypatch, session)
    encoder = DinoV2ImageEncoder()

    with pytest.raises(ValueError, match=r"expected \(1, 384\)"):
        encoder.get_image_embeddings([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_dinov2_encoder_rejects_non_rgb_shapes(monkeypatch):
    install_fake_session(monkeypatch, FakeSession())
    encoder = DinoV2ImageEncoder()

    with pytest.raises(ValueError, match="grayscale, RGB, or RGBA"):
        encoder.get_image_embeddings([np.zeros((8, 8, 2), dtype=np.uint8)])
