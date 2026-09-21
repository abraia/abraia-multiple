import json
from types import SimpleNamespace

import numpy as np

from abraia.inference.models.classification import (
    ResNetClassifier,
    preprocess_resnet,
)
from abraia.inference.registry import create_model


def test_resnet_preprocess_matches_training_input_shape():
    image = np.zeros((300, 500, 3), dtype=np.uint8)

    tensor = preprocess_resnet(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_resnet_classifier_runs_exported_model(tmp_path, monkeypatch):
    model_path = tmp_path / "resnet18.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".json").write_text(json.dumps({
        "task": "classification",
        "kind": "resnet",
        "backbone": "resnet18",
        "inputShape": [1, 3, 224, 224],
        "classes": ["cat", "dog"],
    }))

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, inputs):
            assert inputs["input"].shape == (1, 3, 224, 224)
            return [np.array([[0.0, 2.0]], dtype=np.float32)]

        def close(self):
            pass

    monkeypatch.setattr(
        "abraia.inference.session.create_onnx_session",
        lambda path, providers=None, accelerator=None: FakeSession(),
    )
    model = ResNetClassifier(str(model_path))

    result = model.run(np.zeros((300, 500, 3), dtype=np.uint8))

    assert result[0]["label"] == "dog"
    assert result[0]["class_id"] == 1
    model.close()


def test_registry_selects_resnet_adapter_for_training_model(tmp_path, monkeypatch):
    model_path = tmp_path / "resnet50.onnx"
    captured = []

    class FakeResNet:
        def __init__(self, uri):
            captured.append(uri)

    monkeypatch.setattr(
        "abraia.inference.models.classification.ResNetClassifier",
        FakeResNet,
    )

    model = create_model({
        "kind": "resnet",
        "task": "classification",
        "uri": str(model_path),
    })

    assert isinstance(model, FakeResNet)
    assert captured == [str(model_path)]


def test_registry_uses_resnet_for_generic_classification_task(monkeypatch):
    captured = []

    class FakeResNet:
        def __init__(self, uri):
            captured.append(uri)

    monkeypatch.setattr(
        "abraia.inference.models.classification.ResNetClassifier",
        FakeResNet,
    )

    model = create_model({
        "kind": "resnet",
        "task": "classification",
        "uri": "trained_classifier.onnx",
    })

    assert isinstance(model, FakeResNet)
    assert captured == ["trained_classifier.onnx"]


def test_registry_defaults_resnet_to_classification_task(monkeypatch):
    captured = []

    class FakeResNet:
        def __init__(self, uri):
            captured.append(uri)

    monkeypatch.setattr(
        "abraia.inference.models.classification.ResNetClassifier",
        FakeResNet,
    )

    model = create_model({
        "kind": "resnet",
        "uri": "trained_classifier.onnx",
    })

    assert isinstance(model, FakeResNet)
    assert captured == ["trained_classifier.onnx"]
