"""Canonical task normalization through the public SDK API."""

import pytest

from abraia.inference import normalize_task
from abraia.inference.model_config import model_default_task


@pytest.mark.parametrize("value, expected", [
    (" Classify ", "classification"),
    ("DETECT", "detection"),
    ("segment", "segmentation"),
    ("recognize", "recognition"),
    ("classification", "classification"),
    ("detection", "detection"),
    ("segmentation", "segmentation"),
    ("recognition", "recognition"),
    ("pose", "pose"),
    ("custom", "custom"),
    (None, ""),
])
def test_normalize_task_preserves_canonical_names_and_aliases(value, expected):
    assert normalize_task(value) == expected


def test_normalize_task_applies_default_only_for_none():
    assert normalize_task(None, default="detect") == "detection"
    assert normalize_task("", default="detect") == ""


@pytest.mark.parametrize(
    "kind, uri, expected",
    [
        ("yolov8", "", "detection"),
        ("rtmdet", "multiple/models/rtmdet_ins_small.onnx", "segmentation"),
        ("resnet", "", "classification"),
        ("clip", "", "classification"),
        ("anomaly", "", "anomaly"),
    ],
)
def test_model_default_task_is_centralized(kind, uri, expected):
    assert model_default_task(kind, uri) == expected
