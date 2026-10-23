"""Canonical task normalization through the public SDK API."""

import pytest

from abraia.inference import normalize_task


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
