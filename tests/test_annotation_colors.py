import pytest

from abraia.utils.annotation_colors import label_color_rgb


@pytest.mark.parametrize(("label", "expected"), [
    ("normal", (2, 136, 0)),
    ("Normal", (2, 136, 0)),
    ("anomaly", (215, 0, 0)),
    ("Anomaly", (215, 0, 0)),
    ("background", (119, 118, 111)),
])
def test_semantic_annotation_colors(label, expected):
    assert label_color_rgb({"label": label}, ["normal", "anomaly"]) == expected
