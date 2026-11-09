import numpy as np
import pytest

from abraia.inference.hailo.yolov9 import decode_yolov9_outputs
from abraia.inference.model_config import ModelSpec
from abraia.inference.registry import get_model_run_kwargs


def test_yolov9_hailo_decoder_maps_raw_heads_to_original_boxes():
    class_logits = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
    class_logits[0, 0, 1, 1] = 10.0
    box_logits = np.full((1, 16, 4, 4), -10.0, dtype=np.float32)
    for coordinate in range(4):
        box_logits[0, coordinate * 4 + 1, 1, 1] = 10.0

    metadata = {
        "classes": ["object"],
        "inputShape": [1, 3, 32, 32],
        "regMax": 4,
        "strides": [8],
        "outputRoles": {
            "classes_8": "classes_8",
            "box_distributions_8": "box_distributions_8",
        },
    }
    results = decode_yolov9_outputs(
        {
            "classes_8": class_logits,
            "box_distributions_8": box_logits,
        },
        metadata,
        (32, 32, 3),
        conf_threshold=0.5,
        iou_threshold=0.7,
    )

    assert len(results) == 1
    assert results[0]["label"] == "object"
    assert results[0]["class_id"] == 0
    assert results[0]["score"] > 0.99
    assert results[0]["box"] == [4, 4, 16, 16]


def test_yolov9_hailo_decoder_rejects_mismatched_head_dimensions():
    metadata = {
        "classes": ["object"],
        "inputShape": [1, 3, 32, 32],
        "regMax": 4,
        "strides": [8],
        "outputRoles": {
            "classes_8": "classes_8",
            "box_distributions_8": "box_distributions_8",
        },
    }

    with pytest.raises(ValueError, match="has shape"):
        decode_yolov9_outputs(
            {
                "classes_8": np.zeros((1, 1, 3, 3), dtype=np.float32),
                "box_distributions_8": np.zeros(
                    (1, 16, 4, 4), dtype=np.float32
                ),
            },
            metadata,
            (32, 32, 3),
        )


@pytest.mark.parametrize(
    "labels, expected",
    [(["person"], {"labels": ["person"]}), ([], {})],
)
def test_yolov9_hailo_runtime_passes_only_nonempty_label_filters(labels, expected):
    config = {
        "kind": "yolo9",
        "task": "detection",
        "uri": "model.hef",
        "labels": labels,
    }
    spec = ModelSpec.from_config(config)

    assert get_model_run_kwargs(config, spec) == expected
