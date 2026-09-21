import json

import cv2
import numpy as np
import pytest
from unittest.mock import patch
from abraia.inference import Clip
from abraia.inference.session import (
    OnnxSessionBundle,
    ResourceGroup,
    accelerator_from_providers,
)
from abraia.inference.accelerators import onnx_providers
from abraia.runtime import AsyncInferenceRunner, FrameResult
from abraia.inference.postprocess.decoders import (
    postprocess,
    prepare_input,
    preprocess,
    process_output,
    process_pose_output,
)
from abraia.inference.postprocess.common import softmax
from abraia.inference.postprocess.masks import mask_to_polygon
from abraia.inference.vectors import search_vectors
from abraia.inference.models.ocr import TextRecognizer
from abraia.inference.models.sam import SAM
from abraia.inference.service import InferenceService


@pytest.mark.parametrize("providers, expected", [
    (("CUDAExecutionProvider", "CPUExecutionProvider"), "GPU"),
    (("QNNExecutionProvider", "CPUExecutionProvider"), "NPU"),
    (("HailoExecutionProvider", "CPUExecutionProvider"), "HAILO"),
    (("CoreMLExecutionProvider", "CPUExecutionProvider"), "GPU/NPU"),
    (("CPUExecutionProvider",), "CPU"),
])
def test_accelerator_from_providers(providers, expected):
    assert accelerator_from_providers(providers) == expected


def test_gpu_accelerator_falls_back_to_cpu_provider():
    with patch(
        "abraia.inference.accelerators.get_providers",
        return_value=["CPUExecutionProvider"],
    ):
        assert onnx_providers("gpu") == ["CPUExecutionProvider"]


def test_gpu_accelerator_keeps_gpu_and_cpu_providers_when_available():
    with patch(
        "abraia.inference.accelerators.get_providers",
        return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        assert onnx_providers("gpu") == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_softmax_values():
    logits = np.array([0, 10, -10])
    assert np.isclose(np.sum(softmax(logits)), 1)


def test_search_vectors_returns_ranked_results_for_multiple_queries():
    index = [
        {"vector": np.array([1.0, 0.0]), "name": "x"},
        {"vector": np.array([0.0, 1.0]), "name": "y"},
    ]

    indices, scores = search_vectors(
        np.array([[0.9, 0.1], [0.1, 0.9]]), index
    )

    assert indices[0].tolist() == [0]
    assert indices[1].tolist() == [1]
    assert scores[0][0] > 0.9
    assert scores[1][0] > 0.9


def test_async_runner_preserves_backend_elapsed_time():
    def inference(batch, emit, stop_event):
        emit(FrameResult(batch.records[0], [], elapsed_ms=12.5))

    result = next(iter(AsyncInferenceRunner(["frame"], inference, lambda frame: frame)))

    assert result.elapsed_ms == 12.5


def test_async_runner_rejects_missing_frame_results():
    def inference(batch, emit, stop_event):
        return None

    runner = AsyncInferenceRunner(["frame"], inference, lambda frame: frame)

    with pytest.raises(RuntimeError, match="received 0 of 1"):
        list(runner)


def test_async_runner_propagates_frame_errors():
    failure = ValueError("postprocessing failed")

    def inference(batch, emit, stop_event):
        emit(FrameResult(batch.records[0], None, error=failure))

    runner = AsyncInferenceRunner(["frame"], inference, lambda frame: frame)

    with pytest.raises(ValueError, match="postprocessing failed"):
        list(runner)


def test_clip_import():
    assert Clip is not None


def test_mask_to_polygon_accepts_boolean_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    polygon = mask_to_polygon(mask)

    assert len(polygon) >= 3


def test_mask_to_polygon_selects_largest_outer_component_by_area():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:100, 10:100] = 1
    mask[40:60, 40:60] = 0
    cv2.circle(mask, (112, 112), 6, 1, -1)

    polygon = mask_to_polygon(mask)
    xs, ys = zip(*polygon)

    assert (min(xs), max(xs), min(ys), max(ys)) == (10, 99, 10, 99)
    assert len(polygon) > 4


from abraia.runtime.video import load_images

def test_load_images(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.jpg"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)

    images = load_images(str(p))
    assert len(images) == 1
    assert images[0].shape == (10, 10, 3)

    images_dir = load_images(str(d))
    assert len(images_dir) == 1
    assert images_dir[0].shape == (10, 10, 3)


def test_inference_service_reuses_and_closes_backend_sessions():
    instances = []

    class FakeModel:
        def __init__(self, uri):
            self.uri = uri
            self.closed = False
            instances.append(self)

        def run(self, image):
            return self.uri, image.shape

        def close(self):
            self.closed = True

    service = InferenceService({"fake": FakeModel})
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    assert service.run("model-a", image, backend="fake") == ("model-a", (4, 5, 3))
    assert service.run("model-a", image, backend="fake") == ("model-a", (4, 5, 3))
    assert len(instances) == 1

    service.close()
    assert instances[0].closed


def test_classifier_preprocess_returns_fixed_input_shape_for_wide_images():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    assert preprocess(image, (224, 224)).shape == (1, 3, 224, 224)


def test_detection_decoder_reverses_centered_letterbox_transform():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, scale, padding = prepare_input(
        image, (1, 3, 100, 100), return_transform=True
    )
    assert tensor.shape == (1, 3, 100, 100)
    assert scale == 0.5
    assert padding == (0, 25)

    output = np.zeros((1, 5, 1), dtype=np.float32)
    output[0, :5, 0] = [50, 50, 20, 20, 0.95]
    result = process_output(
        [output],
        size=(200, 100),
        shape=(1, 3, 100, 100),
        classes=["object"],
        conf_threshold=0.5,
        transform=(scale, padding),
    )
    assert result[0]["box"] == [80, 30, 40, 40]


def test_classification_decoder_supports_top_k():
    result = postprocess(
        [np.array([[0.0, 2.0, 1.0]], dtype=np.float32)],
        ["a", "b", "c"],
        top_k=2,
    )
    assert [item["label"] for item in result] == ["b", "c"]


def test_onnx_session_bundle_closes_all_sessions():
    from unittest.mock import patch

    class Session:
        def __init__(self, provider):
            self.provider = provider
            self.closed = False

        def get_providers(self):
            return [self.provider]

        def close(self):
            self.closed = True

    sessions = [Session("CPUExecutionProvider"), Session("CUDAExecutionProvider")]
    with patch(
        "abraia.inference.session.create_onnx_session",
        side_effect=sessions,
    ):
        bundle = OnnxSessionBundle(["a.onnx", "b.onnx"])
    assert bundle.execution_providers == (
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    )
    assert bundle.accelerator == "GPU"
    bundle.close()
    assert all(session.closed for session in sessions)


def test_resource_group_closes_in_reverse_order_and_only_once():
    closed = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    first = Resource("first")
    second = Resource("second")
    group = ResourceGroup()
    assert group.add(first) is first
    assert group.add(second) is second
    group.add(first)

    group.close()
    group.close()

    assert closed == ["second", "first"]


def test_sam_cache_uses_image_content_not_object_identity():
    class FakeSAM:
        def __init__(self):
            self.encode_calls = 0

        def encode(self, image):
            self.encode_calls += 1

        def predict(self, image, prompt):
            return np.zeros(image.shape[:2], dtype=np.uint8)

        def close(self):
            pass

    service = InferenceService()
    service._sam = FakeSAM()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    service.sam_predict(image, [[1, 1, 1]])
    service.sam_predict(image.copy(), [[2, 2, 1]])
    assert service._sam.encode_calls == 1
    image[0, 0, 0] = 1
    service.sam_predict(image, [[2, 2, 1]])
    assert service._sam.encode_calls == 2


def test_sam_box_prompt_uses_xywh_coordinates():
    class FakeSAM:
        def encode(self, image):
            pass

        def predict(self, image, prompt):
            self.prompt = json.loads(prompt)
            return np.zeros(image.shape[:2], dtype=np.uint8)

        def close(self):
            pass

    service = InferenceService()
    service._sam = FakeSAM()
    image = np.zeros((8, 10, 3), dtype=np.uint8)

    service.sam_predict_box(image, [2, 3, 4, 5])

    assert service._sam.prompt == [{
        "type": "rectangle",
        "data": [2.0, 3.0, 6.0, 8.0],
    }]


def test_grounding_dino_sam_prediction_returns_polygons():
    service = InferenceService()
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[2:8, 4:11] = 1
    service.grounding_dino_predict = lambda _image, _prompt: [
        {"label": "cat", "box": [4, 2, 7, 6], "score": 0.9},
    ]
    service.sam_predict_box = lambda _image, _box: mask

    result = service.grounding_dino_sam_predict(image, "cat")

    assert len(result) == 1
    assert result[0]["label"] == "cat"
    assert result[0]["score"] == 0.9
    assert result[0]["polygon"]
    assert "box" not in result[0]


def test_pose_output_decodes_boxes_and_keypoints():
    output = np.zeros((1, 56, 1), dtype=np.float32)
    output[0, :5, 0] = [50, 60, 20, 30, 0.95]
    output[0, 5:, 0] = np.tile([10, 20, 0.9], 17)

    results = process_pose_output(
        [output],
        size=(100, 100),
        shape=(1, 3, 100, 100),
        classes=["person"],
        conf_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0]["box"] == [40, 45, 20, 30]
    assert results[0]["keypoints"].shape == (17, 2)
    assert results[0]["joint_scores"].shape == (17,)
    assert np.allclose(results[0]["keypoints"][0], [10, 20])


def test_text_recognizer_processes_all_recognition_batches():
    class Input:
        name = "input"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, inputs):
            batch = next(iter(inputs.values())).shape[0]
            return [np.zeros((batch, 1, 2), dtype=np.float32)]

    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.rec_image_shape = [3, 32, 320]
    recognizer.rec_batch_num = 6
    recognizer.limited_max_width = 1280
    recognizer.limited_min_width = 16
    recognizer.session = Session()
    recognizer.resize_norm_img = lambda image, ratio: np.zeros(
        (3, 32, 320), dtype=np.float32
    )
    recognizer.postprocess_op = lambda outputs: [
        ("decoded", 1.0)
    ] * len(outputs)

    images = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(7)]

    result = recognizer(images)

    assert len(result) == 7
    assert all(text == "decoded" for text, _score in result)


def test_sam_close_releases_both_sessions():
    class Session:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    sam = SAM.__new__(SAM)
    sam.encoder = Session()
    sam.decoder = Session()
    sam.image_embedding = object()
    sam._closed = False

    sam.close()

    assert sam.encoder is None
    assert sam.decoder is None
    assert sam.image_embedding is None
    assert sam._closed
