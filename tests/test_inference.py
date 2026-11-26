import json
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from unittest.mock import Mock, patch
from abraia.inference.session import (
    OnnxSessionBundle,
    ResourceGroup,
    accelerator_from_providers,
)
from abraia.inference.accelerators import onnx_providers
from abraia.runtime.inference import AsyncInferenceRunner, FrameResult
from abraia.inference.models.classification import postprocess, preprocess
from abraia.inference.models.detection import (
    prepare_input,
    process_output,
    process_pose_output,
    process_rtmdet_output,
)
from abraia.inference.geometry import (
    mask_to_box, mask_to_polygon, mask_to_polygons, normalize, search_vectors, softmax,
)
from abraia.inference.models.ocr import BaseRecLabelDecode, TextRecognizer, TextSystem
from abraia.inference.models.sam import SAM
from abraia.inference.models.classification import ResNetClassifier, preprocess_resnet
from abraia.inference import load_model
from abraia.inference.registry import create_model


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


@pytest.mark.parametrize("method", ["search_similar", "search_text"])
def test_image_search_returns_and_displays_ranked_matches(monkeypatch, method):
    from abraia.inference import search as search_module

    client = Mock()
    client.load_json.return_value = [
        {"path": "project/other.jpg", "vector": [0.0, 1.0]},
        {"path": "project/match.jpg", "vector": [1.0, 0.0]},
    ]
    client.load_image.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    model = Mock()
    model.get_image_embeddings.return_value = np.array([[1.0, 0.0]])
    model.get_text_embeddings.return_value = np.array([[1.0, 0.0]])
    display = Mock()
    monkeypatch.setattr(search_module, "Abraia", lambda: client)
    monkeypatch.setattr(search_module, "Clip", lambda: model)
    monkeypatch.setattr(search_module, "show_image", display)

    search = search_module.ImageSearch("project")
    query = "person" if method == "search_text" else np.zeros((2, 2, 3), dtype=np.uint8)
    results = getattr(search, method)(query, max_results=1)

    assert results == [{"path": "project/match.jpg", "score": 1.0}]
    client.load_image.assert_called_once_with("project/match.jpg")
    display.assert_called_once_with(client.load_image.return_value)
    encoder = model.get_text_embeddings if method == "search_text" else model.get_image_embeddings
    encoder.assert_called_once_with([query])


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


def test_live_async_runner_keeps_newest_result_without_waiting_for_consumer():
    first_consumed = Event()
    completed = Event()
    submitted = []

    def source():
        yield 0
        assert first_consumed.wait(2)
        yield from range(1, 4)

    pending = {}

    def inference(batch, emit, stop_event):
        record, = batch.records
        submitted.append(record.index)
        if record.index in (0, 1):
            emit(FrameResult(record, record.frame))
        else:
            pending[record.index] = record
            if record.index == 3:
                for index in (3, 2):
                    emit(FrameResult(pending[index], index))
        if record.index == 3:
            completed.set()

    runner = AsyncInferenceRunner(
        source(), inference, lambda frame: frame, live_max_in_flight_frames=2,
    )
    frames = iter(runner)
    try:
        assert next(frames).record.index == 0
        first_consumed.set()
        assert completed.wait(2)
        assert [result.record.index for result in frames] == [3]
        assert submitted == [0, 1, 2, 3]
    finally:
        first_consumed.set()
        frames.close()


def test_live_async_runner_stops_a_blocked_capture_on_close():
    stopped = Event()

    class Source:
        def __iter__(self):
            yield 0
            assert stopped.wait(2)

        def stop(self):
            stopped.set()

    def inference(batch, emit, stop_event):
        emit(FrameResult(batch.records[0], []))

    runner = AsyncInferenceRunner(
        Source(), inference, lambda frame: frame, live_max_in_flight_frames=1,
    )
    frames = iter(runner)
    assert next(frames).record.index == 0
    frames.close()
    assert stopped.is_set()


@pytest.mark.parametrize("failure_stage", ["capture", "inference"])
def test_live_async_runner_propagates_errors(failure_stage):
    def source():
        if failure_stage == "capture":
            raise ValueError("live capture failed")
        yield 0

    def inference(batch, emit, stop_event):
        emit(FrameResult(batch.records[0], None, error=ValueError("live inference failed")))

    runner = AsyncInferenceRunner(
        source(), inference, lambda frame: frame, live_max_in_flight_frames=1,
    )
    with pytest.raises(ValueError, match=f"live {failure_stage} failed"):
        list(runner)


def test_ocr_decoder_returns_zero_confidence_for_blank_text():
    decoder = BaseRecLabelDecode.__new__(BaseRecLabelDecode)

    result = decoder.decode(
        np.array([[0, 0]], dtype=np.int64),
        np.array([[0.1, 0.2]], dtype=np.float32),
    )

    assert result == [("", 0.0)]


def test_mask_to_polygon_accepts_boolean_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    polygon = mask_to_polygon(mask)

    assert len(polygon) >= 3


def test_mask_to_polygon_does_not_mutate_mask_when_filling_tiny_holes():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 1
    mask[15:18, 15:18] = 0
    original = mask.copy()

    polygon = mask_to_polygon(mask)

    assert len(polygon) == 4
    assert np.array_equal(mask, original)


def test_mask_to_polygon_hole_cleanup_scales_with_object_area():
    for size, object_bounds, hole_bounds in (
        (120, (5, 105), (50, 53)),
        (320, (5, 305), (150, 159)),
    ):
        mask = np.zeros((size, size), dtype=np.uint8)
        left, right = object_bounds
        mask[left:right, left:right] = 1
        hole_left, hole_right = hole_bounds
        mask[hole_left:hole_right, hole_left:hole_right] = 0

        assert len(mask_to_polygon(mask)) == 4


def test_mask_to_polygon_selects_largest_outer_component_by_area():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:100, 10:100] = 1
    mask[40:60, 40:60] = 0
    cv2.circle(mask, (112, 112), 6, 1, -1)

    polygon = mask_to_polygon(mask)
    xs, ys = zip(*polygon)

    assert (min(xs), max(xs), min(ys), max(ys)) == (10, 99, 10, 99)
    assert len(polygon) > 4
    assert mask_to_box(mask) == (10, 10, 90, 90)
    assert mask_to_box(np.zeros_like(mask)) == (0, 0, 0, 0)


def test_mask_to_polygons_returns_all_objects_and_filters_small_components():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    mask[35:55, 35:55] = 1
    mask[60:62, 60:62] = 1

    polygons = mask_to_polygons(mask, min_area=16)

    assert len(polygons) == 2
    assert len(mask_to_polygons(mask)) == 3


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


@pytest.mark.parametrize("task", ["detection", "segmentation", "pose"])
def test_yolo_candidate_filter_preserves_decoder_results(task):
    shape, size, transform = (1, 3, 100, 100), (200, 100), (0.5, (0, 25))
    rows = np.array([
        [25, 45, 20, 20, .75, .75],  # Tied classes choose the first one.
        [75, 55, 20, 20, .1, .75],
        [25, 45, 20, 20, .70, .1],  # Suppressed by the first overlapping box.
        [50, 45, 20, 20, np.nextafter(np.float32(.5), np.float32(0)), .1],
        [50, 65, 20, 20, .5, .1],
    ], dtype=np.float32)
    output = [rows.T[None]]
    decode = process_output
    if task == "segmentation":
        output = [
            np.column_stack((rows, np.ones(len(rows), dtype=np.float32))).T[None],
            np.ones((1, 1, 10, 10), dtype=np.float32),
        ]
    elif task == "pose":
        decode = process_pose_output
        output = [np.column_stack((rows, rows[:, :2], np.full(len(rows), .8))).astype(np.float32).T[None]]

    kwargs = dict(size=size, shape=shape, classes=["person", "car"], transform=transform)
    results = decode(output, conf_threshold=.5, **kwargs)
    assert [(r["label"], r["box"]) for r in results] == [
        ("car", [130, 40, 40, 40]),
        ("person", [30, 20, 40, 40]),
        ("person", [80, 60, 40, 40]),
    ]
    assert [r["score"] for r in results] == [.75, .75, .5]
    assert [r["label"] for r in decode(output, conf_threshold=.5, labels=["car"], **kwargs)] == ["car"]
    assert decode(output, conf_threshold=np.nextafter(.75, 1.0), **kwargs) == []
    assert decode([output[0][:, :, :0], *output[1:]], conf_threshold=.5, **kwargs) == []
    if task == "segmentation":
        for result in results:
            np.testing.assert_array_equal(result["mask"], np.ones((40, 40), dtype=np.uint8))
    elif task == "pose":
        for result in results:
            x, y, _, _ = result["box"]
            np.testing.assert_array_equal(result["keypoints"], [[x + 20, y + 20]])
            np.testing.assert_array_equal(result["joint_scores"], [float(np.float32(.8))])


def test_rtmdet_decoder_accepts_model_run_arguments_and_filters_labels():
    detections = np.array([[[10, 20, 50, 60, 0.9], [20, 30, 40, 50, 0.8]]])
    class_ids = np.array([[0, 1]], dtype=np.int64)

    result = process_rtmdet_output(
        [detections, class_ids],
        size=(100, 100),
        shape=(1, 3, 100, 100),
        classes=["person", "car"],
        output_names=["dets", "labels"],
        conf_threshold=0.25,
        iou_threshold=0.7,
        approx=0.001,
        labels=["car"],
        transform=(1.0, (0, 0)),
    )

    assert [item["label"] for item in result] == ["car"]
    assert result[0]["box"] == [20, 30, 20, 20]


def test_onnx_model_uses_kind_for_preprocessing_and_decoding(tmp_path, monkeypatch):
    from abraia.inference.models.detection import Model

    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"onnx")
    image = np.full((10, 10, 3), 128, dtype=np.uint8)

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def get_outputs(self):
            return [SimpleNamespace(name=name) for name in ("dets", "labels")[:len(outputs)]]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _names, inputs):
            self.input_tensor = inputs["input"]
            return outputs

        def close(self):
            pass

    session = FakeSession()
    monkeypatch.setattr(
        "abraia.inference.session.create_onnx_session",
        lambda *_args, **_kwargs: session,
    )
    for kind, outputs in (
        ("rtmdet", [np.array([[[1, 1, 5, 5, 0.9]]]), np.array([[0]])]),
        ("yolo9", [np.array([[[1, 1, 5, 5, 0.9, 0]]])]),
    ):
        for explicit_kind in (None, kind):
            model_path.with_suffix(".json").write_text(json.dumps({
                "kind": kind if explicit_kind is None else "yolov8",
                "task": "detection",
                "classes": ["tomato"],
                "inputShape": [1, 3, 10, 10],
            }))
            with Model(str(model_path), kind=explicit_kind) as model:
                results = model.run(image)
                assert model.kind == kind
                assert results[0]["label"] == "tomato"
                assert results[0]["box"] == [1, 1, 4, 4]
                expected_pixel = (
                    (128 - np.array([103.53, 116.28, 123.675]))
                    / np.array([57.375, 57.12, 58.395])
                    if kind == "rtmdet" else np.full(3, 128 / 255)
                )
                np.testing.assert_allclose(
                    session.input_tensor[0, :, 0, 0], expected_pixel, rtol=1e-6,
                )


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


def test_text_system_batches_recognition_across_region_images():
    system = TextSystem.__new__(TextSystem)
    system.drop_score = 0.5
    boxes = [
        np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=np.float32),
        np.array([[1, 1], [5, 1], [5, 3], [1, 3]], dtype=np.float32),
    ]
    system._detect_text_regions = lambda image: ([boxes[image]], [image])
    recognized_images = []
    system.text_classifier = lambda images: images
    system.text_recognizer = lambda images: (
        recognized_images.extend(images) or [("one", 0.9), ("two", 0.8)]
    )

    result = system.run_batch([0, 1])

    assert recognized_images == [0, 1]
    assert [item[0]["text"] for item in result] == ["one", "two"]
    assert result[0][0]["box"] == [0, 0, 4, 2]


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


def test_sam_reencodes_only_when_image_content_changes(monkeypatch):
    from abraia.inference.models import sam as module

    encoder = Mock()
    encoder.get_inputs.return_value = [SimpleNamespace(name="image")]
    encoder.run.side_effect = lambda _names, inputs: [inputs["image"].mean()]
    decoder = Mock()
    decoder.run.side_effect = lambda _names, inputs: (
        np.full((1, 1, 4, 4), inputs["image_embeddings"] - 100, np.float32),
        np.array([[1]]), None,
    )
    bundle = Mock(sessions=(encoder, decoder))
    monkeypatch.setattr(module, "download_file", lambda path: path)
    monkeypatch.setattr(module, "OnnxSessionBundle", lambda *args, **kwargs: bundle)
    image = np.zeros((4, 4, 3), np.uint8)
    prompt = '[{"type":"point","data":[1,1],"label":1}]'
    with SAM() as model:
        model.input_size = (4, 4)
        assert not model.predict(image, prompt).any()
        assert not model.predict(image.copy(), prompt).any()
        assert encoder.run.call_count == 1
        image[:] = 255
        assert (model.predict(image, prompt) == 255).all()
        assert encoder.run.call_count == 2
        assert not model.predict(np.zeros_like(image), prompt).any()
        assert encoder.run.call_count == 3
    assert model._encoded_image is None
    bundle.close.assert_called_once()


def test_shared_normalization_matches_reference_and_leaves_input_intact():
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    expected = (image / 255 - [.485, .456, .406]) / [.229, .224, .225]
    values = normalize(image)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, expected, atol=2e-7)
    np.testing.assert_array_equal(image.ravel(), np.arange(24))


def test_grounding_dino_adapter_uses_rgb_without_changing_bgr_helper(monkeypatch):
    from abraia.inference.models import grounding_dino as module

    image = np.full((4, 8, 3), (255, 0, 0), np.uint8)
    model = module.GroundingDINOModel.__new__(module.GroundingDINOModel)
    model._closed = False
    model.input_shape = (1, 3, 8, 8)
    model.letterbox = True
    model.force_full_pixel_mask = False
    model._tokenize = lambda _prompt: ({"input_ids": [[0]]}, None)
    model._build_inputs = lambda image, _tokens, mask: {"image": image, "mask": mask}
    model.tokenizer = object()
    model.session = Mock()
    monkeypatch.setattr(module, "decode_outputs", Mock(return_value=[]))

    assert model.run(image, labels=["object"]) == []
    inputs = model.session.run.call_args.args[1]
    rgb, mask, scale, padding = module.prepare_image(image, model.input_shape, color_order="RGB")
    bgr, *_ = module.prepare_image(image[..., ::-1], model.input_shape)
    np.testing.assert_array_equal(inputs["image"], rgb)
    np.testing.assert_array_equal(inputs["mask"], mask)
    np.testing.assert_array_equal(rgb, bgr)
    assert rgb[0, 0, 2, 0] > 0 and rgb[0, 2, 2, 0] < 0
    assert (scale, padding) == (1, (0, 2))
    assert not mask[:, :2].any() and mask[:, 2:6].all()


def test_resnet_preprocess_matches_training_input_shape():
    image = np.zeros((300, 500, 3), dtype=np.uint8)

    tensor = preprocess_resnet(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_grounding_dino_disables_onnx_mem_pattern(tmp_path, monkeypatch):
    from abraia.inference.models import grounding_dino

    model_path = tmp_path / "grounding_dino.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".json").write_text(json.dumps({
        "inputShape": [1, 3, 800, 800],
    }))
    tokenizer_path = tmp_path / "vocab.txt"
    tokenizer_path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")
    captured = {}

    class FakeSession:
        def get_inputs(self):
            return []

    def initialize_session(self, path, providers=None, accelerator=None,
                           session_options=None):
        captured["session_options"] = session_options
        self.session = FakeSession()

    monkeypatch.setattr(
        grounding_dino.OnnxSessionMixin, "_init_onnx_session",
        initialize_session,
    )

    grounding_dino.GroundingDINOModel(
        str(model_path), tokenizer_uri=str(tokenizer_path)
    )

    assert captured["session_options"].enable_mem_pattern is False


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
        Mock(return_value=FakeSession()),
    )
    model = ResNetClassifier(str(model_path))

    result = model.run(np.zeros((300, 500, 3), dtype=np.uint8))

    assert result[0]["label"] == "dog"
    assert result[0]["class_id"] == 1
    model.close()


def test_resnet_factories_accept_explicit_and_default_classification_tasks(tmp_path, monkeypatch):
    class FakeResNet:
        def __init__(self, uri):
            self.uri = uri

    monkeypatch.setattr(
        "abraia.inference.models.classification.ResNetClassifier",
        FakeResNet,
    )
    for factory, config in (
        (create_model, {"kind": "resnet", "task": "classification", "uri": str(tmp_path / "resnet50.onnx")}),
        (create_model, {"kind": "resnet", "task": "classification", "uri": "trained_classifier.onnx"}),
        (create_model, {"kind": "resnet", "uri": "trained_classifier.onnx"}),
        (load_model, {"kind": "resnet", "uri": "trained_classifier.onnx"}),
    ):
        model = factory(config)
        assert isinstance(model, FakeResNet)
        assert model.uri == config["uri"], config
