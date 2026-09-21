import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from abraia.inference.hailo import device, postprocess
from abraia.inference.hailo.models import resolve_model_type


def test_resolve_model_type_detects_yolov8_segmentation_models():
    assert resolve_model_type(None, "yolov8n_seg", "segmentation") == "v8"


def test_resolve_model_type_keeps_yolov5_segmentation_default():
    assert resolve_model_type(None, "yolov5m_seg_with_nms", "segmentation") == "v5"


def test_resolve_model_type_honors_explicit_value():
    assert resolve_model_type("v5", "yolov8n_seg", "segmentation") == "v5"


def test_model_type_from_onnx_uri_detects_supported_yolo_families():
    from abraia.inference.hailo.models import model_type_from_onnx_uri

    assert model_type_from_onnx_uri("multiple/models/yolov5n-seg.onnx") == "v5"
    assert model_type_from_onnx_uri("multiple/models/yolov8n-seg.onnx") == "v8"
    assert model_type_from_onnx_uri("multiple/models/yolo11n-segment.onnx") == "v8"
    assert model_type_from_onnx_uri("project/custom.onnx") is None


def test_hailo_device_adapter_parses_cli_architecture():
    result = SimpleNamespace(
        returncode=0,
        stdout="Device architecture: HAILO8L",
        stderr="",
    )
    with patch.object(device.subprocess, "run", return_value=result) as run:
        assert device.detect_hailo_arch() == device.HAILO8L_ARCH

    run.assert_called_once_with(
        ("hailortcli", "fw-control", "identify"),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_hailo_device_adapter_accepts_hyphenated_cli_architecture():
    result = SimpleNamespace(
        returncode=0,
        stdout="Device Architecture: HAILO-8",
        stderr="",
    )
    with patch.object(device.subprocess, "run", return_value=result):
        assert device.detect_hailo_arch() == device.HAILO8_ARCH


def test_paired_hailo_uri_prefers_target_specific_compilation(tmp_path):
    from abraia.inference.accelerators import paired_hailo_uri

    onnx_path = tmp_path / "yolov8n_v2.onnx"
    onnx_path.write_bytes(b"onnx")
    target_hef = tmp_path / "yolov8n_v2_hailo8l.hef"
    target_hef.write_bytes(b"hef")

    assert paired_hailo_uri(onnx_path, "detection", "hailo8l") == str(target_hef)


def test_paired_hailo_uri_supports_uploaded_multiple_project_paths():
    from abraia.inference.accelerators import paired_hailo_uri

    assert paired_hailo_uri(
        "multiple/tomato/yolov8n.onnx", "detection", "hailo8"
    ) == "multiple/tomato/yolov8n_hailo8.hef"


def test_hailo_availability_checks_global_remote_paths():
    from abraia.inference.accelerators import hailo_model_available
    from abraia.utils.remote import ARTIFACT_RESOLVER

    config = {"model": {"uri": "multiple/models/yolov8n_hailo8.hef"}}
    with patch.object(ARTIFACT_RESOLVER, "remote_available", return_value=False):
        assert not hailo_model_available(config, "hailo8")


def test_hailo_pairing_recognizes_underscore_segmentation_names():
    from abraia.inference.registry import _hailo_pair_task

    assert _hailo_pair_task("onnx", "detection", "yolov8n_seg.onnx") == "segmentation"


def test_hailo_metadata_supplies_task_and_labels_for_exported_bundle(tmp_path):
    from abraia.inference.hailo import models

    bundle = tmp_path / "custom_hailo_model"
    bundle.mkdir()
    (bundle / "custom.hef").write_bytes(b"hef")
    (bundle / "metadata.yaml").write_text(
        "task: segment\nnames:\n  0: tomato\n  1: leaf\n"
    )

    metadata = models.load_hailo_metadata(bundle)

    assert models.labels_from_metadata(metadata) == ["tomato", "leaf"]
    assert models.resolve_model_type(None, bundle, "segmentation") == "v8"


def test_hailo_format_order_recognizes_native_nms_outputs():
    from abraia.inference.hailo.toolbox import (
        format_order_name,
        is_nms_format_order,
    )

    order = SimpleNamespace(name="HAILO_NMS")
    assert format_order_name(order) == "HAILO_NMS"
    assert is_nms_format_order(order)
    assert is_nms_format_order("HAILO_NMS_BY_CLASS")
    assert is_nms_format_order("FormatOrder.HAILO_NMS_BY_SCORE")
    assert not is_nms_format_order("FormatOrder.HAILO_FORMAT_ORDER")


def test_pose_models_default_to_yolov8_postprocessing():
    assert resolve_model_type(None, "yolov8m_pose", "pose") == "v8"


def test_segmentation_nms_discards_non_finite_boxes():
    prediction = np.zeros((1, 1, 38), dtype=np.float32)
    prediction[0, 0, :5] = [np.nan, 320, 100, 100, 1]
    prediction[0, 0, 5] = 1

    result = postprocess.segment_non_max_suppression(
        prediction, conf_thres=0.25, nm=32, multi_label=False
    )

    assert result[0]["detection_boxes"].shape == (0, 4)


def test_pose_decoding_offsets_keypoint_coordinates_and_preserves_invalid_boxes():
    raw_boxes = [
        np.zeros((1, height, height, 64), dtype=np.float32)
        for height in (20, 40, 80)
    ]
    raw_boxes[0][0, 0, 0, 0] = np.nan
    raw_keypoints = [
        np.zeros((1, height * height, 17, 3), dtype=np.float32)
        for height in (20, 40, 80)
    ]

    boxes, keypoints = postprocess.decode_pose_results(
        raw_boxes, raw_keypoints, [32, 16, 8], (640, 640), 15
    )

    assert boxes.shape == (1, 8400, 4)
    assert keypoints.shape == (1, 8400, 17, 3)
    assert np.isnan(boxes[0, 0]).all()
    assert np.isfinite(keypoints).all()


def test_segmentation_postprocess_uses_runtime_score_threshold():
    endnodes = []
    for height in (20, 40, 80):
        endnodes.extend([
            np.zeros((1, height, height, 64), dtype=np.float32),
            np.full((1, height, height, 80), -10, dtype=np.float32),
            np.zeros((1, height, height, 32), dtype=np.float32),
        ])
    endnodes[7][0, 40, 40, 0] = 0
    endnodes.append(np.zeros((1, 160, 160, 32), dtype=np.float32))
    config = {**postprocess.SEGMENT_CONFIG["v8"], "score_threshold": 0.25}

    result = postprocess.segment_yolov8_postprocess(endnodes, **config)[0]

    assert result["detection_boxes"].shape == (1, 4)


def test_normalized_box_uses_requested_padded_size():
    input_box, padded_box = postprocess.convert_box_from_normalized(
        [0, 0, 1, 1], 16, 0, 16, 16
    )

    assert input_box == [0, 0, 16, 16]
    assert padded_box == [0, 0, 16, 16]


def test_mapped_box_preserves_exclusive_image_bounds():
    assert postprocess.map_box_to_orig(
        [0, 0, 640, 640], (100, 100), (640, 640)
    ) == [0, 0, 100, 100]


def test_letterbox_mapping_uses_original_dimensions():
    transform = postprocess.LetterboxTransform.from_dimensions((720, 1280), (640, 640))

    assert transform.pad_y == 140
    assert postprocess.map_box_to_orig(
        [0, 140, 640, 500], (720, 1280), (640, 640)
    ) == [0, 0, 1280, 720]


def test_nms_mask_uses_model_roi_dimensions_for_letterboxed_video():
    original_box, mask_box = postprocess.convert_nms_box_from_normalized(
        [0, 140 / 640, 1, 500 / 640],
        (640, 640),
        (720, 1280),
    )

    assert original_box == [0, 0, 1280, 720]
    assert mask_box == [0, 0, 640, 360]
    mask = postprocess.resize_mask_to_unpadded_box(
        np.zeros((360, 640), dtype=np.uint8), original_box, mask_box
    )
    assert mask.shape == (720, 1280)


def test_nms_mask_rejects_wrong_model_roi_shape():
    assert postprocess.resize_mask_to_unpadded_box(
        np.zeros(640 * 640, dtype=np.uint8),
        [0, 0, 1280, 720],
        [0, 0, 640, 360],
    ) is None


def test_output_layer_selection_rejects_duplicate_shapes():
    outputs = {
        "first": np.zeros((1, 20, 20, 64)),
        "second": np.zeros((1, 20, 20, 64)),
    }

    try:
        postprocess.select_output_layers(outputs, [(1, 20, 20, 64)])
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("duplicate output shapes should be rejected")


def test_empty_segmentation_result_preserves_mask_dimensions():
    prediction = np.zeros((1, 1, 5 + 2 + 4), dtype=np.float32)

    result = postprocess.segment_non_max_suppression(
        prediction, conf_thres=0.25, nm=4, multi_label=False
    )[0]

    assert result["mask"].shape == (0, 4)
    assert result["detection_classes"].shape == (0,)
    assert result["detection_scores"].shape == (0,)


def test_hailo_pipeline_adapter_reorders_async_results(tmp_path):
    from abraia.inference.hailo import pipeline as hailo_pipeline
    from abraia.runtime import Pipeline

    hef_path = tmp_path / "fake.hef"
    (tmp_path / "fake.json").write_text(
        '{"task": "detection", "classes": ["person"]}'
    )

    class FakeInference:
        def __init__(self, *args, **kwargs):
            assert kwargs["task"] == "detection"
            self.closed = False

        def get_input_shape(self):
            return 2, 2, 3

        def infer_batch(self, input_batch, emit, stop_event, image_batch=None):
            for index in reversed(range(len(input_batch))):
                emit(index, [{"value": int(input_batch[index][0, 0, 0])}])

        def close(self):
            self.closed = True

    frames = [
        np.full((4, 4, 3), value, dtype=np.uint8)
        for value in (1, 2, 3)
    ]

    with patch.object(hailo_pipeline, "HAILO_AVAILABLE", True), \
         patch.object(hailo_pipeline, "ModelInference", FakeInference):
        model = hailo_pipeline.HailoPipelineModel(
            str(hef_path), task="detection", batch_size=1
        )
        last = Pipeline(source=frames, model=model).run()

    assert last.frame_index == 2
    assert last.results == [{"value": 3}]
    assert model.inference.closed


def test_hailo_pipeline_adapter_uses_model_json_labels(tmp_path):
    from abraia.inference.hailo import pipeline as hailo_pipeline

    (tmp_path / "coco.json").write_text(
        '{"task": "detection", "classes": ["person", "car"]}'
    )
    (tmp_path / "custom.json").write_text(
        '{"task": "detection", "classes": ["tomato"]}'
    )

    class FakeInference:
        instances = []

        def __init__(self, *args, **kwargs):
            self.labels = kwargs["labels"]
            self.closed = False
            self.__class__.instances.append(self)

        def get_input_shape(self):
            return 2, 2, 3

        def close(self):
            self.closed = True

    with patch.object(hailo_pipeline, "HAILO_AVAILABLE", True), \
         patch.object(hailo_pipeline, "ModelInference", FakeInference):
        default_model = hailo_pipeline.HailoPipelineModel(str(tmp_path / "coco.hef"))
        custom_model = hailo_pipeline.HailoPipelineModel(str(tmp_path / "custom.hef"))

    assert default_model.labels == ["person", "car"]
    assert default_model.inference.labels == ["person", "car"]
    assert custom_model.labels == ["tomato"]
    assert custom_model.inference.labels == ["tomato"]
    default_model.close()
    custom_model.close()


def test_hailo_pipeline_adapter_reads_native_bundle_metadata(tmp_path):
    from abraia.inference.hailo import pipeline as hailo_pipeline

    bundle = tmp_path / "native_hailo_model"
    bundle.mkdir()
    (bundle / "native.hef").write_bytes(b"hef")
    (bundle / "metadata.yaml").write_text(
        "task: segment\nnames:\n  0: tomato\n"
    )
    (bundle / "abraia.json").write_text(
        '{"task": "segmentation", "classes": ["tomato"]}'
    )

    class FakeInference:
        def __init__(self, hef_path, **kwargs):
            self.hef_path = hef_path
            self.kwargs = kwargs

        def get_input_shape(self):
            return 2, 2, 3

        def close(self):
            pass

    with patch.object(hailo_pipeline, "HAILO_AVAILABLE", True), \
         patch.object(hailo_pipeline, "ModelInference", FakeInference):
        model = hailo_pipeline.HailoPipelineModel(str(bundle))

    assert model.task == "segmentation"
    assert model.labels == ["tomato"]
    assert model.inference.kwargs["model_type"] == "v8"
    model.close()


def test_resolve_hailo_requires_an_explicit_hef_uri():
    from abraia.inference.hailo import models

    assert models.resolve_hef_path("project/model_hailo8l", "detect") is None


def test_resolve_hailo_rejects_a_catalog_name():
    from abraia.inference.hailo import models

    assert models.resolve_hef_path("yolov8n", "detect") is None


def test_resolve_hailo_unknown_name_does_not_download_external_models():
    from abraia.inference.hailo import models

    assert models.resolve_hef_path(
        "not-a-bundled-model", "detect", arch="hailo8"
    ) is None
