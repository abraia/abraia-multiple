import json
import math
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from abraia.cli import _complete_demo_model, _complete_run_demo, cli
from abraia.inference.models.detection import prepare_input, process_obb_output
from abraia.inference.registry import create_model
from abraia.runtime import Pipeline


_INPUT_SHAPE = [1, 3, 100, 100]
_CLASSES = ["ship", "harbor"]
_ROW = [50, 50, 20, 10, .9, .1, math.pi / 6]


def _head(rows):
    return np.asarray(rows, dtype=np.float32).reshape(-1, 7).T[None]


def _decode(rows, **options):
    return process_obb_output([_head(rows)], (200, 100), _INPUT_SHAPE, _CLASSES, **options)


def test_obb_reverses_letterbox_and_preserves_rectangle():
    result, = _decode([_ROW])
    assert result["class_id"] == 0 and result["label"] == "ship"
    assert result["score"] == pytest.approx(.9)
    assert result["box"] == [77, 31, 46, 38]
    rectangle = result["rotated_rectangle_px"]
    assert rectangle["center"] == [100, 50]
    assert rectangle["size"] == [40, 20]
    assert rectangle["angle_deg"] == pytest.approx(30)
    v1 = np.array([math.sqrt(3) * 10, 10])
    v2 = np.array([-5, math.sqrt(3) * 5])
    expected = sorted((np.array([100, 50]) + x * v1 + y * v2).tolist()
                      for x in (-1, 1) for y in (-1, 1))
    np.testing.assert_allclose(sorted(result["polygon"]), expected, atol=1e-5)


def test_obb_inferred_transform_matches_preprocessing_with_odd_padding():
    image = np.zeros((99, 200, 3), dtype=np.uint8)
    _, scale, padding = prepare_input(image, _INPUT_SHAPE, return_transform=True)
    args = ([_head([_ROW])], (200, 99), _INPUT_SHAPE, _CLASSES)
    assert process_obb_output(*args) == process_obb_output(*args, transform=(scale, padding))


def test_obb_nms_respects_rotation_classes_and_filters():
    a = math.pi / 4
    rows = [
        [50, 50, 70, 8, .95, 0, a],
        [50, 50, 70, 8, .90, 0, a],    # Same rectangle and class.
        [50, 50, 70, 8, .88, 0, -a],   # Crossing box with the same envelope.
        [50, 50, 70, 8, 0, .82, a],    # Same rectangle, different class.
        [5, 50, 5, 4, .25, 0, 0],      # Below the confidence threshold.
        [5, 50, 0, 4, .99, 0, np.nan], # Invalid geometry.
    ]
    results = _decode(rows)
    assert [result["label"] for result in results] == ["ship", "ship", "harbor"]
    assert [result["score"] for result in results] == pytest.approx([.95, .88, .82])
    assert results[0]["box"] == results[1]["box"]
    assert results[0]["rotated_rectangle_px"]["angle_deg"] > 0
    assert results[1]["rotated_rectangle_px"]["angle_deg"] < 0
    assert len(_decode(rows, labels=["harbor"])) == 1
    assert _decode(rows, conf_threshold=1) == []
    assert _decode([]) == []


def test_obb_clips_only_axis_aligned_envelope():
    result, = _decode([[2, 50, 10, 8, .9, 0, 0], [-50, 50, 10, 8, .8, 0, 0]])
    assert result["box"] == [0, 42, 14, 16]
    assert min(x for x, _ in result["polygon"]) == -6


@pytest.mark.parametrize("outputs", [[], [np.zeros((1, 6, 2))], [np.zeros((2, 7, 1))]])
def test_obb_rejects_malformed_head(outputs):
    with pytest.raises(ValueError, match="one raw output"):
        process_obb_output(outputs, (200, 100), _INPUT_SHAPE, _CLASSES)


@pytest.fixture
def obb_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper = onnx.helper
    overlapping = [
        50 + 5 * math.cos(math.pi / 6), 50 + 5 * math.sin(math.pi / 6),
        20, 10, .8, .1, math.pi / 6,
    ]
    predictions = _head([_ROW, overlapping])
    graph = helper.make_graph([
        helper.make_node("Constant", [], ["head"], value=onnx.numpy_helper.from_array(predictions)),
    ], "obb", [
        helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, _INPUT_SHAPE),
    ], [
        helper.make_tensor_value_info("head", onnx.TensorProto.FLOAT, predictions.shape),
    ])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    path = tmp_path / "yolov8n-obb.onnx"
    onnx.save_model(model, path)
    path.with_suffix(".json").write_text(json.dumps({
        "kind": "yolov8", "task": "obb", "inputShape": _INPUT_SHAPE, "classes": _CLASSES,
    }))
    return path


def test_obb_uses_the_default_managed_graph_and_sidecar(tmp_path, obb_model, monkeypatch):
    from abraia.utils import remote
    from abraia.utils.cache import RemoteFileCache

    monkeypatch.setattr(remote, "RemoteFileCache", lambda userid: RemoteFileCache(
        userid, root=tmp_path / "cache",
    ))
    downloaded = []

    def download(url, dest):
        downloaded.append(url)
        shutil.copyfile(obb_model.with_name(url.rsplit("/", 1)[-1]), dest)

    monkeypatch.setattr(remote, "download_url", download)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    with create_model({"kind": "yolov8", "task": "obb"}, accelerator="cpu") as model:
        result, = model.run(image)
        assert len(model.run(image, iou_threshold=.7)) == 2
    assert result["rotated_rectangle_px"]["angle_deg"] == pytest.approx(30)
    assert downloaded == [
        remote.url_path("multiple/models/yolov8n-obb.onnx"),
        remote.url_path("multiple/models/yolov8n-obb.json"),
    ]


def test_obb_saved_pipeline_preserves_rotation_and_counts_objects(tmp_path, obb_model):
    Image.new("RGB", (200, 100)).save(tmp_path / "frame.png")
    config = {
        "version": 2, "source": {"type": "image", "src": "frame.png"},
        "steps": [
            {"type": "model", "model": {"kind": "yolov8", "task": "obb", "uri": obb_model.name}},
            {"type": "tracker"},
            {"type": "measure", "features": ["area_px", "rotated_rectangle_px"]},
            {"type": "count", "mode": "region", "polygon": [[0, 0], [190, 0], [190, 90], [0, 90]]},
        ],
        "display": {"show": False, "render_metrics": False},
    }
    with Pipeline.from_dict(config, base_dir=tmp_path, accelerator="cpu") as pipeline:
        context = pipeline.run()
        rendered = pipeline.render(context)
    record, = context.results
    assert context.metrics["object_count"] == 1
    assert record["track_id"] is not None
    assert record["area_px"] == pytest.approx(800, abs=.001)
    assert record["rotated_rectangle_px"]["angle_deg"] == pytest.approx(30)
    for x, y in np.rint(record["polygon"]).astype(int):
        assert rendered[y, x].any()
    assert not rendered[68, 122].any()


def test_obb_demo_completion_source_override_and_output(monkeypatch):
    configuration = {
        "version": 2, "source": {"type": "image", "src": "multiple/images/obb-marina.jpg"},
        "steps": [{"type": "model", "model": {"kind": "yolov8", "task": "obb"}}],
    }
    monkeypatch.setattr("abraia.demo.load_public_pipeline", lambda name: configuration)
    build = MagicMock()
    monkeypatch.setattr(Pipeline, "from_dict", build)
    ctx = SimpleNamespace(params={"project": "demo", "demo_or_source": "obb"})
    assert [item.value for item in _complete_run_demo(ctx, None, "ob")] == ["obb"]
    assert [item.value for item in _complete_demo_model(ctx, None, "")] == ["yolov8n-obb"]
    assert "obb" in CliRunner().invoke(cli, ["run", "demo", "list"]).output
    result = CliRunner().invoke(cli, [
        "run", "demo", "obb", "aerial.mp4", "--model", "yolov8n-obb", "--output", "obb.avi",
    ])
    assert result.exit_code == 0, result.output
    config = build.call_args.args[0]
    assert config["source"] == {"type": "video", "src": "aerial.mp4"}
    assert config["steps"][0]["model"]["uri"] == "multiple/models/yolov8n-obb.onnx"
    assert config["display"]["dest"] == "obb.avi"


def test_obb_explicit_hailo_reports_unsupported_task():
    with pytest.raises(RuntimeError, match="Hailo required for model 'yolov8' \\(obb\\)"):
        create_model({"kind": "yolov8", "task": "obb"}, accelerator="hailo")
