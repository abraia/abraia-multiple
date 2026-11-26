import json
from importlib.util import module_from_spec, spec_from_file_location
import shutil
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from abraia.inference import ModelSpec, load_model
from abraia.runtime import Pipeline
from abraia.runtime.results import table_result_records
from abraia.inference.models.depth import _resize_dav2


@pytest.fixture
def hailo_depth_backend(monkeypatch, request):
    """Exercise model routing and decoding with only the HailoRT boundary mocked."""
    from abraia.inference.hailo import pipeline, toolbox

    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(
        HEF=object, VDevice=object, FormatType=object, HailoSchedulingAlgorithm=object,
    ))
    spec = spec_from_file_location("abraia.inference.hailo._test_toolbox", toolbox.__file__)
    backend = module_from_spec(spec)
    spec.loader.exec_module(backend)
    opened, closed, submitted = [], [], []
    has_nms = getattr(request, "param", False)

    def open_model(self, path, batch_size, *, input_type, output_type):
        opened.append((path, batch_size, input_type, output_type))
        self.nms_postprocess_enabled = has_nms

    def submit(self, inputs, callback):
        submitted.extend(inputs)
        bindings = [
            SimpleNamespace(
                _output_names=["depth"],
                output=lambda: SimpleNamespace(
                    get_buffer=lambda: np.array([[[-8.0], [7.0]]], np.float32),
                ),
            )
            for _ in inputs
        ]
        callback(SimpleNamespace(exception=None), bindings_list=bindings)

    monkeypatch.setattr(backend.HailoInfer, "__init__", open_model)
    monkeypatch.setattr(backend.HailoInfer, "get_input_shape", lambda self: (4, 6, 3))
    monkeypatch.setattr(backend.HailoInfer, "run", submit)
    monkeypatch.setattr(backend.HailoInfer, "wait", lambda self: None)
    monkeypatch.setattr(backend.HailoInfer, "close", lambda self: closed.append(self))
    monkeypatch.setattr(pipeline, "HAILO_AVAILABLE", True)
    monkeypatch.setattr(pipeline, "ModelInference", backend.ModelInference)
    return opened, closed, submitted


@pytest.mark.parametrize("live", [False, True])
def test_yolo26_hailo_depth_saved_pipeline(tmp_path, hailo_depth_backend, live):
    opened, closed, submitted = hailo_depth_backend
    hef = tmp_path / "yolo26n-depth_hailo8.hef"
    hef.touch()
    hef.with_suffix(".json").write_text(json.dumps({
        "format": "hailo", "kind": "yolo26_depth", "task": "depth",
        "cal_a": 0.5, "cal_b": 0.25,
    }))
    image = np.full((3, 10, 3), [20, 80, 200], dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "frame.png")

    class LiveSource:
        is_live = True

        def __iter__(self):
            yield image

    pipeline = Pipeline.from_dict(
        {
            "version": 2,
            "source": {"type": "image", "src": "frame.png"},
            "steps": [{"type": "model", "model": {
                "kind": "yolo26_depth", "task": "depth", "uri": hef.name,
            }}],
            "display": {"show": False},
        },
        base_dir=tmp_path,
        accelerator="hailo",
        input_source=LiveSource() if live else None,
    )
    context = pipeline.run()
    assert opened == [(hef, 1, "UINT8", "FLOAT32")]
    assert len(closed) == 1
    assert pipeline.model.accelerator == "HAILO"
    np.testing.assert_array_equal(submitted[0], cv2.resize(image, (6, 4)))
    result, = context.results
    assert result["depth_units"] == "m"
    assert result["depth_map"].dtype == np.float32
    expected = np.linspace(np.exp(-1.75), np.exp(2.75), image.shape[1])
    np.testing.assert_allclose(
        result["depth_map"], np.broadcast_to(expected, image.shape[:2]), rtol=1e-6,
    )
    assert not np.array_equal(pipeline.render_frame(context), image)


@pytest.mark.parametrize("metadata,key", [
    ({"cal_b": 0}, "cal_a"),
    ({"cal_a": 1}, "cal_b"),
    ({"cal_a": True, "cal_b": 0}, "cal_a"),
    ({"cal_a": 1, "cal_b": float("inf")}, "cal_b"),
])
def test_yolo26_hailo_depth_rejects_invalid_json_before_opening_device(
    tmp_path, hailo_depth_backend, metadata, key,
):
    hef = tmp_path / "yolo26n-depth_hailo8.hef"
    hef.touch()
    hef.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"finite numeric {key}"):
        load_model({"kind": "yolo26_depth", "uri": str(hef)})
    assert hailo_depth_backend[0] == []


@pytest.mark.parametrize("hailo_depth_backend", [True], indirect=True)
def test_yolo26_hailo_depth_rejects_nms_hef_and_closes_device(tmp_path, hailo_depth_backend):
    hef = tmp_path / "detector_hailo8.hef"
    hef.touch()
    hef.with_suffix(".json").write_text('{"cal_a": 1, "cal_b": 0}')

    with pytest.raises(ValueError, match="depth logits instead of NMS outputs"):
        load_model({"kind": "yolo26_depth", "uri": str(hef)})

    opened, closed, _ = hailo_depth_backend
    assert len(opened) == len(closed) == 1


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 3), (0, 2, 1)])
def test_yolo26_hailo_depth_rejects_invalid_output_shape(shape):
    from abraia.inference.hailo.postprocess import yolo26_depth_postprocess

    with pytest.raises(ValueError, match="one HWC depth logit map"):
        yolo26_depth_postprocess(
            np.zeros(shape), (4, 4), (2, 2), {"cal_a": 1, "cal_b": 0},
        )


def make_depth_onnx(path, kind):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper = onnx.helper
    da3 = kind == "depth_anything"
    image_name = "pixel_values" if da3 else "images"
    depth_name = (
        "predicted_depth"
        if kind in ("depth_anything", "depth_anything_v2") else "depth"
    )
    image_shape = [1, 1, 3, "h", "w"] if da3 else [1, 3, 3, 4]
    depth_shape = (
        [1, 1, "h", "w"] if da3
        else [1, "h", "w"] if kind == "depth_anything_v2"
        else [1, 1, 3, 4]
    )
    bias = onnx.numpy_helper.from_array(np.array([2], np.float32), name="bias")
    graph = helper.make_graph([
        helper.make_node(
            "ReduceMean", [image_name], ["mean"], axes=[2 if da3 else 1],
            keepdims=0 if da3 or kind == "depth_anything_v2" else 1,
        ),
        helper.make_node("Add", ["mean", "bias"], [depth_name]),
    ], "depth", [
        helper.make_tensor_value_info(image_name, onnx.TensorProto.FLOAT, image_shape),
    ], [
        helper.make_tensor_value_info(depth_name, onnx.TensorProto.FLOAT, depth_shape),
    ], [bias])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
    )
    helper.set_model_props(model, {"task": "depth"})
    onnx.save_model(model, path)


def test_depth_anything_v2_resize_uses_518_long_edge():
    resized = _resize_dav2(np.zeros((720, 1280, 3), np.uint8), ("h", "w"))
    assert max(resized.shape[:2]) == 518
    assert min(resized.shape[:2]) < 518


@pytest.mark.parametrize("kind,units", [
    ("depth_anything", "relative"),
    ("depth_anything_v2", "relative"),
    ("yolo26_depth", "m"),
])
def test_depth_prediction_and_saved_pipeline(tmp_path, kind, units):
    path = tmp_path / "depth.onnx"
    make_depth_onnx(path, kind)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[:, 4:, 0] = 255
    Image.fromarray(image).save(tmp_path / "frame.png")

    config = {"kind": kind, "uri": str(path)}
    assert ModelSpec.from_config(config).task == "depth"
    with load_model(config) as model:
        prediction = model.predict(image)
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(image)
    assert prediction["depth_units"] == units
    depth = prediction["depth_map"]
    assert depth.shape == (6, 8)
    assert depth.dtype == np.float32
    if units == "m":
        expected = 2 + np.interp(np.linspace(0, 3, 8), range(4), [0, 0, 1, 1]) / 3
        np.testing.assert_allclose(depth, np.broadcast_to(expected, depth.shape))
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        left = 2 + np.mean(-mean / std)
        right = 2 + np.mean(([1, 0, 0] - mean) / std)
        if kind == "depth_anything_v2":
            np.testing.assert_allclose(depth[:, 0], left + right - left, atol=1e-6)
            np.testing.assert_allclose(depth[:, -1], left + right - right, atol=1e-6)
            assert depth[:, -1].mean() < depth[:, 0].mean()
        else:
            np.testing.assert_allclose(depth[:, 0], left, atol=1e-6)
            np.testing.assert_allclose(depth[:, -1], right, atol=1e-6)
    if kind == "yolo26_depth":
        assert depth[:, -1].mean() > depth[:, 0].mean()

    pipeline_file = tmp_path / "pipeline.json"
    pipeline_file.write_text(json.dumps({
        "version": 2,
        "source": {"type": "image", "src": "frame.png"},
        "steps": [{"type": "model", "model": dict(config, uri=path.name)}],
        "display": {"show": False},
    }))
    frames = []
    with Pipeline.from_file(
        pipeline_file, accelerator="cpu", on_frame=lambda context, _: frames.append(context),
    ) as pipeline:
        context = pipeline.run()
    assert frames == [context]
    record, = context.results
    assert record["depth_units"] == units
    np.testing.assert_array_equal(record["depth_map"], depth)
    assert "depth_map" not in table_result_records(context.results)[0]


@pytest.mark.parametrize("kind,filename,units", [
    ("depth_anything", "depth_anything_v3_small.onnx", "relative"),
    ("depth_anything_v2", "depth_anything_v2_small.onnx", "relative"),
    ("yolo26_depth", "yolo26n-depth.onnx", "m"),
])
def test_depth_models_use_the_shared_artifact_cache(
    tmp_path, monkeypatch, kind, filename, units,
):
    from abraia.utils import remote
    from abraia.utils.cache import RemoteFileCache

    graph = tmp_path / filename
    make_depth_onnx(graph, kind)
    monkeypatch.setattr(remote, "RemoteFileCache", lambda userid: RemoteFileCache(
        userid, root=tmp_path / "cache",
    ))
    downloads = []

    def download(url, dest):
        downloads.append(url)
        shutil.copyfile(graph, dest)

    monkeypatch.setattr(remote, "download_url", download)
    spec = ModelSpec.from_config({"kind": kind})
    assert spec.default_uri == f"multiple/models/{filename}"
    for _ in range(2):
        with load_model({"kind": kind}) as model:
            assert model.predict(np.zeros((30, 60, 3), np.uint8))["depth_units"] == units
    assert downloads == [remote.url_path(spec.default_uri)]
