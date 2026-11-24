import json
import shutil

import numpy as np
import pytest
from PIL import Image

from abraia.inference import ModelSpec, load_model
from abraia.runtime import Pipeline
from abraia.runtime.results import table_result_records


def make_depth_onnx(path, kind):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper = onnx.helper
    da3 = kind == "depth_anything"
    image_name = "pixel_values" if da3 else "images"
    depth_name = "predicted_depth" if da3 else "depth"
    image_shape = [1, 1, 3, "h", "w"] if da3 else [1, 3, 3, 4]
    depth_shape = [1, 1, "h", "w"] if da3 else [1, 1, 3, 4]
    bias = onnx.numpy_helper.from_array(np.array([2], np.float32), name="bias")
    graph = helper.make_graph([
        helper.make_node(
            "ReduceMean", [image_name], ["mean"], axes=[2 if da3 else 1],
            keepdims=0 if da3 else 1,
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


@pytest.mark.parametrize("kind,units", [
    ("depth_anything", "relative"),
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
        np.testing.assert_allclose(depth[:, 0], 2 + np.mean(-mean / std), atol=1e-6)
        np.testing.assert_allclose(
            depth[:, -1], 2 + np.mean(([1, 0, 0] - mean) / std), atol=1e-6,
        )

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
