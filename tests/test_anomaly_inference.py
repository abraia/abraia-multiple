"""Offline checks for ONNX anomaly inference adapters."""

import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from abraia.inference import load_model
from abraia.runtime import Pipeline


@pytest.fixture
def anomaly_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    graph = onnx.helper.make_graph([
        onnx.helper.make_node(
            "ReduceMean", ["image"], ["anomaly_map"], axes=[1], keepdims=1,
        ),
        onnx.helper.make_node(
            "ReduceMax", ["anomaly_map"], ["score"], axes=[1, 2, 3], keepdims=0,
        ),
    ], "anomaly", [
        onnx.helper.make_tensor_value_info(
            "image", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
        ),
    ], [
        onnx.helper.make_tensor_value_info("score", onnx.TensorProto.FLOAT, [1]),
        onnx.helper.make_tensor_value_info(
            "anomaly_map", onnx.TensorProto.FLOAT, [1, 1, 8, 8]
        ),
    ])
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8,
    )
    onnx.helper.set_model_props(model, {
        "task": "anomaly", "algorithm": "efficientad",
        "image_threshold": "0.5", "pixel_threshold": "0.7",
    })
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def defect_image():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = 255
    return image


def test_anomaly_heatmap_only_colors_pixels_above_threshold():
    pytest.importorskip("cv2")
    from abraia.inference.models.anomaly import render_anomaly_heatmap

    source = np.full((1, 4, 3), 128, dtype=np.uint8)
    anomaly_map = np.array([[0.1, 0.49, 0.5, 0.75]], dtype=np.float32)

    rendered = render_anomaly_heatmap(source, anomaly_map)

    assert np.array_equal(rendered[0, :3], source[0, :3])
    assert not np.array_equal(rendered[0, 3], source[0, 3])


def test_abraia_loads_anomaly_labels_maps_and_regions(anomaly_onnx):
    with load_model({
        "kind": "anomaly", "task": "anomaly", "uri": str(anomaly_onnx),
    }) as model:
        prediction = model.predict(defect_image())
        assert prediction["label"] == "anomaly"
        assert prediction["score"] == 1.0
        assert prediction["anomaly_map"].shape == (8, 8)
        results = model.run(defect_image())
        assert [item["label"] for item in results] == ["anomaly", "defect"]
        assert results[1]["box"] == [2, 2, 4, 4]
        assert len(results[1]["polygon"]) == 4
        assert json.loads(json.dumps(results)) == results
        assert model.run(np.zeros((8, 8, 3), dtype=np.uint8))[0]["label"] == "normal"
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(defect_image())


@pytest.mark.parametrize("follow_up", [False, True])
def test_anomaly_runs_in_abraia_pipeline(anomaly_onnx, follow_up):
    image_path = anomaly_onnx.parent / "part.png"
    Image.fromarray(defect_image()).save(image_path)
    step = {"type": "model", "model": {
        "kind": "anomaly", "task": "anomaly", "uri": anomaly_onnx.name,
    }}
    configuration = {
        "version": 2,
        "source": {"type": "image", "src": image_path.name},
        "steps": [step] + ([step] if follow_up else []),
        "display": {"show": False},
    }

    with Pipeline.from_dict(
        configuration, base_dir=anomaly_onnx.parent, accelerator="cpu",
    ) as pipeline:
        from abraia.runtime.rendering import AnomalyPipelineRenderer

        assert isinstance(pipeline.render, AnomalyPipelineRenderer)
        context = pipeline.run()
        assert context.results[0]["label"] == "anomaly"
        assert context.results[1]["label"] == "defect"
        if follow_up:
            assert context.results[0]["result"]["items"][0]["label"] == "anomaly"
        assert json.loads(json.dumps(context.results)) == context.results
        assert pipeline.render(context).shape == (8, 8, 3)


def test_anomaly_pipeline_runs_without_pytorch(anomaly_onnx):
    image_path = anomaly_onnx.parent / "part.png"
    Image.fromarray(defect_image()).save(image_path)
    script = """
import builtins
import sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'torchvision'):
        raise AssertionError('Inference attempted a PyTorch import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from abraia.runtime import Pipeline
configuration = {
    'version': 2,
    'source': {'type': 'image', 'src': sys.argv[2]},
    'steps': [{'type': 'model', 'model': {
        'kind': 'anomaly', 'task': 'anomaly', 'uri': sys.argv[1],
    }}],
    'display': {'show': False},
}
with Pipeline.from_dict(configuration, accelerator='cpu') as pipeline:
    context = pipeline.run()
    assert context.results[0]['label'] == 'anomaly'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(anomaly_onnx), str(image_path)],
        check=True, capture_output=True, text=True, timeout=45,
    )
