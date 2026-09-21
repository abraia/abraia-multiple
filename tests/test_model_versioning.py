import pytest
from pathlib import Path

from abraia.training.core import (
    next_model_version,
    save_hailo_bundle,
    save_versioned_model,
)


class FakeClient:
    def __init__(self, names):
        self.names = names

    def list_files(self, path):
        assert path == "project/"
        return ([{"name": name} for name in self.names], [])


def test_next_model_version_uses_numeric_flat_filenames():
    client = FakeClient([
        "yolov8n_v1.onnx",
        "yolov8n_v3.onnx",
        "yolov8n_v10.onnx",
        "yolov8n-seg_v20.onnx",
        "yolov8n_v10.json",
    ])

    assert next_model_version(client, "project", "yolov8n") == 11


def test_next_model_version_starts_at_one():
    assert next_model_version(
        FakeClient(["yolov8n.onnx"]), "project", "yolov8n"
    ) == 1


def test_save_versioned_model_skips_a_remote_collision():
    class SavingClient:
        def __init__(self):
            self.uploads = []
            self.metadata = []

        def list_files(self, path):
            assert path == "project/"
            return ([], [])

        def check_file(self, path):
            return path == "project/yolov8n_v1.onnx"

        def upload_file(self, source, path):
            self.uploads.append((source, path))

        def save_json(self, path, value):
            self.metadata.append((path, value))

    client = SavingClient()
    result = save_versioned_model(
        client,
        "project",
        "yolov8n",
        "exported.onnx",
        {"metrics": {"mAP": 0.8}},
    )

    assert result["name"] == "yolov8n_v2.onnx"
    assert client.uploads == [("exported.onnx", "project/yolov8n_v2.onnx")]
    assert client.metadata == [(
        "project/yolov8n_v2.json",
        {"metrics": {"mAP": 0.8}},
    )]


def test_save_versioned_model_removes_onnx_when_metadata_save_fails():
    class FailingMetadataClient:
        def list_files(self, path):
            return ([], [])

        def check_file(self, path):
            return False

        def upload_file(self, source, path):
            self.uploaded = (source, path)

        def save_json(self, path, value):
            raise RuntimeError("metadata unavailable")

        def remove_file(self, path):
            self.removed = path

    client = FailingMetadataClient()

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        save_versioned_model(
            client, "project", "yolov8n", "exported.onnx", {"metrics": {}}
        )

    assert client.removed == "project/yolov8n_v1.onnx"


def test_save_hailo_bundle_uploads_sidecars_and_flat_hef(tmp_path):
    bundle = tmp_path / "yolov8n_hailo_model"
    bundle.mkdir()
    (bundle / "yolov8n.hef").write_bytes(b"hef")
    (bundle / "metadata.yaml").write_text("task: detect\n")
    (bundle / "nms_config.json").write_text("{}")

    class HailoClient:
        def __init__(self):
            self.uploads = []
            self.manifests = []

        def upload_file(self, source, path):
            self.uploads.append((Path(source).name, path))

        def save_json(self, path, value):
            self.manifests.append((path, value))

    client = HailoClient()
    result = save_hailo_bundle(
        client,
        "project",
        "yolov8n",
        bundle,
        "hailo8l",
        version=3,
        metadata={"format": "hailo"},
    )

    assert result["hef"] == "project/yolov8n_v3_hailo8l.hef"
    assert result["bundle"] == "project/yolov8n_v3_hailo8l"
    assert ("yolov8n.hef", "project/yolov8n_v3_hailo8l/yolov8n.hef") in client.uploads
    assert ("yolov8n.hef", "project/yolov8n_v3_hailo8l.hef") in client.uploads
    assert client.manifests == [
        (
            "project/yolov8n_v3_hailo8l/abraia.json",
            {"format": "hailo"},
        )
    ]
