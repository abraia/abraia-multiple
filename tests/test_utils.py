from pathlib import Path
from unittest.mock import patch

import numpy as np

from abraia import APIError
from abraia import Abraia
from abraia.utils import ArtifactResolver, load_json, resolve_model_file, save_json
from abraia.utils.remote import is_managed_model_path


def test_save_json_uses_destination_first_and_encodes_numpy_values(tmp_path):
    destination = tmp_path / 'nested' / 'values.json'
    payload = {'value': np.int64(3), 'mask': np.array([True, False])}

    assert save_json(destination, payload) == destination
    assert load_json(destination) == {'value': 3, 'mask': [True, False]}


def test_documented_rendering_helpers_are_available_from_utils():
    from abraia.utils import render_counter, render_region

    assert callable(render_counter)
    assert callable(render_region)


def test_managed_model_uri_prefers_remote_asset_over_local_copy(tmp_path):
    local = tmp_path / "multiple" / "models" / "model.onnx"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    cached = tmp_path / "cached-model.onnx"
    cached.write_bytes(b"remote")

    with patch(
        "abraia.utils.remote.download_file",
        return_value=str(cached),
    ) as download:
        assert resolve_model_file(str(local)) == str(local)
        assert resolve_model_file("multiple/models/model.onnx") == str(cached)

    download.assert_called_once_with("multiple/models/model.onnx")


def test_api_error_preserves_text_messages():
    error = APIError("server failed", 500)

    assert error.message == "server failed"
    assert str(error) == "server failed"
    assert error.code == 500


def test_managed_model_namespace_does_not_accept_legacy_models_path():
    assert is_managed_model_path("multiple/models/model.onnx")
    assert not is_managed_model_path("models/model.onnx")


def test_artifact_resolver_prefers_managed_namespace_over_local_copy(tmp_path):
    local = tmp_path / "multiple" / "models" / "model.onnx"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")

    reference = ArtifactResolver().reference("multiple/models/model.onnx")

    assert reference.local_path is None
    assert reference.remote_path == "multiple/models/model.onnx"
    assert reference.source == "managed"


def test_artifact_resolver_accepts_remote_head_without_content_length():
    class Response:
        ok = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    resolver = ArtifactResolver()
    with patch("abraia.utils.remote._get_url_session") as get_session:
        get_session.return_value.head.return_value = Response()
        assert resolver.remote_available("multiple/models/model.hef")


def test_artifact_resolver_falls_back_to_a_range_get_when_head_fails():
    class Response:
        def __init__(self, ok):
            self.ok = ok

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    resolver = ArtifactResolver()
    with patch("abraia.utils.remote._get_url_session") as get_session:
        get_session.return_value.head.return_value = Response(False)
        get_session.return_value.get.return_value = Response(True)
        assert resolver.remote_available("multiple/models/model.hef")
        get_session.return_value.get.assert_called_once()


def test_transform_image_does_not_leak_default_parameters_between_calls():
    client = object.__new__(Abraia)
    client._client = None
    client.auth = None
    client.userid = "user"

    response = type("Response", (), {"status_code": 200, "content": b"image"})()
    with patch.object(client, "_request", return_value=response) as request, \
         patch("abraia.client.save_data"):
        client.transform_image("source.jpg", "first.jpg")
        client.transform_image("source.jpg", "second.png")

    assert request.call_args_list[0].kwargs["params"]["format"] == "jpg"
    assert request.call_args_list[1].kwargs["params"]["format"] == "png"
