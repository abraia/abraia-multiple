from abraia import config
from abraia.sources import normalize_source_type
from abraia.runtime.config import (
    format_points,
    load_pipeline_document,
    parse_points,
    save_pipeline_document,
)


def test_load_uses_user_id_encoded_in_key(monkeypatch):
    encoded_key = config.base64encode("user-123:secret")
    monkeypatch.delenv("ABRAIA_ID", raising=False)
    monkeypatch.setenv("ABRAIA_KEY", encoded_key)

    assert config.load() == ("user-123", encoded_key)


def test_load_prefers_explicit_user_id(monkeypatch):
    encoded_key = config.base64encode("encoded-user:secret")
    monkeypatch.setenv("ABRAIA_ID", "configured-user")
    monkeypatch.setenv("ABRAIA_KEY", encoded_key)

    assert config.load() == ("configured-user", encoded_key)


def test_load_resolves_key_only_config_file(monkeypatch, tmp_path):
    encoded_key = config.base64encode("file-user:secret")
    config_path = tmp_path / "abraia"
    config_path.write_text(f"abraia_key: {encoded_key}\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", str(config_path))
    monkeypatch.delenv("ABRAIA_ID", raising=False)
    monkeypatch.delenv("ABRAIA_KEY", raising=False)

    assert config.load() == ("file-user", encoded_key)


def test_pipeline_point_helpers_round_trip_editor_text():
    points = parse_points("0, 1; 2.5, -3")

    assert points == [[0.0, 1.0], [2.5, -3.0]]
    assert format_points(points) == "0, 1; 2.5, -3"


def test_pipeline_document_helpers_round_trip_json(tmp_path):
    filename = tmp_path / "pipeline.json"
    configuration = {"version": 1, "stages": []}

    save_pipeline_document(filename, configuration)

    assert load_pipeline_document(filename) == configuration


def test_source_type_aliases_have_one_canonical_name():
    assert normalize_source_type("images") == "image"
    assert normalize_source_type(" usb_camera ") == "camera"
    assert normalize_source_type("stream") == "stream"
