from abraia import config
from abraia.sources import normalize_source_type
from abraia.runtime.config import (
    PipelineDraft,
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


def test_save_writes_only_the_api_key(monkeypatch, tmp_path):
    config_path = tmp_path / "abraia"
    monkeypatch.setattr(config, "CONFIG_FILE", str(config_path))
    encoded_key = config.base64encode("saved-user:secret")

    config.save(encoded_key)

    assert config_path.read_text(encoding="utf-8") == (
        f"abraia_key: {encoded_key}\n"
    )


def test_pipeline_point_helpers_round_trip_editor_text():
    points = parse_points("0, 1; 2.5, -3")

    assert points == [[0.0, 1.0], [2.5, -3.0]]
    assert format_points(points) == "0, 1; 2.5, -3"


def test_pipeline_document_helpers_round_trip_json(tmp_path):
    filename = tmp_path / "pipeline.json"
    configuration = {"version": 1, "stages": []}

    save_pipeline_document(filename, configuration)

    assert load_pipeline_document(filename) == configuration


def test_composed_steps_include_tracker_and_region_processors():
    configuration = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "detect", "type": "model", "input": "frame", "model": {
                "kind": "yolov8", "task": "detection", "uri": "model.onnx"
            }},
            {"id": "track", "type": "tracker"},
            {"id": "count", "type": "line_counter", "line": [[0, 0], [10, 10]]},
            {"id": "filter", "type": "region_filter", "polygon": [[0, 0], [10, 0], [10, 10]]},
            {"id": "timer", "type": "region_timer", "polygon": [[0, 0], [10, 0], [10, 10]]},
        ],
    }

    draft = PipelineDraft.from_dict(configuration)

    assert draft.validation_errors() == []
    serialized = draft.to_dict()
    assert serialized["version"] == 2
    assert [step["type"] for step in serialized["steps"]] == [
        "model", "tracker", "line_counter", "region_filter", "region_timer"
    ]


def test_version_one_tracker_stages_remain_version_one():
    draft = PipelineDraft.from_dict({
        "version": 1,
        "source": {"type": "image", "src": "frame.jpg"},
        "model": {"kind": "yolov8", "task": "detection", "uri": "model.onnx"},
        "stages": [{"type": "tracker"}],
    })

    assert draft.to_dict()["version"] == 1


def test_pipeline_draft_steps_are_canonical_with_stages_compatibility():
    steps = [{"type": "tracker"}]
    draft = PipelineDraft(steps=steps)

    assert draft.steps == steps
    assert draft.stages is draft.steps

    draft.stages = [{"type": "line_counter"}]
    assert draft.steps == [{"type": "line_counter"}]

    legacy_draft = PipelineDraft(stages=steps)
    assert legacy_draft.steps == steps


def test_source_type_aliases_have_one_canonical_name():
    assert normalize_source_type("images") == "image"
    assert normalize_source_type(" usb_camera ") == "camera"
    assert normalize_source_type("stream") == "stream"
