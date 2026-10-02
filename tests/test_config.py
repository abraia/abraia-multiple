import json
from pathlib import Path

import pytest

from abraia import config
from abraia.sources import normalize_source_type
from abraia.runtime.config import (
    PipelineDraft,
    format_points,
    parse_points,
)
from abraia.runtime.pipeline_rules import (
    is_valid_stage_geometry,
    stages_missing_tracker,
)
from abraia.runtime.pipeline_schema import PipelineStep


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


def test_composed_steps_include_tracker_and_region_processors():
    configuration = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"id": "detect", "type": "model", "model": {
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
    assert "model" not in serialized
    assert "stages" not in serialized
    assert [step["type"] for step in serialized["steps"]] == [
        "model", "tracker", "line_counter", "region_filter", "region_timer"
    ]
    assert all("id" not in step for step in serialized["steps"])


def test_pipeline_draft_rejects_unsupported_versions():
    with pytest.raises(ValueError, match="Only version 2"):
        PipelineDraft.from_dict({"version": 3})


def test_pipeline_draft_stores_ordered_steps():
    steps = [{"type": "tracker"}]
    draft = PipelineDraft(steps=steps)

    assert draft.steps == steps


def test_pipeline_draft_does_not_repair_missing_primary_model():
    draft = PipelineDraft(steps=[{"type": "tracker"}])

    with pytest.raises(ValueError, match="must start with a model step"):
        draft.to_dict()


def test_pipeline_draft_rejects_non_object_steps():
    with pytest.raises(ValueError, match="steps must contain objects"):
        PipelineDraft.from_dict({
            "version": 2,
            "source": {"type": "image", "src": "frame.jpg"},
            "steps": [{
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "task": "detection"},
            }, "crop"],
        })


def test_pipeline_step_view_and_geometry_rules_reject_malformed_values():
    step = PipelineStep.from_config({
        "id": "line",
        "type": "line_counter",
        "line": [[0, 0], ["bad", 10]],
    })

    assert step.id == "line"
    assert step.type == "line_counter"
    assert not is_valid_stage_geometry(step.type, step)
    assert not is_valid_stage_geometry(
        "region_filter",
        {"polygon": [[0, 0], [10, 0]]},
    )


def test_disabled_dependency_stages_do_not_require_a_tracker():
    assert stages_missing_tracker([
        {"type": "line_counter", "enabled": False},
    ]) == []
    assert stages_missing_tracker([
        {"type": "line_counter"},
    ]) == [(1, "line_counter")]


def test_runtime_exports_pipeline_step_schema_metadata():
    from abraia.runtime import (
        PIPELINE_STAGE_DEFINITIONS,
        PipelineStep as PublicPipelineStep,
    )

    assert PublicPipelineStep is PipelineStep
    assert set(PIPELINE_STAGE_DEFINITIONS) == {
        "model",
        "filter",
        "crop",
        "tracker",
        "line_counter",
        "region_filter",
        "region_timer",
    }


def test_pipeline_schema_is_packaged_and_validates_v2_documents():
    schema_path = Path(__file__).parents[1] / "abraia" / "runtime" / "pipeline-v2.schema.json"
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("draft/2020-12/schema")

    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate({
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [{
            "type": "model",
            "model": {"kind": "yolov8", "task": "detection"},
        }],
    })
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({
            "version": 2,
            "source": {"type": "image", "src": "frame.jpg"},
            "steps": [{"id": "crop", "type": "crop"}],
        })
def test_source_type_aliases_have_one_canonical_name():
    assert normalize_source_type("images") == "image"
    assert normalize_source_type(" usb_camera ") == "camera"
    assert normalize_source_type("stream") == "stream"
