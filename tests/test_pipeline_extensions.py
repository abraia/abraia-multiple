"""Exercise extensions through the SDK public API without application imports."""

from types import SimpleNamespace

import numpy as np
import pytest

from abraia.inference import ModelDescriptor, ModelSpec, register_model_factory
from abraia.runtime import (
    Pipeline,
    PipelineDraft,
    infer_source_type,
    register_model_step_factory,
    register_source_factory,
)


@pytest.fixture(autouse=True)
def isolate_extensions(monkeypatch):
    from abraia.inference import model_config, registry
    from abraia.runtime import composition, sources

    monkeypatch.setattr(registry, "_MODEL_FACTORIES", registry._MODEL_FACTORIES.copy())
    monkeypatch.setattr(composition, "_MODEL_STEP_FACTORIES", {})
    monkeypatch.setattr(sources, "_SOURCE_FACTORIES", {})
    monkeypatch.setattr(sources, "_SOURCE_SUFFIXES", {})
    descriptors = model_config._MODEL_DESCRIPTORS.copy()
    yield
    model_config._MODEL_DESCRIPTORS.clear()
    model_config._MODEL_DESCRIPTORS.update(descriptors)


@pytest.mark.parametrize("outcome", ("complete", "cancel", "failure"))
def test_extension_receives_opaque_data_and_releases_resources(outcome):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    payload = object()
    calls = []

    class Source:
        frame_rate = 1

        def __init__(self, path, **options):
            calls.append(("source", path, options["source_type"]))

        def __iter__(self):
            yield {"frame": image, "data": {"extension_payload": payload}}

        def set_display_enabled(self, enabled):
            assert enabled is False

        def close(self):
            calls.append("source closed")

    class Step:
        def __init__(self, step_id, model, model_kwargs, *, primary):
            self.model = model
            assert primary
            assert model_kwargs == {}

        def __call__(self, context):
            assert context.frame is image
            assert context.data["extension_payload"] is payload
            if outcome == "failure":
                raise RuntimeError("extension failed")
            context.results = [{"label": "extension result"}]
            return context

        def close(self):
            calls.append("step closed")
            self.model.close()

    def make_model(spec, config, base_dir, session_options):
        assert spec.kind == "test_extension"
        return SimpleNamespace(close=lambda: calls.append("model closed"))

    register_source_factory("test_source", Source, suffixes=(".example",))
    register_model_factory(
        "test_extension", make_model,
        descriptor=ModelDescriptor(frozenset({"classification"}), follow_up=True),
    )
    register_model_step_factory("test_extension", Step)
    config = {
        "version": 2,
        "source": {"type": "image", "src": "input.example"},
        "steps": [{"type": "model", "model": {
            "kind": "test_extension", "task": "classification",
        }}],
        "display": {"show": False},
    }
    assert PipelineDraft.from_dict(config).validation_errors() == []
    assert infer_source_type("input.EXAMPLE") == "image"
    with Pipeline.from_dict(config) as pipeline:
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="extension failed"):
                pipeline.run()
        else:
            result = pipeline.run(is_cancelled=lambda: outcome == "cancel")
            if outcome == "cancel":
                assert result is None
            else:
                assert result.results == [{"label": "extension result"}]
    assert calls.count("model closed") == 1
    assert calls.count("step closed") == 1
    assert "source closed" in calls


def test_source_registration_can_replace_suffixes_and_use_explicit_type():
    from abraia.runtime.sources import create_source

    register_source_factory("test_source", lambda *_a, **_kw: "old", suffixes=(".old",))
    register_source_factory("test_source", lambda *_a, **_kw: "new", suffixes=(".new",))

    assert infer_source_type("input.old") is None
    assert infer_source_type("input.new") == "image"
    assert create_source("input.new", {"source_type": "image"}) == "new"
    assert create_source("input", {"source_type": "test_source"}) == "new"


def test_model_step_construction_failure_closes_unwrapped_model_and_source():
    closed = []
    source = SimpleNamespace(close=lambda: closed.append("source"), frame_rate=1)
    model = SimpleNamespace(close=lambda: closed.append("model"))
    register_source_factory("test_source", lambda *_a, **_kw: source)
    register_model_factory(
        "test_extension", lambda *_a: model,
        descriptor=ModelDescriptor(frozenset({"classification"})),
    )

    def fail(*args, **kwargs):
        raise ValueError("step construction failed")

    register_model_step_factory("test_extension", fail)
    with pytest.raises(ValueError, match="step construction failed"):
        Pipeline.from_dict({
            "version": 2,
            "source": {"type": "test_source", "src": "input"},
            "steps": [{"type": "model", "model": {
                "kind": "test_extension", "task": "classification",
            }}],
        })
    assert sorted(closed) == ["model", "source"]


def test_registered_model_capabilities_apply_to_follow_up_validation():
    config = {
        "version": 2,
        "source": {"type": "image", "src": "input.jpg"},
        "steps": [
            {"type": "model", "model": {"kind": "yolov8"}},
            {"type": "model", "model": {
                "kind": "test_extension", "task": "classification",
            }},
        ],
    }
    for follow_up in (False, True):
        register_model_factory(
            "test_extension", lambda *_a: None,
            descriptor=ModelDescriptor(frozenset({"classification"}), follow_up=follow_up),
        )
        errors = PipelineDraft.from_dict(config).validation_errors()
        assert bool(errors) is not follow_up
    assert ModelSpec.from_config({
        "kind": "test_extension", "task": "detection",
    }).validation_error()[0] == "task"
