from types import SimpleNamespace

import numpy as np

import abraia.inference
import abraia.runtime.execution as execution


def test_run_pipeline_reports_frame_events_and_closes_runtime(monkeypatch):
    calls = {}
    frame_callbacks = []
    status_messages = []
    configuration = {
        "version": 2,
        "source": {"type": "image", "src": "frame.png"},
        "steps": [],
        "display": {"show": True},
    }

    class FakePipeline:
        def __init__(self, on_frame):
            self.model = object()
            self.on_frame = on_frame
            self.render = lambda context: context.frame.copy()
            self.closed = False

        def run(self, is_cancelled):
            assert is_cancelled() is False
            self.on_frame(SimpleNamespace(
                frame=np.zeros((8, 8, 3), dtype=np.uint8),
                frame_index=4,
                results=[{"label": "object"}],
                metrics={"inference_ms": 2.0},
            ), 3.25)

        def close(self):
            self.closed = True

    pipeline = None

    def from_dict(_cls, config, **kwargs):
        nonlocal pipeline
        calls["config"] = config
        calls["kwargs"] = kwargs
        pipeline = FakePipeline(kwargs["on_frame"])
        return pipeline

    monkeypatch.setattr(
        execution.Pipeline, "from_dict", classmethod(from_dict)
    )
    monkeypatch.setattr(
        abraia.inference, "get_model_accelerator", lambda _model: "HAILO"
    )
    monkeypatch.setattr(
        "abraia.utils.draw.render_status", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "abraia.utils.draw.render_resolution", lambda *_args, **_kwargs: None
    )

    result = execution.run_pipeline(
        configuration,
        frame_callback=frame_callbacks.append,
        is_cancelled=lambda: False,
        max_events=1,
        status_callback=status_messages.append,
    )

    assert calls["config"] is not configuration
    assert calls["config"]["display"]["show"] is False
    assert calls["kwargs"]["on_frame"]
    assert result["frame_count"] == 1
    assert result["events"] == frame_callbacks
    assert result["events"][0]["frame"] == 4
    assert result["events"][0]["detections"] == 1
    assert result["events"][0]["elapsed_ms"] == 3.25
    assert result["events"][0]["accelerator"] == "HAILO"
    assert result["events"][0]["display_frame"].shape == (8, 8, 3)
    assert status_messages and "Model and source ready" in status_messages[0]
    assert pipeline.closed
