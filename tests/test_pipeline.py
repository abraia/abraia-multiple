import json
from pathlib import Path
from threading import Barrier, Event
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch

import numpy as np
import pytest

from abraia.runtime import Pipeline
from abraia.runtime.composition import ModelStep, ProcessorStep
from abraia.runtime.pipeline import FrameContext
from abraia.runtime.inference import (
    AsyncInferenceRunner,
    FrameBatch,
    FrameRecord,
    FrameResult,
)


def test_async_inference_runner_batches_and_preserves_source_order():
    calls = []

    def inference(batch: FrameBatch, emit, stop_event):
        frames = [record.frame for record in batch.records]
        calls.append(frames)
        for record in reversed(batch.records):
            emit(FrameResult(record, [{"frame": record.frame}]))

    runner = AsyncInferenceRunner(
        source=["frame-0", "frame-1", "frame-2"],
        inference=inference,
        preprocess=lambda frame: frame.upper(),
        batch_size=2,
    )

    results = list(runner)

    assert calls == [["frame-0", "frame-1"], ["frame-2"]]
    assert [result.record.index for result in results] == [0, 1, 2]
    assert [result.results[0]["frame"] for result in results] == [
        "frame-0", "frame-1", "frame-2"
    ]


def test_async_inference_runner_propagates_worker_errors_and_stops_producer():
    def inference(input_queue, output_queue, stop_event):
        raise RuntimeError("backend failed")

    runner = AsyncInferenceRunner(
        source=range(100),
        inference=inference,
        preprocess=lambda frame: frame,
        batch_size=1,
        input_queue_size=1,
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        list(runner)

    assert runner._closed


class FakeSource:
    frame_rate = 2

    def __iter__(self):
        yield "frame-1"
        yield "frame-2"


class FakeModel:
    def __init__(self):
        self.calls = []

    def run(self, frame, **kwargs):
        self.calls.append((frame, kwargs))
        return [{"box": [0, 0, 1, 1], "score": 1.0}]


class FakeDisplay:
    def __init__(self):
        self.frames = []

    def show(self, frame):
        self.frames.append(frame)


class CloseableSource:
    frame_rate = 1

    def __init__(self, frames=("frame",)):
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self):
        self.closed = True


class CloseableModel:
    def __init__(self, fail=False):
        self.closed = False
        self.fail = fail

    def run(self, frame, **kwargs):
        if self.fail:
            raise RuntimeError("inference failed")
        return []

    def close(self):
        self.closed = True


def test_pipeline_runs_model_steps_and_display_in_order():
    model = FakeModel()
    display = FakeDisplay()
    calls = []

    def step(context):
        calls.append((context.frame, context.frame_time))
        context.results[0]["processed"] = True
        return context

    last = Pipeline(
        source=FakeSource(),
        model=model,
        steps=[ModelStep("model", model, {"labels": ["person"]}), step],
        display=display,
        render=lambda context: (context.frame, context.results[0]["processed"]),
    ).run()

    assert [call[0] for call in model.calls] == ["frame-1", "frame-2"]
    assert model.calls[0][1] == {"labels": ["person"]}
    assert calls == [("frame-1", 0.0), ("frame-2", 0.5)]
    assert display.frames == [("frame-1", True), ("frame-2", True)]
    assert last.frame_index == 1


def test_pipeline_stops_when_display_requests_quit():
    source = CloseableSource(frames=("frame-1", "frame-2"))

    class AsyncModel:
        def __init__(self):
            self.frames = []
            self.closed = False

        def iter_inference(self, frames, **kwargs):
            for frame_index, frame in enumerate(frames):
                self.frames.append(frame)
                yield FrameResult(
                    FrameRecord(frame_index, frame, 0),
                    [],
                )

        def close(self):
            self.closed = True

    model = AsyncModel()

    class QuitDisplay:
        quit = False

        def show(self, frame):
            self.quit = True

    last = Pipeline(
        source=source,
        model=model,
        display=QuitDisplay(),
    ).run()

    assert last.frame == "frame-1"
    assert model.frames == ["frame-1"]
    assert source.closed
    assert model.closed


@pytest.mark.parametrize(
    "hold_image, show, has_gui_session, frame_count, expected_polls",
    [
        (True, True, True, 1, 2),
        (False, True, True, 1, 0),
        (True, False, True, 1, 0),
        (True, True, True, 2, 0),
        (True, True, False, 1, 0),
    ],
)
def test_pipeline_holds_single_image_preview_until_quit(
    tmp_path, monkeypatch, hold_image, show, has_gui_session, frame_count, expected_polls
):
    import cv2

    from abraia.media.sources.opencv import ImageSequenceSource
    from abraia.runtime.video import VideoOutput

    for index in range(frame_count):
        image_path = tmp_path / f"{index}.png"
        assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), np.uint8))
    source = ImageSequenceSource(str(tmp_path))
    display = VideoOutput(fps=source.frame_rate, size=(source.width, source.height))
    if not show:
        display.set_display_enabled(False)
    opened_windows = []
    waits = []

    def wait_key(delay):
        waits.append(delay)
        return ord("q") if waits.count(50) == 2 else -1

    monkeypatch.setattr("abraia.media.display._has_gui_session", lambda: has_gui_session)
    monkeypatch.setattr(cv2, "namedWindow", lambda name, _flags: opened_windows.append(name))
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)

    with Pipeline(source=source, model=None, display=display) as pipeline:
        assert pipeline.source is not pipeline.display
        last = pipeline.run(hold_image=hold_image)

    assert last.frame_index == frame_count - 1
    assert opened_windows == (["Video"] if show and has_gui_session else [])
    assert waits.count(50) == expected_polls


@pytest.mark.parametrize("metrics", [{}, {"render_metrics": False}])
def test_configured_pipeline_renders_inference_stats_in_recordings(monkeypatch, metrics):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    model = CloseableModel()
    model.accelerator = "GPU"
    config = {
        "version": 2,
        "steps": [{"type": "model", "model": {"kind": "yolov8"}}],
        "display": {
            "show": False, "dest": "result.avi", "render_results": False, **metrics,
        },
    }
    display = FakeDisplay()
    display.set_display_enabled = lambda enabled: None
    monkeypatch.setattr("abraia.runtime.video.VideoOutput", lambda **kwargs: display)
    monkeypatch.setattr(
        "abraia.inference.registry._create_model_from_spec", lambda *args, **kwargs: model
    )
    stats = []
    sizes = []

    def render_status(output, *, fps, accelerator):
        stats.append((fps, accelerator))
        output[0, 0] = 255

    monkeypatch.setattr("abraia.utils.draw.render_status", render_status)
    monkeypatch.setattr(
        "abraia.utils.draw.render_resolution", lambda output: sizes.append(output.shape)
    )

    Pipeline.from_dict(config, input_source=[frame, frame]).run()

    assert len(display.frames) == 2
    assert not frame.any()
    if metrics:
        assert not stats and not sizes
        assert not any(output.any() for output in display.frames)
    else:
        assert stats[0] == (None, "GPU")
        assert stats[1][0] > 0 and stats[1][1] == "GPU"
        assert sizes == [frame.shape, frame.shape]
        assert all(output[0, 0].tolist() == [255, 255, 255] for output in display.frames)


def test_pipeline_honors_cancellation_before_processing_next_frame():
    cancelled = [False]
    released = Event()

    class Source:
        frame_rate = 30

        def __iter__(self):
            yield "frame-1"
            cancelled[0] = True
            assert released.wait(timeout=3), "Cancellation did not unblock the source"
            yield "frame-2"

        def stop(self):
            released.set()

        def close(self):
            pass

    class Model:
        def __init__(self):
            self.frames = []
            self.closed = False

        def run(self, frame):
            self.frames.append(frame)
            return []

        def close(self):
            self.closed = True

    model = Model()
    last = Pipeline(Source(), model).run(is_cancelled=lambda: cancelled[0])

    assert model.frames == ["frame-1"]
    assert model.closed
    assert last.frame == "frame-1"


def test_pipeline_supports_ordered_async_model_results():
    class AsyncModel:
        def run(self, frame, **kwargs):
            raise AssertionError("The synchronous model API should not be used")

        def iter_inference(self, source, **kwargs):
            for frame_index, frame in enumerate(source):
                yield FrameResult(
                    FrameRecord(frame_index, frame, 0),
                    [{"frame": frame}],
                    4.5,
                )

        def close(self):
            pass

    seen = []

    def step(context):
        seen.append(context.frame_index)
        return context

    model = AsyncModel()
    last = Pipeline(
        source=["frame-1", "frame-2"],
        model=model,
        steps=[ModelStep("model", model), step],
    ).run()

    assert seen == [0, 1]
    assert last.frame_index == 1


def test_pipeline_passes_model_kwargs_to_async_models():
    received = {}

    class AsyncModel:
        def iter_inference(self, source, **kwargs):
            received.update(kwargs)
            yield FrameResult(
                FrameRecord(0, "frame", 0),
                [],
                1.0,
            )

    Pipeline(
        source=["frame"],
        model=AsyncModel(),
        model_kwargs={"threshold": 0.6},
    ).run()

    assert received == {"threshold": 0.6}


def test_pipeline_closes_owned_resources_after_successful_run():
    source = CloseableSource()
    model = CloseableModel()

    Pipeline(source=source, model=model).run()

    assert source.closed
    assert model.closed


def test_pipeline_closes_owned_resources_when_inference_fails():
    source = CloseableSource()
    model = CloseableModel(fail=True)

    with pytest.raises(RuntimeError, match="inference failed"):
        Pipeline(source=source, model=model).run()

    assert source.closed
    assert model.closed


def test_tracker_step_updates_context_results():
    class Tracker:
        def update(self, results):
            results[0]["track_id"] = 7
            return results

    context = FrameContext(frame=None, frame_index=0, frame_time=0, results=[{}])
    ProcessorStep("tracker", Tracker(), "tracker")(context)

    assert context.results[0]["track_id"] == 7


def test_line_counter_step_exposes_metrics():
    class Counter:
        def update(self, results):
            return 3, 2

    context = FrameContext(frame=None, frame_index=0, frame_time=0, results=[])
    ProcessorStep("counter", Counter(), "line_counter")(context)

    assert context.metrics["line_counter"] == {"in": 3, "out": 2}


def test_composed_pipeline_builds_stateful_processing_steps():
    class Video:
        frame_rate = 1

        def __iter__(self):
            yield np.zeros((12, 12, 3), dtype=np.uint8)

        def close(self):
            pass

    class Detector:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _image, **_kwargs):
            return [{"box": [1, 1, 3, 3], "label": "object", "score": 0.9}]

        def close(self):
            pass

    class Tracker:
        def __init__(self, **_kwargs):
            pass

        def update(self, results):
            return [dict(result, track_id=1) for result in results]

        def close(self):
            pass

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [
            {"type": "model", "model": {
                "kind": "yolov8", "task": "detection", "uri": "detector.onnx"
            }},
            {"type": "tracker"},
            {"type": "line_counter", "line": [[0, 0], [10, 10]]},
            {"type": "region_filter", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"type": "region_timer", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        ],
        "display": {"show": False},
    }

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: Video()), \
         patch("abraia.inference.models.detection.Model", Detector), \
         patch("abraia.inference.tracker.Tracker", Tracker):
        pipeline = Pipeline.from_dict(config)
        context = pipeline.run()

    assert [step.operation for step in pipeline.steps[1:]] == [
        "tracker", "line_counter", "region_filter", "region_timer"
    ]
    assert context.results[0]["track_id"] == 1
    assert context.metrics["line_counter"] == {"in": 0, "out": 0}
    assert context.metrics["region"]["count"] == 1


def test_region_filter_step_keeps_named_views():
    def filter_results(results):
        return [results[0]], [results[1]]

    inside, outside = {"id": 1}, {"id": 2}
    context = FrameContext(
        frame=None,
        frame_index=0,
        frame_time=0,
        results=[inside, outside],
    )
    ProcessorStep("region", filter_results, "region_filter")(context)

    assert context.results == [inside]
    assert context.views["in_region"] == [inside]
    assert context.views["out_region"] == [outside]


def test_pipeline_builds_source_and_model_concurrently():
    ready_together = Barrier(2)

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *_args, **_kwargs):
            ready_together.wait(timeout=3)

        def close(self):
            pass

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            ready_together.wait(timeout=3)

        def close(self):
            pass

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [{"type": "model", "model": {
            "kind": "yolov8", "task": "detection", "uri": "model.onnx"
        }}],
        "display": {"show": False},
    }

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config, accelerator="cpu")

    assert isinstance(pipeline.model, FakeModel)
    pipeline.close()


def test_pipeline_from_file_builds_components_and_resolves_paths():
    class FakeVideo:
        frame_rate = 12

        def __init__(self, src, resolution, fps, source_type=None):
            self.src = src
            self.resolution = resolution
            self.fps = fps
            self.width, self.height = resolution

    class FakeModel:
        def __init__(self, uri, **kwargs):
            self.uri = uri

    class FakeTracker:
        def __init__(self, **kwargs):
            self.options = kwargs

    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "pipeline.json"
        config_path.write_text(json.dumps({
            "version": 2,
            "source": {"type": "image", "src": "frame.jpg"},
            "steps": [
                {
                    "id": "model",
                    "type": "model",
                    "model": {
                        "kind": "yolov8",
                        "uri": "model.onnx",
                        "labels": ["person"],
                    },
                },
                {"id": "tracker", "type": "tracker"},
                {
                    "id": "line_counter",
                    "type": "line_counter",
                    "line": [[0, 0], [10, 10]],
                },
            ],
            "display": {"show": False, "dest": "output.avi"},
        }))

        with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda src, options: FakeVideo(
                src, options["resolution"], options["fps"], options["source_type"]
        )), \
             patch("abraia.runtime.video.VideoOutput") as video_output, \
             patch("abraia.inference.models.detection.Model", FakeModel), \
             patch("abraia.inference.tracker.Tracker", FakeTracker):
            pipeline = Pipeline.from_file(str(config_path))

        assert pipeline.source.src == str(Path(temp_dir) / "frame.jpg")
        video_output.assert_called_once_with(
            dest=str(Path(temp_dir) / "output.avi"), fps=12, size=(1920, 1080)
        )
        assert pipeline.model.uri == "model.onnx"
        assert isinstance(pipeline.steps[1], ProcessorStep)
        assert pipeline.steps[1].operation == "tracker"
        assert isinstance(pipeline.steps[2], ProcessorStep)
        assert pipeline.steps[2].operation == "line_counter"
        assert "tracker" in pipeline.components
        assert "line_counter" in pipeline.components


def test_pipeline_resolves_local_model_uri_relative_to_config(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    received = []

    class FakeVideo:
        frame_rate = 12

        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self, uri, **kwargs):
            received.append(uri)

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [{
            "id": "model",
            "type": "model",
            "model": {"kind": "yolov8", "uri": "model.onnx"},
        }],
        "display": {"show": False},
    }

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config, base_dir=tmp_path)

    assert received == [str(model_path)]
    pipeline.close()


def test_pipeline_from_dict_rejects_unsupported_versions():
    with pytest.raises(ValueError, match="Only version 2"):
        Pipeline.from_dict({"version": 3})


def test_pipeline_from_dict_rejects_unknown_step():
    config = {
        "version": 2,
        "source": {"src": 0},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "uri": "model.onnx"},
            },
            {"id": "unknown", "type": "unknown"},
        ],
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            pytest.fail("Invalid step must be rejected before source creation")

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pytest.fail("Invalid step must be rejected before model creation")

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        try:
            Pipeline.from_dict(config)
        except ValueError as error:
            assert "Unknown composed pipeline step type" in str(error)
        else:
            raise AssertionError("Expected unknown step to be rejected")


def test_pipeline_closes_resources_if_renderer_factory_fails():
    closed = []

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            closed.append("source")

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            closed.append("model")

    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [{"type": "model", "model": {
            "kind": "yolov8", "task": "detection", "uri": "model.onnx"
        }}],
        "display": {"show": False},
    }

    def fail_renderer(_renderer, _config):
        raise RuntimeError("renderer setup failed")

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        with pytest.raises(RuntimeError, match="renderer setup failed"):
            Pipeline.from_dict(
                config,
                accelerator="cpu",
                renderer_factory=fail_renderer,
            )

    assert closed.count("source") == 1
    assert closed.count("model") == 1


def test_pipeline_from_dict_skips_disabled_steps():
    config = {
        "version": 2,
        "source": {"src": "clip.mp4"},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "uri": "model.onnx"},
            },
            {"id": "line_counter", "type": "line_counter", "enabled": False},
        ],
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            pass

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config)

    assert len(pipeline.steps) == 1


def test_pipeline_from_dict_rejects_invalid_model_spec():
    config = {
        "version": 2,
        "source": {"src": 0},
        "steps": [{
            "id": "model",
            "type": "model",
            "model": {"kind": "unknown_model", "uri": "model.onnx"},
        }],
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            pytest.fail("Invalid model must be rejected before source creation")

        def close(self):
            pass

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()):
        with pytest.raises(ValueError, match="Unknown detector kind"):
            Pipeline.from_dict(config)


def test_pipeline_from_dict_rejects_tracker_dependent_step_ordering():
    config = {
        "version": 2,
        "source": {"src": 0},
        "steps": [
            {
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "uri": "model.onnx"},
            },
            {
                "id": "counter",
                "type": "line_counter",
                "line": [[0, 0], [10, 10]],
            },
        ],
    }

    with pytest.raises(ValueError, match="requires an earlier tracker"):
        Pipeline.from_dict(config)


def test_pipeline_from_dict_keeps_camera_indices_as_camera_sources():
    config = {
        "version": 2,
        "source": {"type": "camera", "src": "0"},
        "steps": [{
            "id": "model",
            "type": "model",
            "model": {"kind": "yolov8", "uri": "model.onnx"},
        }],
    }

    class FakeFrameSource:
        frame_rate = 30

        def __init__(self, src):
            self.src = src

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

    with patch(
        "abraia.runtime.pipeline.create_frame_source",
        side_effect=lambda src, _kwargs: FakeFrameSource(src),
    ), patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config)

    assert pipeline.source.src == 0


def test_pipeline_factory_supports_builtin_face_detector_without_uri():
    class FakeRetinaface:
        def __init__(self, prob_threshold, iou_threshold):
            self.options = prob_threshold, iou_threshold

        def run(self, frame):
            return [{"box": [1, 2, 3, 4], "score": 0.9, "label": "face"}]

        def close(self):
            pass

    with patch("abraia.inference.models.faces.Retinaface", FakeRetinaface):
        from abraia.inference.registry import create_model

        detector = create_model({
            "task": "detect",
            "kind": "face",
            "params": {"prob_threshold": 0.8, "iou_threshold": 0.4},
        })
        results = detector.run("frame")

    assert results == [{"box": [1, 2, 3, 4], "score": 0.9, "label": "face"}]
    assert detector.options == (0.8, 0.4)
    detector.close()


def test_pipeline_factory_supports_builtin_license_plate_detector():
    class FakePlateDetector:
        def __init__(self, threshold, iou_threshold, out_size):
            self.options = threshold, iou_threshold, out_size

        def run(self, frame):
            return [{
                "box": [1, 2, 3, 4],
                "score": 0.8,
                "points": [],
                "label": "license_plate",
            }]

    with patch("abraia.inference.models.plates.LicensePlateDetector", FakePlateDetector):
        from abraia.inference.registry import create_model

        detector = create_model({
            "kind": "license_plate",
            "params": {"threshold": 0.85, "iou_threshold": 0.15},
        })

    assert detector.options == (0.85, 0.15, 300)
    assert detector.run("frame")[0]["label"] == "license_plate"


def test_pipeline_factory_resolves_architecture_model_uris():
    from abraia.inference.registry import create_model

    class FakeModel:
        def __init__(self, uri, kind):
            self.uri = uri
            self.kind = kind

    cases = [
        ({"kind": "yolov8", "task": "detection"}, "multiple/models/yolov8n.onnx"),
        ({"kind": "yolov8", "task": "segmentation"}, "multiple/models/yolov8n-seg.onnx"),
        ({"kind": "yolov8", "task": "pose"}, "multiple/models/yolov8n_pose.onnx"),
        ({"kind": "yolov8", "task": "detection", "size": "large"}, "multiple/models/yolov8l.onnx"),
        ({"kind": "rtmdet", "task": "detection", "size": "medium"}, "multiple/models/rtmdet_medium_v1.onnx"),
        ({"kind": "rtmdet", "task": "segmentation", "size": "large"}, "multiple/models/rtmdet_ins_large_v1.onnx"),
        ({"kind": "yolov8", "task": "segmentation", "uri": "multiple/models/yolov8n-seg.onnx"}, "multiple/models/yolov8n-seg.onnx"),
    ]
    with patch("abraia.inference.models.detection.Model", FakeModel):
        for config, expected_uri in cases:
            model = create_model(config)
            assert model.uri == expected_uri, config
            assert model.kind == config["kind"]


def test_pipeline_factory_infers_hailo_backend_from_architecture_kind_and_uri():
    class FakeHailoModel:
        def __init__(self, uri, task, **kwargs):
            self.uri = uri
            self.task = task

    with patch(
        "abraia.inference.hailo.pipeline.HailoPipelineModel",
        FakeHailoModel,
    ):
        from abraia.inference.registry import create_model

        model = create_model({
            "kind": "yolov8",
            "task": "pose",
            "uri": "multiple/models/yolov8m_pose_hailo8.hef",
        })

    assert model.uri.endswith(".hef")
    assert model.task == "pose"


def test_pipeline_factory_pairs_onnx_models_with_available_hailo_models():
    from abraia.inference.registry import create_model

    class FakeHailoModel:
        def __init__(self, uri, task, **kwargs):
            self.uri = uri
            self.task = task
            self.kwargs = kwargs

    with patch("abraia.inference.hailo.pipeline.HailoPipelineModel", FakeHailoModel), \
         patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8"), \
         patch("abraia.inference.models.detection.Model") as onnx_model, \
         patch("abraia.utils.remote.remote_available", return_value=True) as available:
        model = create_model({
            "kind": "yolov8",
            "uri": "multiple/models/yolov8n.onnx",
            "conf_threshold": 0.4,
            "iou_threshold": 0.6,
        }, accelerator="auto")

    assert model.uri == "multiple/models/yolov8n_hailo8.hef"
    assert model.task == "detection"
    assert model.kwargs == {
        "kind": "yolov8",
        "score_threshold": 0.4,
        "iou_threshold": 0.6,
    }
    onnx_model.assert_not_called()
    available.assert_called_once_with("multiple/models/yolov8n_hailo8.hef")


def test_saved_pipeline_requires_hailo_for_each_active_model_or_uses_auto():
    from abraia.inference.model_config import ModelSpec
    from abraia.runtime.pipeline import _create_pipeline_models

    configs = [{"kind": "yolov8"}, {"kind": "ocr"}]
    plans = [(config, ModelSpec.from_config(config)) for config in configs]
    with patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8"), \
         patch("abraia.inference.registry.hailo_artifact_available", return_value=True), \
         patch("abraia.inference.hailo.pipeline.HailoPipelineModel") as hailo, \
         patch("abraia.inference.models.ocr.TextSystem") as onnx:
        with pytest.raises(RuntimeError, match="Hailo required for model 'ocr'"):
            _create_pipeline_models(plans, None, "hailo")
        hailo.assert_not_called()
        onnx.assert_not_called()
        models = _create_pipeline_models(plans, None, "auto")
        assert [(entry[0], entry[2].backend) for entry in models] == [
            (hailo.return_value, "hailo"), (onnx.return_value, "onnx"),
        ]


def test_pipeline_factory_requires_a_resnet_classification_model_uri():
    with pytest.raises(ValueError, match="requires a model 'uri'"):
        from abraia.inference.registry import create_model

        create_model({"kind": "resnet", "task": "classification"})


def test_pipeline_from_dict_passes_pose_options_to_onnx_model():
    class FakeVideo:
        frame_rate = 30

        def __iter__(self):
            yield np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            pass

    class FakeModel:
        def __init__(self):
            self.calls = []

        def run(self, frame, **kwargs):
            self.calls.append((frame, kwargs))
            return [{
                "label": "person",
                "score": 0.9,
                "box": [0, 0, 10, 10],
                "keypoints": np.zeros((17, 2)),
                "joint_scores": np.ones(17),
            }]

        def close(self):
            pass

    model = FakeModel()
    config = {
        "version": 2,
        "source": {"type": "image", "src": "frame.jpg"},
        "steps": [{
            "id": "model",
            "type": "model",
            "model": {
                "kind": "yolov8",
                "task": "pose",
                "uri": "multiple/models/yolov8m_pose.onnx",
                "labels": ["person"],
                "conf_threshold": 0.4,
                "iou_threshold": 0.6,
            },
        }],
        "display": {"show": False, "render_results": False},
    }

    with patch("abraia.inference.registry._create_model_from_spec", return_value=model), \
         patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()):
        pipeline = Pipeline.from_dict(config)
        pipeline.run()

    assert len(model.calls) == 1
    assert model.calls[0][0].shape == (8, 8, 3)
    assert model.calls[0][1] == {
        "labels": ["person"],
        "conf_threshold": 0.4,
        "iou_threshold": 0.6,
    }


def test_pipeline_factory_passes_canonical_hailo_task_from_model_field():
    class FakeHailoModel:
        def __init__(self, uri, task, **kwargs):
            self.uri = uri
            self.task = task
            self.kwargs = kwargs

    with patch(
        "abraia.inference.hailo.pipeline.HailoPipelineModel",
        FakeHailoModel,
    ):
        from abraia.inference.registry import create_model

        model = create_model({
            "kind": "yolov8",
            "task": "pose",
            "uri": "yolov8m_pose.hef",
        })
        segmentation = create_model({
            "kind": "yolov8",
            "task": "segment",
            "uri": "yolov8n_seg.hef",
        })

    assert model.task == "pose"
    assert segmentation.task == "segmentation"


def test_pipeline_factory_rejects_legacy_hailo_model_kind():
    from abraia.inference.registry import create_model

    with pytest.raises(ValueError, match="Unknown detector kind"):
        create_model({
            "task": "segmentation",
            "kind": "hailo_segmentation",
            "uri": "yolov8n_seg",
        })


def test_pipeline_factory_supports_face_recognition_with_json_index():
    class FakeFaceRecognizer:
        def __init__(self, index, threshold):
            self.index = index
            self.threshold = threshold

        def run(self, frame):
            return [{
                "box": [1, 2, 3, 4],
                "score": 0.9,
                "label": self.index[0]["name"],
                "identity_score": self.threshold,
            }]

    with patch("abraia.inference.models.faces.FaceRecognizer", FakeFaceRecognizer):
        from abraia.inference.registry import create_model

        detector = create_model({
            "task": "recognition",
            "kind": "face",
            "params": {
                "index": [{"name": "alice", "vector": [1, 0, 0]}],
                "threshold": 0.6,
            },
        })
        results = detector.run("frame")

    assert results[0]["label"] == "alice"
    assert results[0]["score"] == 0.9
    assert results[0]["identity_score"] == 0.6


def test_pipeline_factory_supports_plate_recognition():
    class FakePlateRecognizer:
        def __init__(self, threshold, iou_threshold, out_size):
            self.options = threshold, iou_threshold, out_size

        def run(self, frame):
            return [{"box": [1, 2, 3, 4], "score": 0.9, "text": "1234ABC"}]

    with patch("abraia.inference.models.plates.PlateRecognizer", FakePlateRecognizer):
        from abraia.inference.registry import create_model

        detector = create_model({
            "task": "recognition",
            "kind": "license_plate",
            "params": {"threshold": 0.85, "iou_threshold": 0.15},
        })

    assert detector.options == (0.85, 0.15, 300)
    assert detector.run("frame")[0]["text"] == "1234ABC"


def test_pipeline_factory_supports_ocr_recognition():
    class FakeTextSystem:
        def __init__(self, drop_score):
            self.drop_score = drop_score

        def run(self, frame):
            return [{
                "box": [10, 20, 30, 20],
                "text": "AB123",
                "score": 0.92,
                "label": "AB123",
            }]

        def close(self):
            pass

    with patch("abraia.inference.models.ocr.TextSystem", FakeTextSystem):
        from abraia.inference.registry import create_model

        recognizer = create_model({
            "task": "recognition",
            "kind": "ocr",
            "params": {"drop_score": 0.7},
        })
        results = recognizer.run("frame")

    assert results[0]["label"] == "AB123"
    assert results[0]["box"] == [10, 20, 30, 20]
    assert recognizer.drop_score == 0.7


def test_pipeline_context_manager_closes_builtin_detector():
    class FakeModel:
        def __init__(self):
            self.closed = False

        def run(self, frame):
            return []

        def close(self):
            self.closed = True

    model = FakeModel()
    with Pipeline(source=[], model=model) as pipeline:
        assert pipeline.model is model
    assert model.closed


def test_pipeline_keeps_file_output_enabled_when_preview_is_disabled():
    class FakeVideo:
        frame_rate = 1
        width = 1
        height = 1
        resolution = (1, 1)

        def __iter__(self):
            yield np.zeros((1, 1, 3), dtype=np.uint8)

    class FakeVideoOutput:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.display_enabled = True
            self.show_calls = 0

        def set_display_enabled(self, enabled):
            self.display_enabled = enabled

        def show(self, _frame):
            self.show_calls += 1

        def close(self):
            pass

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

        def run(self, frame, **kwargs):
            return []

    with patch("abraia.runtime.pipeline.create_frame_source", side_effect=lambda *_args: FakeVideo()), \
         patch("abraia.runtime.video.VideoOutput", FakeVideoOutput), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict({
            "version": 2,
            "source": {"src": "frame.jpg"},
            "steps": [{
                "id": "model",
                "type": "model",
                "model": {"kind": "yolov8", "uri": "model.onnx"},
            }],
            "display": {"show": False, "dest": "output.avi"},
        })
        pipeline.run()

    assert pipeline.display is not pipeline.source
    assert Path(pipeline.display.options["dest"]).name == "output.avi"
    assert pipeline.display.display_enabled is False
    assert pipeline.display.show_calls == 1


def test_yolo9_demo_variant_pairs_with_available_hailo_model(public_demo_configs):
    from abraia.demo import resolve_pipeline
    from abraia.runtime.config import iter_pipeline_steps
    from abraia.runtime.pipeline import _create_pipeline_models, _model_plans

    config = public_demo_configs["detect"]
    config["steps"][0]["model"].update(labels=["person"], conf_threshold=0.4)
    with patch("abraia.demo.load_public_pipeline", return_value=config), \
         patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8"), \
         patch("abraia.inference.registry.hailo_artifact_available", return_value=True) as available, \
         patch("abraia.inference.hailo.yolov9.HailoYOLOv9Model") as adapter:
        selected = resolve_pipeline("detect", model="yolo9t")
        plans = _model_plans(iter_pipeline_steps(selected["steps"]))
        (_, run_options, spec), = _create_pipeline_models(plans, None, "hailo")

    uri = "multiple/models/yolo9t_hailo8.hef"
    assert spec.uri == uri
    assert run_options == {"labels": ["person"]}
    adapter.assert_called_once_with(uri, score_threshold=0.4)
    available.assert_called_once_with(uri, "hailo8")


def test_cli_search_command_routes_to_search_images():
    from click.testing import CliRunner
    import abraia.cli as module

    with patch("abraia.demo.search_images") as search:
        result = CliRunner().invoke(
            module.cli,
            ["search", "my-project", "red car"],
        )

    assert result.exit_code == 0, result.output
    search.assert_called_once_with("my-project", query="red car")


@pytest.mark.parametrize("accelerator,expected", [("hailo", "hailo"), ("auto", "onnx")])
def test_cli_routes_apple_demo_accelerator_to_pipeline(public_demo_configs, accelerator, expected):
    from click.testing import CliRunner
    import abraia.cli as module

    with patch("abraia.demo.load_public_pipeline", return_value=public_demo_configs["apple"]), \
            patch("abraia.runtime.Pipeline.from_dict") as build:
        result = CliRunner().invoke(
            module.cli,
            [
                "run",
                "demo",
                "apple",
                "video.mp4",
                "--accelerator",
                accelerator,
                "--output",
                "rendered.avi",
            ],
        )

    assert result.exit_code == 0, result.output
    config = build.call_args.args[0]
    assert config["source"]["src"] == "video.mp4"
    assert config["display"]["dest"] == "rendered.avi"
    assert config["steps"][0]["model"]["uri"].endswith(".onnx")
    assert build.call_args.kwargs["accelerator"] == expected


def test_cli_routes_face_demo_output_to_tracker():
    from click.testing import CliRunner
    import abraia.cli as module

    with patch("abraia.demo.track_faces") as track_faces:
        result = CliRunner().invoke(
            module.cli,
            ["run", "demo", "faces", "video.mp4", "--output", "rendered.avi"],
        )

    assert result.exit_code == 0, result.output
    track_faces.assert_called_once_with(
        "video.mp4", accelerator="auto", output="rendered.avi"
    )


def test_cli_routes_project_output_to_pipeline_config():
    from click.testing import CliRunner
    import abraia.cli as module

    class FakePipeline:
        config = None
        components = {}

        @classmethod
        def from_dict(
            cls, config, accelerator="auto", on_frame=None,
        ):
            cls.config = config
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def run(self, *, hold_image=False):
            return None

    config = {
        "version": 2,
        "source": {"type": "video", "src": "input.mp4"},
        "steps": [],
        "display": {"show": True},
    }
    with patch.object(module.abraia, "load_json", return_value=config), \
            patch.object(module.abraia, "download_file", return_value="/cache/input.mp4"), \
            patch("abraia.runtime.Pipeline", FakePipeline):
        result = CliRunner().invoke(
            module.cli,
            ["run", "my-project", "--output", "rendered.avi"],
        )

    assert result.exit_code == 0, result.output
    assert FakePipeline.config["display"]["dest"] == "rendered.avi"


def test_cli_run_uses_the_saved_pipeline():
    import abraia.cli as module

    config = {
        "source": {"type": "video", "src": "saved.mp4"},
        "steps": [{"model": {"kind": "yolov8", "uri": "model.onnx"}}],
    }
    with patch.object(module.abraia, "load_json", return_value=config) as load_json, \
            patch.object(module.abraia, "download_file", return_value="/cache/saved.mp4") as materialize, \
            patch("abraia.runtime.Pipeline") as pipeline:
        module.run.callback("project", "", None, "auto")

    load_json.assert_called_once_with("project/pipeline.json")
    materialize.assert_called_once_with("project/saved.mp4")
    assert config["source"]["src"] == "/cache/saved.mp4"
    pipeline.from_dict.assert_called_once_with(config, accelerator="auto", on_frame=ANY)
    pipeline.from_dict.return_value.__enter__.return_value.run.assert_called_once_with(
        hold_image=True
    )


def test_cli_process_map_bounds_repeat_iterators():
    import itertools
    import abraia.cli as module

    assert module.process_map(
        lambda value, suffix: value + suffix,
        [1, 2],
        itertools.repeat(10),
    ) == [11, 12]


def test_cli_custom_run_overrides_the_saved_pipeline_source():
    import abraia.cli as module

    config = {
        "source": {"type": "video", "src": "saved.mp4"},
        "steps": [{"model": {"kind": "yolov8", "uri": "model.onnx"}}],
    }
    with patch.object(module.abraia, "load_json", return_value=config), \
            patch("abraia.runtime.Pipeline") as pipeline:
        module.run.callback("project", "cat", "image.jpg", "auto")

    passed_config = pipeline.from_dict.call_args.args[0]
    assert passed_config["source"] == {"type": "image", "src": "image.jpg"}
