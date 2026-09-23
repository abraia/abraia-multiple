import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest

from abraia.runtime import (
    AsyncInferenceRunner,
    FrameBatch,
    FrameContext,
    FrameRecord,
    FrameResult,
    LineCounterStage,
    Pipeline,
    RegionFilterStage,
    TrackerStage,
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


def test_pipeline_runs_model_stages_and_display_in_order():
    model = FakeModel()
    display = FakeDisplay()
    calls = []

    def stage(context):
        calls.append((context.frame, context.frame_time))
        context.results[0]["processed"] = True
        return context

    last = Pipeline(
        source=FakeSource(),
        model=model,
        stages=[stage],
        display=display,
        render=lambda context: (context.frame, context.results[0]["processed"]),
        model_kwargs={"labels": ["person"]},
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


def test_pipeline_run_without_cancellation_callback_uses_noop_cancellation():
    pipeline = Pipeline(source=FakeSource(), model=FakeModel())

    assert pipeline.run() is not None


def test_pipeline_honors_cancellation_before_processing_next_frame():
    cancelled = [False]

    class Source:
        frame_rate = 30

        def __iter__(self):
            yield "frame-1"
            cancelled[0] = True
            yield "frame-2"

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

    def stage(context):
        seen.append(context.frame_index)
        return context

    last = Pipeline(
        source=["frame-1", "frame-2"],
        model=AsyncModel(),
        stages=[stage],
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


def test_tracker_stage_updates_context_results():
    class Tracker:
        def update(self, results):
            results[0]["track_id"] = 7
            return results

    context = FrameContext(frame=None, frame_index=0, frame_time=0, results=[{}])
    TrackerStage(Tracker())(context)

    assert context.results[0]["track_id"] == 7


def test_line_counter_stage_exposes_metrics():
    class Counter:
        def update(self, results):
            return 3, 2

    context = FrameContext(frame=None, frame_index=0, frame_time=0, results=[])
    LineCounterStage(Counter())(context)

    assert context.metrics["line_counter"] == {"in": 3, "out": 2}


def test_region_filter_stage_keeps_named_views():
    class RegionFilter:
        def update(self, results):
            return [results[0]], [results[1]]

    inside, outside = {"id": 1}, {"id": 2}
    context = FrameContext(
        frame=None,
        frame_index=0,
        frame_time=0,
        results=[inside, outside],
    )
    RegionFilterStage(RegionFilter())(context)

    assert context.results == [inside]
    assert context.views["in_region"] == [inside]
    assert context.views["out_region"] == [outside]


def test_pipeline_from_file_builds_components_and_resolves_paths():
    class FakeVideo:
        frame_rate = 12

        def __init__(self, src, resolution, fps, dest, source_type=None):
            self.src = src
            self.resolution = resolution
            self.fps = fps
            self.dest = dest

        def set_display_enabled(self, enabled):
            self.display_enabled = enabled

    class FakeModel:
        def __init__(self, uri, **kwargs):
            self.uri = uri

    class FakeTracker:
        def __init__(self, **kwargs):
            self.options = kwargs

    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "pipeline.json"
        config_path.write_text(json.dumps({
            "version": 1,
            "source": {"type": "image", "src": "frame.jpg"},
            "model": {"kind": "yolov8", "uri": "model.onnx", "labels": ["person"]},
            "stages": [
                {"type": "tracker", "enabled": "auto"},
                {"type": "line_counter", "line": [[0, 0], [10, 10]]},
            ],
            "display": {"show": False, "dest": "output.avi"},
        }))

        with patch("abraia.runtime.video.Video", FakeVideo), \
             patch("abraia.inference.models.detection.Model", FakeModel), \
             patch("abraia.inference.Tracker", FakeTracker):
            pipeline = Pipeline.from_file(str(config_path))

        assert pipeline.source.src == str(Path(temp_dir) / "frame.jpg")
        assert pipeline.source.dest == str(Path(temp_dir) / "output.avi")
        assert pipeline.model.uri == "model.onnx"
        assert "tracker" not in pipeline.components
        assert "line_counter" in pipeline.components


def test_pipeline_resolves_local_model_uri_relative_to_config(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    received = []

    class FakeVideo:
        frame_rate = 12

        def __init__(self, *args, **kwargs):
            self.closed = False

        def set_display_enabled(self, enabled):
            self.display_enabled = enabled

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self, uri, **kwargs):
            received.append(uri)

    config = {
        "source": {"type": "image", "src": "frame.jpg"},
        "model": {"kind": "yolov8", "uri": "model.onnx"},
        "display": {"show": False},
    }

    with patch("abraia.runtime.video.Video", FakeVideo), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config, base_dir=tmp_path)

    assert received == [str(model_path)]
    pipeline.close()


def test_pipeline_from_dict_rejects_unknown_stage():
    config = {
        "source": {"src": 0},
        "model": {"kind": "yolov8", "uri": "model.onnx"},
        "stages": [{"type": "unknown"}],
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            pass

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

    with patch("abraia.runtime.video.Video", FakeVideo), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        try:
            Pipeline.from_dict(config)
        except ValueError as error:
            assert "Unknown pipeline stage type" in str(error)
        else:
            raise AssertionError("Expected unknown stage to be rejected")


def test_pipeline_from_dict_skips_disabled_stages():
    config = {
        "source": {"src": "clip.mp4"},
        "model": {"kind": "yolov8", "uri": "model.onnx"},
        "stages": [{"type": "line_counter", "enabled": False}],
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            pass

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

    with patch("abraia.runtime.video.Video", FakeVideo), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict(config)

    assert pipeline.stages == []


def test_pipeline_from_dict_keeps_camera_indices_as_camera_sources():
    config = {
        "source": {"type": "camera", "src": "0"},
        "model": {"kind": "yolov8", "uri": "model.onnx"},
    }

    class FakeVideo:
        frame_rate = 30

        def __init__(self, src, resolution, fps, dest, source_type=None):
            self.src = src

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

    with patch("abraia.runtime.video.Video", FakeVideo), \
         patch("abraia.inference.models.detection.Model", FakeModel):
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


def test_pipeline_factory_uses_default_architecture_model_uris():
    class FakeModel:
        def __init__(self, uri):
            self.uri = uri

    with patch("abraia.inference.models.detection.Model", FakeModel):
        from abraia.inference.registry import create_model

        object_detector = create_model({"kind": "yolov8", "task": "detection"})
        segmentation_detector = create_model({"kind": "yolov8", "task": "segmentation"})

    assert object_detector.uri == "multiple/models/yolov8n.onnx"
    assert segmentation_detector.uri == "multiple/models/yolov8n-seg.onnx"


def test_pipeline_factory_uses_architecture_kind_with_task_for_default_uri():
    class FakeModel:
        def __init__(self, uri):
            self.uri = uri

    with patch("abraia.inference.models.detection.Model", FakeModel):
        from abraia.inference.registry import create_model

        segmentation = create_model({
            "kind": "yolov8",
            "task": "segmentation",
        })

    assert segmentation.uri == "multiple/models/yolov8n-seg.onnx"


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


def test_pipeline_factory_uses_task_for_architecture_model():
    class FakeModel:
        def __init__(self, uri):
            self.uri = uri

    with patch("abraia.inference.models.detection.Model", FakeModel):
        from abraia.inference.registry import create_model

        segmentation = create_model({
            "kind": "yolov8",
            "task": "segmentation",
            "uri": "multiple/models/yolov8n-seg.onnx",
        })

    assert segmentation.uri == "multiple/models/yolov8n-seg.onnx"


def test_pipeline_factory_pairs_onnx_models_with_available_hailo_models():
    from abraia.inference.registry import create_model

    class FakeHailoModel:
        def __init__(self, uri, task, **kwargs):
            self.uri = uri
            self.task = task
            self.kwargs = kwargs

    with patch("abraia.inference.hailo.pipeline.HailoPipelineModel", FakeHailoModel), \
         patch("abraia.inference.registry.hailo_device_arch", return_value="hailo8"), \
         patch("abraia.inference.registry.hailo_model_available", return_value=True):
        model = create_model({
            "kind": "yolov8",
            "uri": "multiple/models/yolov8n.onnx",
        }, accelerator="auto")

    assert model.uri == "multiple/models/yolov8n_hailo8.hef"
    assert model.task == "detection"


def test_pipeline_factory_uses_default_pose_model_uri():
    class FakeModel:
        def __init__(self, uri):
            self.uri = uri

    with patch("abraia.inference.models.detection.Model", FakeModel):
        from abraia.inference.registry import create_model

        pose_detector = create_model({"kind": "yolov8", "task": "pose"})

    assert pose_detector.uri == "multiple/models/yolov8n_pose.onnx"


def test_pipeline_factory_requires_a_resnet_classification_model_uri():
    with pytest.raises(ValueError, match="requires a model 'uri'"):
        from abraia.inference.registry import create_model

        create_model({"kind": "resnet", "task": "classification"})


def test_pipeline_factory_resolves_requested_model_size():
    class FakeModel:
        def __init__(self, uri):
            self.uri = uri

    with patch("abraia.inference.models.detection.Model", FakeModel):
        from abraia.inference.registry import create_model

        detector = create_model({"kind": "yolov8", "task": "detection", "size": "large"})

    assert detector.uri == "multiple/models/yolov8l.onnx"


def test_pipeline_from_dict_passes_pose_options_to_onnx_model():
    class FakeVideo:
        frame_rate = 30

        def __init__(self, *args, **kwargs):
            self.display_enabled = False

        def set_display_enabled(self, enabled):
            self.display_enabled = enabled

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
        "source": {"type": "image", "src": "frame.jpg"},
        "model": {
            "kind": "yolov8",
            "task": "pose",
            "uri": "multiple/models/yolov8m_pose.onnx",
            "labels": ["person"],
            "conf_threshold": 0.4,
            "iou_threshold": 0.6,
        },
        "display": {"show": False, "render_results": False},
    }

    with patch("abraia.inference.registry.create_model", return_value=model), \
         patch("abraia.runtime.video.Video", FakeVideo):
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

        def __init__(self, src, resolution, fps, dest, source_type=None):
            self.src = src
            self.dest = dest
            self.show_calls = 0
            self.display_enabled = True

        def set_display_enabled(self, enabled):
            self.display_enabled = enabled

        def __iter__(self):
            yield np.zeros((1, 1, 3), dtype=np.uint8)

        def show(self, frame):
            self.show_calls += 1

    class FakeModel:
        def __init__(self, uri, **kwargs):
            pass

        def run(self, frame, **kwargs):
            return []

    with patch("abraia.runtime.video.Video", FakeVideo), \
         patch("abraia.inference.models.detection.Model", FakeModel):
        pipeline = Pipeline.from_dict({
            "source": {"src": "frame.jpg"},
            "model": {"kind": "yolov8", "uri": "model.onnx"},
            "display": {"show": False, "dest": "output.avi"},
        })
        pipeline.run()

    assert pipeline.display is pipeline.source
    assert pipeline.source.display_enabled is False
    assert pipeline.source.show_calls == 1


def test_demo_definitions_are_pipeline_configs():
    from abraia.demo import PIPELINES

    assert PIPELINES
    for config in PIPELINES.values():
        assert config["version"] == 1
        assert "src" in config["source"]
        if config["model"].get("kind") not in ("face", "license_plate", "ocr"):
            assert "uri" in config["model"]
        assert all("type" in stage for stage in config["stages"])


def test_plate_demo_is_available_to_monitor_objects():
    from abraia import demo

    class FakePipeline:
        received = None
        components = {}

        @classmethod
        def from_dict(cls, config, on_frame=None):
            cls.received = config
            return cls()

        def run(self):
            return None

    with patch("abraia.demo.Pipeline", FakePipeline):
        demo.monitor_objects("camera", demo="plates")

    assert FakePipeline.received["source"]["src"] == "camera"
    assert FakePipeline.received["model"] == {
        "task": "recognition",
        "kind": "license_plate",
        "params": {"threshold": 0.85, "iou_threshold": 0.15},
    }


def test_hailo_accelerator_uses_shared_pipeline():
    from abraia import demo

    class FakePipeline:
        received = None
        components = {}

        @classmethod
        def from_dict(cls, config, on_frame=None, accelerator=None):
            cls.received = config
            return cls()

        def run(self):
            return None

    with patch("abraia.demo.Pipeline", FakePipeline), \
         patch("abraia.demo._hailo_device_arch", return_value="hailo8"), \
         patch("abraia.demo._hailo_model_available", return_value=True):
        demo.monitor_objects(0, demo="pose", accelerator="hailo")

    assert FakePipeline.received["model"] == {
        "task": "pose",
        "kind": "yolov8",
        "uri": "multiple/models/yolov8m_pose_hailo8.hef",
        "params": {"model_type": "v8"},
    }
    assert FakePipeline.received["source"]["src"] == 0
    assert FakePipeline.received["stages"] == [{"type": "tracker"}]


def test_hailo_pipeline_definitions_use_pipeline_schema():
    from abraia.demo import PIPELINE_DEVICES

    hailo_demos = ("detect", "tomato", "apple", "segment", "pose")
    for name in hailo_demos:
        config = PIPELINE_DEVICES[name]
        assert config["version"] == 1
        assert "source" in config
        assert "model" in config
        assert config["model"]["uri"].endswith(".onnx")
        assert "uri" in config["model"]
        assert all("type" in stage for stage in config["stages"])


def test_cli_search_command_routes_to_search_images():
    from importlib.machinery import SourceFileLoader
    from click.testing import CliRunner

    module = SourceFileLoader("workspace_cli", "scripts/abraia").load_module()
    with patch("abraia.demo.search_images") as search:
        result = CliRunner().invoke(
            module.cli,
            ["search", "my-project", "red car"],
        )

    assert result.exit_code == 0, result.output
    search.assert_called_once_with("my-project", query="red car")


def test_cli_routes_demo_accelerator_to_shared_monitor():
    from importlib.machinery import SourceFileLoader
    from click.testing import CliRunner

    module = SourceFileLoader("workspace_cli_accelerator", "scripts/abraia").load_module()
    with patch("abraia.demo.monitor_objects") as monitor:
        result = CliRunner().invoke(
            module.cli,
            [
                "run",
                "demo",
                "tomato",
                "video.mp4",
                "--accelerator",
                "hailo",
            ],
        )

    assert result.exit_code == 0, result.output
    monitor.assert_called_once_with(
        "video.mp4",
        "tomato",
        accelerator="hailo",
    )


def test_cli_rejects_the_legacy_hailo_demo_mode():
    from importlib.machinery import SourceFileLoader
    from click.testing import CliRunner

    module = SourceFileLoader("workspace_cli_legacy_hailo", "scripts/abraia").load_module()
    result = CliRunner().invoke(module.cli, ["run", "hailo", "tomato"])

    assert result.exit_code != 0
    assert "run demo <name> --accelerator hailo" in result.output


def test_cli_process_media_uses_runtime_video_and_image_helpers():
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader("workspace_cli_media", "scripts/abraia").load_module()
    with patch("abraia.utils.get_type", return_value="image/jpeg"), \
            patch("abraia.utils.load_image", return_value="decoded"), \
            patch("abraia.utils.show_image") as show_image:
        module.process_media("image.jpg", lambda image: ("processed", image))

    show_image.assert_called_once_with(("processed", "decoded"))


def test_cli_process_map_bounds_repeat_iterators():
    from importlib.machinery import SourceFileLoader
    import itertools

    module = SourceFileLoader("workspace_cli_map", "scripts/abraia").load_module()

    assert module.process_map(
        lambda value, suffix: value + suffix,
        [1, 2],
        itertools.repeat(10),
        max_workers=1,
    ) == [11, 12]


def test_cli_anonymize_dataset_passes_the_worker_callable():
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader("workspace_cli_anonymize", "scripts/abraia").load_module()
    with patch("abraia.cli.input_files", return_value=["project/image.jpg"]), \
            patch("abraia.utils.get_type", return_value="image/jpeg"), \
                patch("abraia.cli.process_map") as process_map:
        files = module.process_dataset("project", anonymize_images=True)

    assert files == ["project/image.jpg"]
    process_map.assert_called_once_with(
        module.anonymize,
        ["project/image.jpg"],
        desc="Anonymizing images",
    )


def test_cli_custom_run_prefers_explicit_source_argument():
    from importlib.machinery import SourceFileLoader

    module = SourceFileLoader("workspace_cli_source", "scripts/abraia").load_module()
    original_userid = module.abraia.userid
    module.abraia.userid = "user"
    try:
        with patch("abraia.training.list_models", return_value=["model.onnx"]), \
                patch("abraia.inference.models.detection.Model"), \
                patch("abraia.cli.process_media") as process_media:
            module.run.callback("project", "cat", "image.jpg", "auto")
    finally:
        module.abraia.userid = original_userid

    assert process_media.call_args.args[0] == "image.jpg"
