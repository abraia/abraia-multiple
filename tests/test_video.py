from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import pytest

from abraia.runtime.video import Camera, FrameSource, StreamSource, Video, VideoOutput, VideoSource


class FakePicamera:
    def __init__(self, frame):
        self.frame = frame

    def capture_array(self):
        return self.frame


def _camera_with_backend(**values):
    camera = Camera.__new__(Camera)
    camera.width = 1
    camera.height = 1
    camera.fps = 30
    camera.picam2 = values.get("picam2")
    camera.cap = values.get("cap")
    camera._opened = True
    camera._io_lock = threading.Lock()
    return camera


def test_picamera_read_normalizes_rgb888_to_rgb():
    # RGB888 is exposed by Picamera2 as BGR byte order for NumPy/OpenCV.
    bgr_frame = np.array([[[10, 20, 30]]], dtype=np.uint8)
    camera = _camera_with_backend(picam2=FakePicamera(bgr_frame))

    ret, frame = camera.read()

    assert ret is True
    np.testing.assert_array_equal(frame, [[[30, 20, 10]]])


def test_opencv_camera_read_also_returns_rgb():
    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, np.array([[[10, 20, 30]]], dtype=np.uint8)

    camera = _camera_with_backend(cap=FakeCapture())

    ret, frame = camera.read()

    assert ret is True
    np.testing.assert_array_equal(frame, [[[30, 20, 10]]])


def test_video_close_is_idempotent():
    class Resource:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    source, writer = Resource(), Resource()
    video = Video.__new__(Video)
    video.cap = source
    video.images = None
    video._capture_stop = threading.Event()
    video._capture_lock = threading.Lock()
    video._capture_thread_lock = threading.Lock()
    video._capture_queue = None
    video._capture_thread = None
    video._output = VideoOutput.__new__(VideoOutput)
    video._output.out = writer
    video._output.win_name = ''
    video._output.close = writer.release

    video.close()
    video.close()

    assert source.releases == 1
    assert writer.releases == 1


def test_video_output_skips_conversion_without_preview_or_writer(monkeypatch):
    output = VideoOutput()
    output.set_display_enabled(False)

    def unexpected_conversion(*_args):
        pytest.fail("Disabled video output should not convert frames")

    monkeypatch.setattr(cv2, "cvtColor", unexpected_conversion)
    output.show(np.zeros((1, 1, 3), dtype=np.uint8))


def test_sketcher_reports_missing_gui_without_waiting_for_keys(monkeypatch):
    from abraia.media.display import Sketcher

    monkeypatch.setattr("abraia.media.display._has_gui_session", lambda: False)

    def unexpected_wait(*_args):
        pytest.fail("An unopened sketch window should not wait for keys")

    monkeypatch.setattr(cv2, "waitKey", unexpected_wait)
    sketcher = Sketcher(np.zeros((1, 1, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="No graphical session"):
        sketcher.run()


def test_frame_source_rejects_empty_image_directories(tmp_path):
    with pytest.raises(ValueError, match="No readable images"):
        FrameSource(str(tmp_path))


def test_video_uses_shared_source_for_image_inputs(tmp_path):
    image_path = tmp_path / "frame.png"
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    video = Video(str(image_path), source_type="image")
    try:
        frames = list(video)
    finally:
        video.close()

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], [[[30, 20, 10]]])


@pytest.mark.parametrize("materialize", [iter, list])
def test_image_directory_decodes_lazily_and_skips_unreadable_files(
    tmp_path, monkeypatch, materialize
):
    from abraia.media.sources import opencv

    for name in ("a.png", "broken.png", "z.png"):
        (tmp_path / name).touch()
    decoded = []

    def read_image(path):
        decoded.append(path.name)
        if path.name != "broken.png":
            return np.full((1, 1, 3), ord(path.name[0]), dtype=np.uint8)

    monkeypatch.setattr(opencv, "_read_image", read_image)
    with Video(str(tmp_path), source_type="image") as video:
        assert decoded == ["a.png"]
        assert [int(frame[0, 0, 0]) for frame in materialize(video)] == [ord("a"), ord("z")]
        assert decoded == ["a.png", "broken.png", "z.png"]
        assert video.frames == len(video) == 2
        assert int(video.get_frame(1)[0, 0, 0]) == ord("z")
        assert int(video.get_frame(-1)[0, 0, 0]) == ord("z")
        assert int(video.images[1][0, 0, 0]) == ord("z")
        assert video.get_frame(2) is None


@pytest.mark.parametrize("count", [-1, float("nan"), float("inf")])
def test_stream_accepts_unknown_frame_count(monkeypatch, count):
    class Capture:
        def get(self, prop_id):
            return count if prop_id == cv2.CAP_PROP_FRAME_COUNT else 0

        def release(self):
            pass

    monkeypatch.setattr("abraia.media.sources.opencv.open_capture", lambda *_a, **_kw: Capture())
    source = StreamSource("rtsp://example.invalid/video")
    try:
        assert source.frames == len(source) == 0
    finally:
        source.close()


@pytest.mark.parametrize(
    "create_source",
    [lambda: VideoSource("clip.mp4", video_unpaced=True),
     lambda: StreamSource("rtsp://example.invalid/video")],
    ids=["video", "stream"],
)
def test_cancellation_during_read_releases_capture_once(monkeypatch, create_source):
    read_started = threading.Event()
    read_can_finish = threading.Event()

    class Capture:
        releases = 0

        def get(self, prop_id):
            assert prop_id != cv2.CAP_PROP_POS_MSEC
            return 0

        def read(self):
            read_started.set()
            assert read_can_finish.wait(2)
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

        def release(self):
            self.releases += 1

    capture = Capture()
    monkeypatch.setattr("abraia.media.sources.opencv.open_capture", lambda *_a, **_kw: capture)
    source = create_source()

    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = pool.submit(list, source)
        try:
            assert read_started.wait(2)
            close = pool.submit(source.close)
            close.result(timeout=1)
            assert capture.releases == 0
            if isinstance(source, StreamSource):
                assert frames.result(timeout=1) == []
        finally:
            read_can_finish.set()
        assert frames.result(timeout=2) == []
    source._capture_thread.join(timeout=2)
    source.close()
    assert capture.releases == 1


def test_unpaced_video_skips_timestamp_queries(monkeypatch):
    class Capture:
        frames = iter([np.zeros((1, 1, 3), dtype=np.uint8)])

        def get(self, prop_id):
            assert prop_id != cv2.CAP_PROP_POS_MSEC
            return 0

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    monkeypatch.setattr("abraia.media.sources.opencv.open_capture", lambda *_a, **_kw: Capture())

    assert len(list(VideoSource("clip.mp4", video_unpaced=True))) == 1


def test_stream_reads_all_frames_in_order_and_releases_at_end(monkeypatch):
    class Capture:
        values = iter(range(5))
        releases = 0

        def get(self, prop_id):
            return 30 if prop_id == cv2.CAP_PROP_FPS else 0

        def read(self):
            value = next(self.values, None)
            if value is None:
                return False, None
            return True, np.full((1, 1, 3), value, dtype=np.uint8)

        def release(self):
            self.releases += 1

    capture = Capture()
    monkeypatch.setattr("abraia.media.sources.opencv.open_capture", lambda *_a, **_kw: capture)

    source = StreamSource("rtsp://example.invalid/video")
    try:
        assert [int(frame[0, 0, 0]) for frame in source] == list(range(5))
    finally:
        source.close()

    assert capture.releases == 1
