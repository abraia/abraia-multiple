import threading

import cv2
import numpy as np
import pytest

from abraia.runtime.output import VideoOutput
from abraia.runtime.video import Camera, FrameSource, Video


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
    video.out = writer
    video.win_name = ''
    video._output = VideoOutput.__new__(VideoOutput)
    video._output.out = writer
    video._output.win_name = ''
    video._output.close = writer.release

    video.close()
    video.close()

    assert source.releases == 1
    assert writer.releases == 1


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
