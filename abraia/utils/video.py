import os
import cv2
import time
import logging
import threading

from typing import Any, Optional, Tuple

from .draw import (
    render_resolution,
    render_status,
)
from .filesystem import make_dirs

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False


def is_stream_url(src: Any) -> bool:
    """Return True if the input looks like a supported network stream URL."""
    src_lower = str(src).lower()
    return src_lower.startswith(("rtsp://", "http://", "https://"))


def is_video(src: Any) -> bool:
    """Return True if the input is a video file."""
    src = str(src)
    return os.path.isfile(src) and src.lower().endswith(VIDEO_SUFFIXES)


def is_image(src: Any) -> bool:
    """Return True if the input is an image file."""
    src = str(src)
    return os.path.isfile(src) and src.lower().endswith(IMAGE_EXTENSIONS)


def is_camera(src: Any) -> bool:
    """Return True if the input is a camera source."""
    return str(src).strip().isdigit()


def open_capture(
    src: Any,
    source_type: Optional[str] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: float = 30,
) -> Any:
    """Open a camera, video file, or network stream.

    Capture objects returned by this function expose OpenCV's capture API.
    Camera sources are wrapped by :class:`Camera`, which also supports the
    Raspberry Pi camera backend and returns RGB frames.
    """
    source_type = source_type or ("camera" if is_camera(src) else None)
    if source_type in ("camera", "rpi_camera"):
        camera_src = int(str(src).strip()) if is_camera(src) else src
        capture = Camera(camera_src, resolution=resolution, fps=fps)
    else:
        if source_type == "video" and not os.path.exists(src):
            raise FileNotFoundError(f"Video file not found: {src}")
        capture = cv2.VideoCapture(src)
        if source_type == "usb_camera":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            capture.set(cv2.CAP_PROP_FPS, fps)

    if not capture.isOpened():
        capture.release()
        kind = source_type or "video"
        raise RuntimeError(f"Failed to open {kind} source: {src}")
    return capture


def read_rgb(capture: Any) -> Tuple[bool, Any]:
    """Read one RGB frame from either a ``Camera`` or OpenCV capture."""
    ret, frame = capture.read()
    if not ret or frame is None:
        return False, None
    if isinstance(capture, Camera):
        return True, frame
    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class Camera:
    """Camera wrapper whose :meth:`read` method always returns RGB frames.

    Picamera2's ``RGB888`` stream is stored as ``[B, G, R]`` bytes for
    OpenCV/NumPy, despite the format name.  Normalize that buffer here so
    callers do not need to know which capture backend is active.
    """
    def __init__(self, src=0, resolution=(1920, 1080), fps=30):
        self.width, self.height = resolution
        self.fps = fps
        self.picam2 = None
        self.cap = None
        self._opened = False
        self._io_lock = threading.Lock()

        if is_raspberry_pi() and is_camera(src) and int(str(src).strip()) == 0:
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"format": "RGB888", "size": resolution},
                    controls={"FrameRate": fps}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self._opened = True
            except Exception as e:
                logger.error(f"Failed to open Picamera2 on Raspberry Pi: {e}")
                if self.picam2:
                    try:
                        self.picam2.stop()
                    except Exception:
                        pass
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None
                self._opened = False

        if not self._opened:
            cap_src = int(str(src).strip()) if is_camera(src) else src
            self.cap = cv2.VideoCapture(cap_src)
            if is_camera(src):
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            if self.cap.isOpened():
                self._opened = True
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resolution[0])
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or resolution[1])

    def isOpened(self):
        return self._opened and (self.picam2 is not None or (self.cap is not None and self.cap.isOpened()))

    def read(self):
        if not self.isOpened():
            return False, None
        with self._io_lock:
            if self.picam2:
                try:
                    frame = self.picam2.capture_array()
                    if frame is None:
                        return False, None
                    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    logger.debug("Failed to capture a Picamera2 frame", exc_info=True)
                    return False, None
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, None
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return True, frame_rgb
        return False, None

    def get(self, prop_id: int) -> float:
        if self.cap:
            return self.cap.get(prop_id)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if self.cap:
            return self.cap.set(prop_id, value)
        return False

    def release(self):
        self._opened = False
        with self._io_lock:
            if self.picam2:
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                try:
                    self.picam2.close()
                except Exception:
                    pass
                self.picam2 = None
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None


class Video:
    def __init__(self, src=0, resolution=(1920, 1080), fps=30, dest=None):
        self.out = None
        self.quit = False
        self._display_enabled = True
        self.win_name = ''
        self.cap = open_capture(src, resolution=resolution, fps=fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resolution[0])
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or resolution[1])
        self.duration = round(self.frames / self.fps, 3) if self.fps > 0 else 0
        self.frame_rate = self.fps
        if dest:
            make_dirs(dest)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(dest, fourcc, self.fps, (self.width, self.height))
            if not self.out.isOpened():
                self.out.release()
                self.out = None
                raise RuntimeError(f"Unable to open video destination: {dest}")
        self.t0 = time.time()

    def __len__(self):
        return self.frames

    def __iter__(self):
        try:
            while self.cap is not None and self.cap.isOpened():
                if self.quit:
                    break
                ret, frame = read_rgb(self.cap)
                if not ret:
                    break
                yield frame
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release capture, writer, and any display window."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                logger.debug("Failed to release video source", exc_info=True)
            finally:
                self.cap = None
        if self.out is not None:
            try:
                self.out.release()
            except Exception:
                logger.debug("Failed to release video writer", exc_info=True)
            finally:
                self.out = None
        if self.win_name:
            try:
                cv2.destroyWindow(self.win_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
            finally:
                self.win_name = ''

    def get_frame(self, frame_num):
        if self.cap is None:
            return None
        if isinstance(self.cap, Camera) and hasattr(self.cap, 'picam2') and self.cap.picam2:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = read_rgb(self.cap)
        return frame if ret else None

    def show(self, frame):
        t1 = time.time()
        display_frame = frame.copy()
        render_status(display_frame, fps=1 / (t1 - self.t0) if t1 > self.t0 else 0)
        render_resolution(display_frame)
        self.t0 = t1
        out = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        if self.out:
            self.out.write(out)
        if not self._display_enabled:
            return
        try:
            if not self.win_name:
                window_name = 'Video'
                cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
                self.win_name = window_name
            cv2.imshow(self.win_name, out)
            ch = cv2.waitKey(1) & 0xFF
            if (ch == 27 or ch == ord('q')) or cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True
        except cv2.error as error:
            # OpenCV GUI backends are optional and may reject windows created
            # from a worker thread (notably on macOS). Do not abort inference.
            logger.warning("OpenCV display unavailable: %s", error)
            self._display_enabled = False
            if self.win_name:
                try:
                    cv2.destroyWindow(self.win_name)
                except cv2.error:
                    pass
                self.win_name = ''


if __name__ == "__main__":
    video = Video()
    for frame in video:
        video.show(frame)
