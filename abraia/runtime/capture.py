"""Camera and OpenCV capture adapters used by runtime frame sources."""

import logging
import os
import threading
from typing import Any, Optional, Tuple

import cv2

from ..sources import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    infer_source_type as infer_media_source_type,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = IMAGE_SUFFIXES


def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as fileobj:
            return "Raspberry Pi" in fileobj.read()
    except Exception:
        return False


def is_stream_url(src: Any) -> bool:
    """Return whether ``src`` looks like a supported network stream URL."""
    return str(src).lower().startswith(("rtsp://", "http://", "https://"))


def is_video(src: Any) -> bool:
    """Return whether ``src`` is an existing video file."""
    value = str(src)
    return os.path.isfile(value) and value.lower().endswith(VIDEO_SUFFIXES)


def is_image(src: Any) -> bool:
    """Return whether ``src`` is an existing image file."""
    value = str(src)
    return os.path.isfile(value) and value.lower().endswith(IMAGE_EXTENSIONS)


def is_camera(src: Any) -> bool:
    """Return whether ``src`` identifies a camera device."""
    return str(src).strip().isdigit()


def infer_source_type(src: Any) -> Optional[str]:
    """Infer the runtime source type, requiring local files to exist."""
    return infer_media_source_type(
        src,
        require_exists=True,
        image_type="images",
    )


def open_capture(
    src: Any,
    source_type: Optional[str] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: float = 30,
) -> Any:
    """Open a camera, video file, or network stream."""
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
    """Read one RGB frame from either a :class:`Camera` or OpenCV capture."""
    ret, frame = capture.read()
    if not ret or frame is None:
        return False, None
    if isinstance(capture, Camera):
        return True, frame
    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class Camera:
    """Camera wrapper whose ``read`` method always returns RGB frames."""

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
                    controls={"FrameRate": fps},
                )
                self.picam2.configure(config)
                self.picam2.start()
                self._opened = True
            except Exception as error:
                logger.error("Failed to open Picamera2 on Raspberry Pi: %s", error)
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
                self.width = int(
                    self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resolution[0]
                )
                self.height = int(
                    self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or resolution[1]
                )

    def isOpened(self):
        return self._opened and (
            self.picam2 is not None
            or (self.cap is not None and self.cap.isOpened())
        )

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
            if self.cap:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, None
                return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return False, None

    def get(self, prop_id: int) -> float:
        if self.cap:
            return self.cap.get(prop_id)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_POS_FRAMES):
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


__all__ = [
    "Camera",
    "IMAGE_EXTENSIONS",
    "infer_source_type",
    "is_camera",
    "is_image",
    "is_raspberry_pi",
    "is_stream_url",
    "is_video",
    "open_capture",
    "read_rgb",
]
