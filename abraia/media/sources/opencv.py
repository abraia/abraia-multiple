"""Image, video, camera, and stream sources backed by OpenCV."""

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import cv2

from .types import (
    IMAGE_SUFFIXES,
    infer_source_type as infer_media_source_type,
    normalize_source_type,
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
    return infer_media_source_type(src) == "stream"


def is_image(src: Any) -> bool:
    """Return whether ``src`` is an existing image file."""
    value = str(src)
    return os.path.isfile(value) and value.lower().endswith(IMAGE_EXTENSIONS)


def is_camera(src: Any) -> bool:
    """Return whether ``src`` identifies a camera device."""
    return infer_media_source_type(src) == "camera"


def infer_source_type(src: Any) -> Optional[str]:
    """Infer the runtime source type, requiring local files to exist."""
    return infer_media_source_type(
        src,
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
    camera_index = int(str(src).strip()) if is_camera(src) else None
    source_type = source_type or ("camera" if camera_index is not None else None)
    if source_type in ("camera", "rpi_camera"):
        camera_src = src if camera_index is None else camera_index
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
        camera_index = int(str(src).strip()) if is_camera(src) else None

        if is_raspberry_pi() and camera_index == 0:
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
            cap_src = src if camera_index is None else camera_index
            self.cap = cv2.VideoCapture(cap_src)
            if camera_index is not None:
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


CAMERA_RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}


def load_images(src: str) -> List[Any]:
    """Load one image or all supported images in a directory as RGB arrays."""
    path = Path(src)
    from ..images import load_image

    def read_image(image_path: Path):
        try:
            return load_image(image_path, max_size=None)
        except (OSError, ValueError):
            return None

    if is_image(str(path)):
        image = read_image(path)
        return [image] if image is not None else []
    if path.is_dir():
        images = [
            read_image(image_path)
            for image_path in sorted(path.glob("*"))
            if is_image(str(image_path))
        ]
        return [image for image in images if image is not None]
    return []


class FrameSource:
    """Common OpenCV-backed source for local media and capture devices."""

    def __init__(
        self,
        src=0,
        source_type=None,
        resolution=(1920, 1080),
        fps=30,
        target_fps=None,
        video_unpaced=False,
        stop_event=None,
    ):
        self.src = src
        self.source_type = self._source_type(src, source_type)
        self.resolution = self._camera_resolution(resolution)
        self.requested_fps = fps
        self.target_fps = target_fps
        self.video_unpaced = video_unpaced
        self.stop_event = stop_event or threading.Event()
        self.cap = None
        self.images = None
        self.width = None
        self.height = None
        self.source_fps = None
        self._capture_stop = threading.Event()
        self._capture_queue = None
        self._capture_thread = None

        if self.source_type == "images":
            self.images = load_images(str(src))
            if not self.images:
                raise ValueError(f"No readable images found in: {src}")
            self.height, self.width = self.images[0].shape[:2]
        else:
            self.cap = open_capture(
                src,
                source_type=self.source_type,
                resolution=self.resolution,
                fps=fps,
            )
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.resolution[0]
            )
            self.height = int(
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.resolution[1]
            )
        self.frame_rate = self.source_fps or fps
        if self.source_type in ("camera", "usb_camera", "rpi_camera"):
            self._capture_queue = queue.Queue(maxsize=1)
            self._capture_thread = threading.Thread(
                target=self._prefetch_camera_frames,
                name="abraia-camera-capture",
                daemon=True,
            )
            self._capture_thread.start()

    def _prefetch_camera_frames(self):
        capture = self.cap
        while not self._capture_stop.is_set() and capture is not None:
            try:
                ret, frame = read_rgb(capture)
            except Exception:
                logger.debug("Camera frame capture failed", exc_info=True)
                break
            if not ret:
                break
            try:
                self._capture_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._capture_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def _read_source_frame(self):
        if self._capture_queue is None:
            return read_rgb(self.cap)
        while not self._capture_stop.is_set():
            try:
                return True, self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                thread = self._capture_thread
                if thread is not None and not thread.is_alive():
                    return False, None
        return False, None

    @staticmethod
    def _camera_resolution(resolution) -> Tuple[int, int]:
        if isinstance(resolution, str):
            return CAMERA_RESOLUTION_MAP.get(resolution, (1280, 720))
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
        return 1920, 1080

    @staticmethod
    def _source_type(src, source_type):
        if source_type in ("image", "images"):
            return "images"
        if source_type:
            return source_type
        return infer_source_type(src)

    @property
    def has_capture(self) -> bool:
        return self.cap is not None

    @property
    def has_images(self) -> bool:
        return self.images is not None

    def isOpened(self):
        if self.has_images:
            return True
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop_id: int) -> float:
        if self.cap is not None:
            return self.cap.get(prop_id)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width or 0)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height or 0)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.frame_rate or 0)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.images or []))
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if self.cap is not None:
            return self.cap.set(prop_id, value)
        return False

    def _generate_frames(self) -> Generator[Any, None, None]:
        if self.has_images:
            for image in self.images:
                if self.stop_event.is_set() or getattr(self, "quit", False):
                    break
                yield image
            return

        source_type = self.source_type
        camera_source = source_type in ("camera", "usb_camera", "rpi_camera", "stream")
        video_source = source_type == "video"
        target_fps = max(
            float(self.target_fps or self.frame_rate or 0),
            0,
        )
        source_fps = self.source_fps or 0
        should_drop = target_fps > 0 and (source_fps == 0 or target_fps < source_fps)
        should_pace = video_source and not self.video_unpaced
        keep_period = 1.0 / target_fps if should_drop and camera_source else 0
        video_keep_period_ms = (
            1000.0 / target_fps
            if should_drop and video_source and should_pace
            else 0
        )

        next_keep_timestamp = time.monotonic()
        video_start_ms, wall_start_time = None, None
        next_keep_video_ms = None
        try:
            while self.cap is not None:
                if self.stop_event.is_set() or getattr(self, "quit", False):
                    break
                ret, frame = self._read_source_frame()
                if not ret:
                    break

                current_pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if should_pace:
                    if video_start_ms is None:
                        video_start_ms, wall_start_time = current_pos_ms, time.monotonic()
                    if video_keep_period_ms:
                        if next_keep_video_ms is None:
                            next_keep_video_ms = current_pos_ms
                        if current_pos_ms + 1e-3 < next_keep_video_ms:
                            continue
                        while current_pos_ms + 1e-3 >= next_keep_video_ms:
                            next_keep_video_ms += video_keep_period_ms
                    desired_wall_time = wall_start_time + (
                        current_pos_ms - video_start_ms
                    ) / 1000.0
                    if time.monotonic() < desired_wall_time:
                        time.sleep(max(0, desired_wall_time - time.monotonic()))

                if keep_period:
                    if time.monotonic() < next_keep_timestamp:
                        continue
                    next_keep_timestamp = time.monotonic() + keep_period
                yield frame
        finally:
            # Frame decoding may run on the Hailo producer thread. Release
            # the capture here, but leave display/writer ownership to the
            # pipeline thread, which created and presents the window.
            self._close_capture()

    def __iter__(self):
        yield from self._generate_frames()

    def __len__(self):
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def get_frame(self, frame_num):
        if self.has_images:
            try:
                return self.images[frame_num]
            except IndexError:
                return None
        if self.cap is None:
            return None
        if isinstance(self.cap, Camera) and getattr(self.cap, "picam2", None):
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = read_rgb(self.cap)
        return frame if ret else None

    def close(self):
        self._close_capture()

    def _close_capture(self):
        """Release only the input capture used by the frame iterator."""
        self._capture_stop.set()
        capture, self.cap = self.cap, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.debug("Failed to release frame source", exc_info=True)
        thread = self._capture_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)


class ImageSource(FrameSource):
    """Read one image as a single-frame source."""

    def __init__(self, src, **kwargs):
        if isinstance(src, (str, os.PathLike)) and Path(src).is_dir():
            raise ValueError("ImageSource expects one image; use ImageSequenceSource")
        super().__init__(src, source_type="image", **kwargs)


class ImageSequenceSource(FrameSource):
    """Read a directory of images in sorted filename order."""

    def __init__(self, src, **kwargs):
        super().__init__(src, source_type="images", **kwargs)


class VideoSource(FrameSource):
    """Read a prerecorded video file."""

    def __init__(self, src, **kwargs):
        super().__init__(src, source_type="video", **kwargs)


class CameraSource(FrameSource):
    """Read from a local camera handled by OpenCV or Picamera2."""

    def __init__(self, src, source_type="camera", **kwargs):
        super().__init__(src, source_type=source_type, **kwargs)


class StreamSource(FrameSource):
    """Read from a network stream handled by OpenCV."""

    def __init__(self, src, **kwargs):
        super().__init__(src, source_type="stream", **kwargs)


def _create_input_source(src, source_type, resolution, fps, video_unpaced):
    """Choose the built-in input adapter while retaining FrameSource behavior."""
    raw_type = str(source_type or "").strip().lower()
    canonical_type = normalize_source_type(raw_type)
    if not canonical_type:
        canonical_type = infer_source_type(src) or "video"
    options = {
        "resolution": resolution,
        "fps": fps,
        "video_unpaced": video_unpaced,
    }

    if canonical_type == "image":
        if raw_type in ("images", "sequence", "image_sequence") or (
            isinstance(src, (str, os.PathLike)) and Path(src).is_dir()
        ):
            return ImageSequenceSource(src, **options)
        return ImageSource(src, **options)
    if canonical_type == "video":
        return VideoSource(src, **options)
    if canonical_type == "camera":
        camera_type = (
            raw_type if raw_type in ("usb_camera", "rpi_camera") else "camera"
        )
        return CameraSource(src, source_type=camera_type, **options)
    if canonical_type == "stream":
        return StreamSource(src, **options)
    return FrameSource(
        src,
        source_type=source_type,
        resolution=resolution,
        fps=fps,
        video_unpaced=video_unpaced,
    )


__all__ = [
    "CAMERA_RESOLUTION_MAP",
    "Camera",
    "CameraSource",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "ImageSequenceSource",
    "ImageSource",
    "StreamSource",
    "VideoSource",
    "infer_source_type",
    "is_camera",
    "is_image",
    "is_raspberry_pi",
    "is_stream_url",
    "load_images",
    "open_capture",
    "read_rgb",
]
