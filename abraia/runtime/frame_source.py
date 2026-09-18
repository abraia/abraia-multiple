"""Image and video frame-source iteration."""

import cv2
import logging
import threading
import time

from pathlib import Path
from typing import Any, Generator, List, Tuple

from .capture import infer_source_type, is_image, open_capture, read_rgb

logger = logging.getLogger(__name__)


CAMERA_RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}


def load_images(src: str) -> List[Any]:
    """Load one image or all supported images in a directory as RGB arrays."""
    path = Path(src)

    def read_image(image_path: Path):
        image = cv2.imread(str(image_path))
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    """Shared frame source for synchronous and threaded runtimes."""

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
        return self.images is not None and len(self.images) > 0

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

        source_type = getattr(self, "source_type", getattr(self, "input_type", None))
        camera_source = source_type in ("camera", "usb_camera", "rpi_camera", "stream")
        video_source = source_type == "video"
        target_fps = max(
            float(getattr(self, "target_fps", None) or self.frame_rate or 0),
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
                ret, frame = read_rgb(self.cap)
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
            self.close()

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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = read_rgb(self.cap)
        return frame if ret else None

    def close(self):
        capture, self.cap = self.cap, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.debug("Failed to release frame source", exc_info=True)


__all__ = ["CAMERA_RESOLUTION_MAP", "FrameSource", "load_images"]
