import os
import sys
import cv2
import time
import logging
import threading
import numpy as np

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

from .draw import (
    render_resolution,
    render_status,
)

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


def make_dirs(dest):
    """Create directory if it doesn't exist."""
    dirname = os.path.dirname(dest)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False


def is_stream_url(src: str) -> bool:
    """Return True if the input looks like a supported network stream URL."""
    src_lower = src.lower()
    return (src_lower.startswith("rtsp://") or src_lower.startswith("http://") or src_lower.startswith("https://"))


def is_video(src: str) -> bool:
    """Return True if the input is a video file."""
    return os.path.isfile(src) and src.lower().endswith(VIDEO_SUFFIXES)


def is_image(src: str) -> bool:
    """Return True if the input is an image file."""
    return os.path.isfile(src) and src.lower().endswith(IMAGE_EXTENSIONS)


def is_camera(src: Any) -> bool:
    """Return True if the input is a camera source."""
    src_str = str(src)
    return src_str.isdigit()


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

        if is_raspberry_pi() and (src == 0 or src == '0'):
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
                self.picam2 = None
                self._opened = False

        if not self._opened:
            cap_src = int(src) if isinstance(src, str) and src.isdigit() else src
            self.cap = cv2.VideoCapture(cap_src)
            if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
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
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            cam_src = int(src) if isinstance(src, str) else src
            self.cap = Camera(src=cam_src, resolution=resolution, fps=fps)
        else:
            self.cap = cv2.VideoCapture(src)
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
        self.t0 = time.time()

    def __len__(self):
        return self.frames

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is False or frame is None or self.quit:
                break
            if isinstance(self.cap, Camera):
                yield frame
            else:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.cap.release()
        if self.out:
            self.out.release()
        if self.win_name:
            cv2.destroyWindow(self.win_name)
            cv2.waitKey(1)

    def get_frame(self, frame_num):
        if isinstance(self.cap, Camera) and hasattr(self.cap, 'picam2') and self.cap.picam2:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret is False or frame is None:
            return None
        if isinstance(self.cap, Camera):
            return frame
        else:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def show(self, frame):
        t1 = time.time()
        render_status(frame, fps=1 / (t1 - self.t0) if t1 > self.t0 else 0)
        render_resolution(frame)
        self.t0 = t1
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
