"""Compatibility façade for runtime frame sources and video output."""

import time

import cv2

from ..utils.draw import render_resolution, render_status
from .capture import (
    Camera,
    IMAGE_EXTENSIONS,
    infer_source_type,
    is_camera,
    is_image,
    is_raspberry_pi,
    is_stream_url,
    is_video,
    open_capture,
    read_rgb,
)
from .frame_source import CAMERA_RESOLUTION_MAP, FrameSource, load_images
from .output import VideoOutput


class Video(FrameSource):
    """Frame source with optional file output and OpenCV preview."""

    def __init__(
        self,
        src=0,
        resolution=(1920, 1080),
        fps=30,
        dest=None,
        source_type=None,
        video_unpaced=True,
    ):
        self.out = None
        self.quit = False
        self._display_enabled = True
        self.win_name = ""
        self._output = None
        super().__init__(
            src,
            source_type=source_type,
            resolution=resolution,
            fps=fps,
            video_unpaced=video_unpaced,
        )
        self.fps = self.source_fps or fps
        self.frames = len(self.images) if self.has_images else int(
            self.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )
        self.duration = round(self.frames / self.fps, 3) if self.fps > 0 else 0
        self.frame_rate = self.fps
        try:
            self._output = VideoOutput(
                dest=dest,
                fps=self.fps,
                size=(self.width, self.height),
            )
            self.out = self._output.out
        except Exception:
            super().close()
            raise
        self.t0 = time.time()

    def __len__(self):
        return self.frames

    def __iter__(self):
        yield from super().__iter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release capture, writer, and any display window."""
        super().close()
        output, self._output = getattr(self, "_output", None), None
        if output is not None:
            output.close()
            self.out = None
            self.win_name = ""
            return
        # Keep close compatible with lightweight test doubles and subclasses
        # constructed without calling __init__.
        if self.out is not None:
            self.out.release()
            self.out = None
        if self.win_name:
            try:
                cv2.destroyWindow(self.win_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
            finally:
                self.win_name = ""

    def get_frame(self, frame_num):
        if self.cap is None:
            return None
        if isinstance(self.cap, Camera) and getattr(self.cap, "picam2", None):
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = read_rgb(self.cap)
        return frame if ret else None

    def show(self, frame):
        """Render runtime overlays, then write and/or display the frame."""
        t1 = time.time()
        display_frame = frame.copy()
        render_status(
            display_frame,
            fps=1 / (t1 - self.t0) if t1 > self.t0 else 0,
            accelerator=getattr(self, "accelerator", None),
        )
        render_resolution(display_frame)
        self.t0 = t1

        if self._output is None:
            return
        self._output.display_enabled = self._display_enabled
        self._output.win_name = self.win_name
        self._output.show(display_frame)
        self.out = self._output.out
        self.win_name = self._output.win_name
        self._display_enabled = self._output.display_enabled
        self.quit = self._output.quit


if __name__ == "__main__":
    video = Video()
    for frame in video:
        video.show(frame)


__all__ = [
    "CAMERA_RESOLUTION_MAP",
    "Camera",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "Video",
    "infer_source_type",
    "is_camera",
    "is_image",
    "is_raspberry_pi",
    "is_stream_url",
    "is_video",
    "load_images",
    "open_capture",
    "read_rgb",
]
