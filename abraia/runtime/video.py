"""Compatibility video API backed by the reusable media package."""

import time

import cv2

from ..media.display import Window
from ..media.outputs import VideoWriter
from ..media.sources.opencv import (
    CAMERA_RESOLUTION_MAP,
    Camera,
    CameraSource,
    FrameSource,
    IMAGE_EXTENSIONS,
    ImageSequenceSource,
    ImageSource,
    StreamSource,
    VideoSource,
    _create_input_source,
    is_camera,
    is_image,
    is_raspberry_pi,
    is_stream_url,
    load_images,
    open_capture,
    read_rgb,
)
from .sources import infer_source_type as infer_runtime_source_type
from ..utils.draw import render_resolution, render_status

def infer_source_type(src):
    """Infer a runtime source type while requiring local files to exist."""
    return infer_runtime_source_type(
        src,
        src,
        require_exists=True,
        image_type="images",
    )


class OpenCVDisplay:
    """Show frames in a window and report user quit requests to the pipeline."""

    def __init__(self):
        self.window = Window("Video")
        self.win_name = ""
        self.display_enabled = True
        self._display_started = False
        self.quit = False

    def show(self, frame):
        """Display one RGB frame."""
        if self.display_enabled:
            self._update_display(self.window.show(frame))

    def _show_bgr(self, frame):
        """Display a frame already converted to OpenCV's BGR order."""
        if self.display_enabled:
            self._update_display(self.window.show_bgr(frame))

    def _update_display(self, visible):
        if self.window.error:
            self.quit = self._display_started
            self.display_enabled = False
            self.win_name = ""
            return
        self._display_started = True
        self.win_name = self.window.name
        if not visible:
            self.quit = True

    def close(self):
        """Close the preview window; safe to call repeatedly."""
        self.window.close()
        self.win_name = ""

    def set_display_enabled(self, enabled: bool):
        """Enable or disable the preview window."""
        self.display_enabled = bool(enabled)


class VideoOutput(OpenCVDisplay):
    """Combine runtime preview with media file output."""

    def __init__(self, dest=None, fps=30, size=(0, 0)):
        super().__init__()
        self._writer = VideoWriter(dest=dest, fps=fps, size=size)
        self.out = self._writer.out

    def show(self, frame):
        output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write_bgr(output)
        self._show_bgr(output)

    def close(self):
        self._writer.close()
        self.out = None
        super().close()


class Video(FrameSource):
    """Compatibility facade combining a media source and video output."""

    def __init__(
        self,
        src=0,
        resolution=(1920, 1080),
        fps=30,
        dest=None,
        source_type=None,
        video_unpaced=True,
    ):
        self._output = None
        self._input = _create_input_source(
            src,
            source_type,
            self._camera_resolution(resolution),
            fps,
            video_unpaced,
        )
        self.fps = self._input.source_fps or fps
        self.frames = len(self._input.images) if self._input.has_images else int(
            self._input.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )
        self.duration = round(self.frames / self.fps, 3) if self.fps > 0 else 0
        self.frame_rate = self.fps
        try:
            self._output = VideoOutput(
                dest=dest,
                fps=self.fps,
                size=(self.width, self.height),
            )
        except Exception:
            self._input.close()
            raise
        self.t0 = None

    def __getattr__(self, name):
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            return getattr(input_source, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_") or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return
        input_source = self.__dict__.get("_input")
        if input_source is not None and hasattr(input_source, name):
            setattr(input_source, name, value)
            return
        object.__setattr__(self, name, value)

    def __iter__(self):
        input_source = self.__dict__.get("_input")
        return iter(input_source) if input_source is not None else super().__iter__()

    def __len__(self):
        return self.frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Release capture, writer, and any display window."""
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            input_source.close()
        else:
            super().close()
        output, self._output = self._output, None
        if output is not None:
            output.close()

    def set_display_enabled(self, enabled: bool):
        """Configure preview visibility without exposing sink internals."""
        if self._output is not None:
            self._output.set_display_enabled(enabled)

    def get(self, prop_id: int) -> float:
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            return input_source.get(prop_id)
        return super().get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            return input_source.set(prop_id, value)
        return super().set(prop_id, value)

    def isOpened(self):
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            return input_source.isOpened()
        return super().isOpened()

    def get_frame(self, frame_num):
        input_source = self.__dict__.get("_input")
        if input_source is not None:
            return input_source.get_frame(frame_num)
        return super().get_frame(frame_num)

    def show(self, frame):
        """Render runtime overlays, then write and/or display the frame."""
        self._show(frame, copy=True)

    def show_rendered(self, frame):
        """Display a writable frame returned by the built-in pipeline renderer."""
        self._show(frame, copy=False)

    def _show(self, frame, copy):
        t1 = time.perf_counter()
        display_frame = frame.copy() if copy else frame
        render_status(
            display_frame,
            fps=1 / (t1 - self.t0) if self.t0 is not None and t1 > self.t0 else 0,
            accelerator=getattr(self, "accelerator", None),
        )
        render_resolution(display_frame)
        self.t0 = t1

        if self._output is not None:
            self._output.show(display_frame)

    @property
    def quit(self):
        return self._output is not None and self._output.quit

    @quit.setter
    def quit(self, value):
        if self._output is not None:
            self._output.quit = bool(value)


if __name__ == "__main__":
    video = Video()
    for frame in video:
        video.show(frame)


__all__ = [
    "CAMERA_RESOLUTION_MAP",
    "Camera",
    "CameraSource",
    "FrameSource",
    "IMAGE_EXTENSIONS",
    "ImageSequenceSource",
    "ImageSource",
    "StreamSource",
    "Video",
    "VideoOutput",
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
