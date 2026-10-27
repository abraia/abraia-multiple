"""Media output sinks for files and network streams."""

import logging
import shutil
import socket
import subprocess
from urllib.parse import urlparse

import cv2
import numpy as np

from ..utils.filesystem import make_dirs

logger = logging.getLogger(__name__)


class VideoWriter:
    """Write RGB frames to a video file."""

    def __init__(self, dest=None, fps=30, size=(0, 0)):
        self.out = None
        if dest:
            make_dirs(dest)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(dest, fourcc, fps, size)
            if not self.out.isOpened():
                self.out.release()
                self.out = None
                raise RuntimeError(f"Unable to open video destination: {dest}")

    def write(self, frame):
        """Write one RGB frame when a destination was configured."""
        if self.out is not None:
            self.write_bgr(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def write_bgr(self, frame):
        """Write one frame already converted to OpenCV's BGR channel order."""
        if self.out is not None:
            self.out.write(frame)

    def close(self):
        """Release the video writer; safe to call repeatedly."""
        if self.out is not None:
            try:
                self.out.release()
            except Exception:
                logger.debug("Failed to release video writer", exc_info=True)
            finally:
                self.out = None


class RTSPOutput:
    """Publish RGB frames to an RTSP ingest URL as a low-latency H.264 stream.

    An RTSP server such as MediaMTX must accept publishers at ``url``. FFmpeg
    must be installed with the ``libx264`` encoder and RTSP muxer enabled.
    """

    def __init__(self, url, fps, size, bitrate=4_000_000):
        if not isinstance(url, str) or not url.startswith("rtsp://"):
            raise ValueError("RTSP output URL must start with 'rtsp://'")
        if fps <= 0:
            raise ValueError("RTSP output FPS must be greater than zero")
        if len(size) != 2 or any(int(value) <= 0 for value in size):
            raise ValueError("RTSP output size must contain two positive values")
        if int(bitrate) <= 0:
            raise ValueError("RTSP output bitrate must be greater than zero")

        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError(
                "RTSP output requires FFmpeg with libx264. On Raspberry Pi OS, "
                "install it with `sudo apt update && sudo apt install ffmpeg`."
            )
        encoders = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
        if encoders.returncode != 0 or "libx264" not in encoders.stdout:
            raise RuntimeError(
                "FFmpeg is installed without the libx264 encoder. Install the "
                "Raspberry Pi OS ffmpeg package with `sudo apt install ffmpeg`."
            )

        address = urlparse(url)
        host = address.hostname
        port = address.port or 554
        if not host:
            raise ValueError("RTSP output URL must include a server host")
        try:
            with socket.create_connection((host, port), timeout=2):
                pass
        except OSError as error:
            raise RuntimeError(
                f"No RTSP server is reachable at {host}:{port}. Install and "
                "start MediaMTX using the Raspberry Pi OS instructions in "
                "README.md. Download the matching arm64 or armv7 release from "
                "https://github.com/bluenviron/mediamtx/releases."
            ) from error

        self.url = url
        self.width, self.height = (int(value) for value in size)
        self._closed = False
        self._stopping = False
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(float(fps)),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-g",
                str(max(1, round(float(fps)))),
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-b:v",
                str(int(bitrate)),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                url,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            bufsize=0,
        )

    def write(self, frame):
        """Encode and publish one RGB frame."""
        if self._closed or self._stopping:
            return
        if self._process.poll() is not None:
            raise RuntimeError(
                f"FFmpeg stopped while publishing RTSP stream to {self.url}"
            )

        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.shape != (
            self.height,
            self.width,
            3,
        ):
            raise ValueError(
                "RTSP output frames must be uint8 RGB and match the configured "
                "source size "
                f"({self.width}x{self.height})"
            )
        frame_bytes = memoryview(np.ascontiguousarray(frame)).cast("B")
        try:
            while frame_bytes:
                written = self._process.stdin.write(frame_bytes)
                if not written:
                    raise BrokenPipeError
                frame_bytes = frame_bytes[written:]
        except (BrokenPipeError, OSError) as error:
            if self._stopping:
                return
            raise RuntimeError(
                f"Unable to publish RTSP stream to {self.url}"
            ) from error

    def stop(self):
        """Stop the FFmpeg publisher so a blocked frame write can finish."""
        self._stopping = True
        if self._process.poll() is None:
            self._process.terminate()

    def close(self):
        """Close the publisher and wait briefly for FFmpeg to exit."""
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


__all__ = ["RTSPOutput", "VideoWriter"]
