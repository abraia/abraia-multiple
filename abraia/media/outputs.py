"""File-based media output sinks."""

import logging

import cv2

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


__all__ = ["VideoWriter"]
