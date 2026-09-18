"""Video writing and optional OpenCV display output."""

import logging

import cv2

from ..utils.filesystem import make_dirs

logger = logging.getLogger(__name__)


class VideoOutput:
    """Own a video writer and an optional OpenCV preview window."""

    def __init__(self, dest=None, fps=30, size=(0, 0)):
        self.out = None
        self.win_name = ""
        self.display_enabled = True
        self.quit = False
        if dest:
            make_dirs(dest)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(dest, fourcc, fps, size)
            if not self.out.isOpened():
                self.out.release()
                self.out = None
                raise RuntimeError(f"Unable to open video destination: {dest}")

    def show(self, frame):
        """Write an RGB frame and, when enabled, display it in a window."""
        output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.out:
            self.out.write(output)
        if not self.display_enabled:
            return
        try:
            if not self.win_name:
                self.win_name = "Video"
                cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(self.win_name, output)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (27, ord("q"))
                or cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                self.quit = True
        except cv2.error as error:
            logger.warning("OpenCV display unavailable: %s", error)
            self.display_enabled = False
            if self.win_name:
                try:
                    cv2.destroyWindow(self.win_name)
                except cv2.error:
                    pass
                self.win_name = ""

    def close(self):
        """Release the writer and preview window; safe to call repeatedly."""
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
                self.win_name = ""


__all__ = ["VideoOutput"]
