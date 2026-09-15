import cv2
import numpy as np

from typing import Any, Callable, Optional
from .draw import draw_overlay_mask


class Sketcher:
    """
    Window Sketcher.

    Keys:
      s     - save & exit
      r     - reset
      ESC   - exit
    """

    def __init__(self, img: np.ndarray, radius: int = 7):
        self.window = Window('Image')
        self.radius = radius
        self.handle_click: Optional[Callable[[list], Any]] = None
        self.element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        self.window.on_click(self.on_mouse)
        self.load(img)

    def load(self, img: np.ndarray) -> None:
        self.img = img
        self.mask = np.zeros(img.shape[:2], np.uint8)
        self.show(self.img)

    def dilate(self, mask: np.ndarray) -> np.ndarray:
        return cv2.dilate(mask, self.element)

    def show(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if mask is not None:
            img = draw_overlay_mask(img, mask, (255, 0, 0), 0.5)
        self.output = img
        self.window.show(img)

    def on_click(self, callback: Callable[[list], Any]) -> None:
        self.handle_click = callback

    def on_mouse(self, point: list) -> None:
        if self.handle_click:
            res = self.handle_click(point)
            if isinstance(res, tuple):
                self.show(res[0], res[1])
            else:
                self.show(res)

    def run(self) -> np.ndarray:
        out = self.img
        while True:
            ch = 0xFF & cv2.waitKey()
            if ch in (27, ord('q')):
                out = self.img
                break
            if ch == ord('r'):
                self.load(self.img)
            if ch == ord('s'):
                out = self.output
                break
        self.window.close()
        return out


class Window:
    """Class to show RGB images using cv2 with mouse click support."""

    def __init__(self, name: str = "Window"):
        self.name = name
        self.opened = False
        self.handle_click: Optional[Callable[[list], Any]] = None
        self.error: Optional[str] = None

    def __enter__(self) -> 'Window':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def show(self, img: np.ndarray, delay: int = 1) -> bool:
        """Show an RGB image (numpy array)."""
        try:
            if not self.opened:
                cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL)
                self.opened = True
                self.error = None
                if self.handle_click:
                    cv2.setMouseCallback(self.name, self._on_mouse)

            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
            cv2.imshow(self.name, bgr_img)
            return self.poll(delay)
        except Exception as error:
            self.error = str(error)
            self.close()
            return False

    def poll(self, delay: int = 1) -> bool:
        """Process window events and report whether the window is visible."""
        if not self.opened:
            return False
        try:
            key = cv2.waitKey(delay) & 0xFF
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
            if key in (27, ord('q')) or visible < 1:
                self.close()
                return False
            return True
        except Exception as error:
            self.error = str(error)
            self.close()
            return False

    def on_click(self, callback: Callable[[list], Any]) -> None:
        """Set mouse click callback."""
        self.handle_click = callback
        if self.opened:
            cv2.setMouseCallback(self.name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.handle_click:
            res = self.handle_click([x, y])
            if res is not None:
                self.show(res[0] if isinstance(res, tuple) else res, delay=1)

    def close(self) -> None:
        if self.opened:
            try:
                cv2.destroyWindow(self.name)
                cv2.waitKey(1)
            except Exception:
                pass
            self.opened = False
