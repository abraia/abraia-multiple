'''
Magic Eraser.

Keys:
  s     - save & exit
  r     - reset
  ESC   - exit
'''

import cv2
import numpy as np

from .draw import draw_overlay_mask


class Sketcher:
    def __init__(self, img, radius=7):
        print(__doc__)
        self.prev_pt = None
        self.win_name = 'Image'
        self.radius = radius
        self.load(img)
        self.element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        cv2.setMouseCallback(self.win_name, self.on_mouse)
        
    def load(self, img):
        self.img = img
        self.prev_pt = None
        self.mask = np.zeros(img.shape[:2], np.uint8)
        self.show(self.img)

    def dilate(self, mask):
        return cv2.dilate(mask, self.element)

    def show(self, img, mask=None):
        if mask is not None:
            img = draw_overlay_mask(img, mask, (255, 0, 0), 0.5)
        self.output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.win_name, self.output)
        cv2.waitKey(1)

    def on_click(self, callback):
        self.handle_click = callback

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.handle_click:
                self.show(self.handle_click([x, y]))

    def run(self):
        out = self.img
        while True:
            ch = 0xFF & cv2.waitKey()
            if ch == 27 or ch == ord('q'):
                out = self.img
                break
            if ch == ord('r'):
                self.load(self.img)
            if ch == ord('s'):
                out = self.output
                break
        cv2.destroyWindow(self.win_name)
        return out
