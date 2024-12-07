'''
Sketcher.

Keys:
  SPACE - callback
  r     - reset the mask
  s     - save output
  ESC   - exit
'''

import cv2
import numpy as np

from .draw import draw_overlay_mask


class Sketcher:
    def __init__(self, img, radius=11):
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

    def on_click(self, callback):
        self.handle_click = callback

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(self.mask, self.prev_pt, pt, 255, self.radius)
            self.prev_pt = pt
        else:
            self.prev_pt = None
        if event == cv2.EVENT_LBUTTONUP:
            if self.handle_click:
                self.handle_click(pt)
        if self.prev_pt:
            self.show(self.img, self.mask)

    def run(self, callback):
        while True:
            ch = 0xFF & cv2.waitKey()
            if ch == 27 or ch == ord('q'):
                break
            if ch == ord(' '):
                self.show(callback(self.img, self.mask))
            if ch == ord('r'):
                self.load(self.img)
            if ch == ord('s'):
                cv2.imwrite('output.png', self.output)
        cv2.destroyWindow(self.win_name)
