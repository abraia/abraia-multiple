import cv2
import numpy as np


class Sketcher:
    def __init__(self, src, radius=15):
        print(__doc__)
        self.img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        self.img_msk = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.prev_pt = None
        self.win_name = 'Image'
        self.dests = [self.img_msk, self.mask]
        self.colors = [(255, 255, 255), 255]
        self.radius = radius
        self.dirty = False
        self.show(self.img_msk)
        cv2.setMouseCallback(self.win_name, self.on_mouse)

    def show(self, img):
        cv2.imshow(self.win_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors):
                cv2.line(dst, self.prev_pt, pt, color, self.radius)
            self.dirty = True
            self.prev_pt = pt
            self.show(self.img_msk)
        else:
            self.prev_pt = None

    def run(self, callback):
        while True:
            ch = 0xFF & cv2.waitKey()
            if ch == 27 or ch == ord('q'):
                break
            if ch == ord(' '):
                self.show(callback(self.img, self.mask))
            if ch == ord('r'):
                self.img_msk[:] = self.img
                self.mask[:] = 0
                self.show(self.img_msk)
        cv2.destroyWindow(self.win_name)
