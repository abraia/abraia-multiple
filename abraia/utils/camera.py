import os
import cv2


class Camera:
    def __init__(self, src=0, resolution=(1920, 1080), fps=30, dest=None):
        from picamera2 import Picamera2
        self.out = None
        self.quit = False
        self.win_name = ''
        self.width, self.height = resolution
        self.fps = fps
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(
            main={"format": 'BGR888', "size": (self.width, self.height)},
            controls={"FrameRate": self.fps}))
        self.picam2.start()
        if dest:
            dirname = os.path.dirname(dest)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.out = cv2.VideoWriter(dest, fourcc, self.fps, (self.width, self.height))

    def __iter__(self):
        while not self.quit:
            frame = self.picam2.capture_array()
            if frame is None:
                break
            yield frame
        self.picam2.stop()
        if self.out:
            self.out.release()
        if self.win_name:
            cv2.destroyWindow(self.win_name)
            cv2.waitKey(1)

    def write(self, frame):
        if self.out:
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def show(self, frame):
        if not self.win_name:
            self.win_name = 'Camera'
            cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(self.win_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        ch = 0xFF & cv2.waitKey(1)
        if (ch == 27 or ch == ord('q')) or cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
            self.quit = True


if __name__ == "__main__":
    camera = Camera()
    for frame in camera:
        camera.show(frame)
