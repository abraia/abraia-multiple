import os
import cv2
import time
import psutil
from .draw import draw_text_multiline


def is_raspberry():
    path = '/proc/device-tree/model'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return 'Raspberry Pi' in f.read()
    return False


class Video:
    def __init__(self, src=0, resolution=(1920, 1080), fps=30, dest=None):
        self.out = None
        self.quit = False
        self.win_name = ''
        self.picam2 = None
        if src == 0 and is_raspberry():
            from picamera2 import Picamera2
            from libcamera import Transform
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_video_configuration(
                main={"format": 'BGR888', "size": resolution},
                transform=Transform(hflip=True, vflip=True),
                controls={"FrameRate": fps}))
            self.picam2.start()
            self.fps = fps
            self.width, self.height = resolution
            self.frames = 0
            self.duration = 0
        else:
            self.cap = cv2.VideoCapture(src)
            if isinstance(src, int):
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = round(self.frames / self.fps, 3)
        self.frame_rate = self.fps
        if dest:
            dirname = os.path.dirname(dest)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.out = cv2.VideoWriter(dest, fourcc, self.fps, (self.width, self.height))
        self.t0 = time.time()

    def __len__(self):
        return self.frames

    def __iter__(self):
        if self.picam2:
            while not self.quit:
                frame = self.picam2.capture_array()
                if frame is None:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.picam2.stop()
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret is False or frame is None or self.quit:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cap.release()
        if self.out:
            self.out.release()
        if self.win_name:
            cv2.destroyWindow(self.win_name)
            cv2.waitKey(1)

    def get_frame(self, frame_num):
        if self.picam2:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret is False or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def write(self, frame):
        if self.out:
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def show(self, frame):
        t1 = time.time()
        draw_text_multiline(frame, [f"FPS: {round(1 / (t1 - self.t0), 1)}", 
                                    f"CPU: {psutil.cpu_percent()}%", 
                                    f"RAM: {round(psutil.virtual_memory().used / (1024**3), 2)} GB"], (10, 40), background_color=(192, 192, 192))
        self.t0 = t1
        if not self.win_name:
            self.win_name = 'Video'
            cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(self.win_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        ch = 0xFF & cv2.waitKey(1 if self.picam2 else int(self.fps))
        if (ch == 27 or ch == ord('q')) or cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
            self.quit = True


if __name__ == "__main__":
    video = Video()
    for frame in video:
        video.show(frame)
