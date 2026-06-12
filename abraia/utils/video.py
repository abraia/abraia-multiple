import os
import sys
import cv2
import time
import queue
import logging
import threading
import numpy as np

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

from . import make_dirs
from .draw import (
    render_resolution,
    render_status,
    calculate_optimal_thickness,
    calculate_optimal_text_scale,
)

logger = logging.getLogger(__name__)


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd": (640, 480), "hd": (1280, 720), "fhd": (1920, 1080)
}


def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False


class PiCamera2CaptureAdapter:
    """Adapter that makes Picamera2 behave like cv2.VideoCapture."""

    def __init__(self, picam2):
        self.picam2 = picam2
        self._opened = True
        self._io_lock = threading.Lock()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        # prevent stop/close while capturing
        with self._io_lock:
            if not self._opened: # re-check after taking lock
                return False, None
            frame = self.picam2.capture_array()

        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            try:
                cfg = self.picam2.camera_configuration()
                size = cfg.get("main", {}).get("size", None)
                if size and len(size) == 2:
                    w, h = int(size[0]), int(size[1])
                    return float(w if prop_id == cv2.CAP_PROP_FRAME_WIDTH else h)
            except Exception:
                pass
            return 0.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return None

    def release(self):
        # stop new reads ASAP
        self._opened = False
        # wait if a read() is currently inside capture_array()
        with self._io_lock:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass


def open_cv_capture(src: Any, source_type: str, resolution=(1280, 720), fps=30) -> Any:
    """Open an OpenCV-based capture source."""
    if source_type == "video" and not os.path.exists(src):
        logger.error(f"File not found: {src}")
        sys.exit(1)
    cap = cv2.VideoCapture(int(src) if source_type == "usb_camera" else src)
    if source_type == "usb_camera":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        logger.error(f"Failed to open {source_type} source: {src}")
        sys.exit(1)
    logger.info(f"Using {source_type} input: {src}")
    return cap


def open_rpi_camera(resolution=(1280, 720), fps=30) -> Optional[Any]:
    """Open Raspberry Pi camera using Picamera2."""
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.error(f"Picamera2 not available: {e}")
        return None
    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": resolution, "format": "RGB888"}, controls={"FrameRate": fps})
        picam2.configure(config)
        picam2.start()
        return PiCamera2CaptureAdapter(picam2)
    except Exception as e:
        logger.error(f"Failed to open RPi camera: {e}")
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        return None


def is_stream_url(src: str) -> bool:
    """Return True if the input looks like a supported network stream URL."""
    src_lower = src.lower()
    return (src_lower.startswith("rtsp://") or src_lower.startswith("http://") or src_lower.startswith("https://"))


def is_video(src: str) -> bool:
    """Return True if the input is a video file."""
    return os.path.isfile(src) and src.lower().endswith(VIDEO_SUFFIXES)


def is_image(src: str) -> bool:
    """Return True if the input is an image file."""
    return os.path.isfile(src) and src.lower().endswith(IMAGE_EXTENSIONS)


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    path = Path(images_path)
    def read_rgb(p: Path):
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        img = read_rgb(path)
        return [img] if img is not None else []
    elif path.is_dir():
        images = [read_rgb(img) for img in path.glob("*") if img.suffix.lower() in IMAGE_EXTENSIONS]
        return [img for img in images if img is not None]
    return []


def get_input_type(src: Any) -> str:
    """Determine the type of input source."""
    src_str = str(src)
    if src_str.isdigit():
        return "rpi_camera" if is_raspberry_pi() else "usb_camera"
    if is_stream_url(src_str):
        return "stream"
    if os.path.exists(src_str):
        return "video" if is_video(src_str) else "images"


class VideoInput:
    def __init__(
        self,
        input_src: str,
        batch_size: int = 1,
        resolution: Optional[str] = None,
        frame_rate: Optional[float] = None,
        video_unpaced: bool = False,
        stop_event: Optional[threading.Event] = None,
    ):
        self.batch_size = batch_size
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.video_unpaced = video_unpaced
        self.stop_event = stop_event or threading.Event()

        self.cap = None
        self.images = None

        self.input_type = get_input_type(input_src)
        if not self.input_type:
            logger.error(f"Invalid input '{input_src}'.")
            sys.exit(1)

        self.width = None
        self.height = None
        self.source_fps = None
        if self.input_type == "images":
            self.images = load_images_opencv(input_src)
        else:
            width, height = 1280, 720
            if self.resolution in CAMERA_RESOLUTION_MAP:
                width, height = CAMERA_RESOLUTION_MAP[self.resolution]
            if self.input_type == "rpi_camera":
                self.cap = open_rpi_camera(resolution=(width, height))
                self.source_fps = 30
            else:
                self.cap = open_cv_capture(input_src, self.input_type, resolution=(width, height))
            if self.cap is not None:
                self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    @property
    def has_capture(self) -> bool:
        return self.cap is not None

    @property
    def has_images(self) -> bool:
        return self.images is not None and len(self.images) > 0

    def _generate_frames(self) -> Generator[np.ndarray, None, None]:
        if self.has_images:
            yield from self.images
            return

        is_camera = self.input_type in ("usb_camera", "rpi_camera", "stream")
        is_video = self.input_type == "video"
        target_fps = self.frame_rate or 0
        source_fps = self.source_fps or 0
        
        should_drop = target_fps > 0 and (source_fps == 0 or target_fps < source_fps)
        should_pace = is_video and not self.video_unpaced
        
        keep_period = 1.0 / target_fps if should_drop and is_camera else 0
        video_keep_period_ms = 1000.0 / target_fps if should_drop and is_video and should_pace else 0
        
        next_keep_timestamp = time.monotonic()
        video_start_ms, wall_start_time = None, None
        next_keep_video_ms = None

        while not self.stop_event.is_set():
            ret, frame_bgr = self.cap.read()
            if not ret: break
            
            current_pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if should_pace:
                if video_start_ms is None:
                    video_start_ms, wall_start_time = current_pos_ms, time.monotonic()
                if video_keep_period_ms:
                    if next_keep_video_ms is None: next_keep_video_ms = current_pos_ms
                    if current_pos_ms + 1e-3 < next_keep_video_ms: continue
                    while current_pos_ms + 1e-3 >= next_keep_video_ms: next_keep_video_ms += video_keep_period_ms
                desired_wall_time = wall_start_time + (current_pos_ms - video_start_ms) / 1000.0
                if time.monotonic() < desired_wall_time:
                    time.sleep(max(0, desired_wall_time - time.monotonic()))
            
            if keep_period:
                if time.monotonic() < next_keep_timestamp: continue
                next_keep_timestamp += keep_period
            
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.cap.release()

    def preprocess(self, input_queue: queue.Queue, preprocess_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        raw_frames, processed_frames = [], []
        for frame in self._generate_frames():
            raw_frames.append(frame)
            processed_frames.append(preprocess_fn(frame))
            if len(raw_frames) >= self.batch_size:
                input_queue.put((raw_frames, processed_frames))
                raw_frames, processed_frames = [], []
        if raw_frames:
            input_queue.put((raw_frames, processed_frames))
        input_queue.put(None)


class VideoDisplay:
    def __init__(self, dest: Optional[str] = None, source_fps: Optional[float] = None, stop_event: Optional[threading.Event] = None):
        self.dest = dest
        self.source_fps = source_fps
        self.stop_event = stop_event or threading.Event()

        self._count = 0
        self._start_time = None

        self.video_writer = None
        self.image_index = 0
        self.window_name = "Output"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def show(self, frame: np.ndarray, fps: float, is_capture: bool = True) -> bool:
        if not hasattr(self, 'thickness'):
            self.thickness = calculate_optimal_thickness(frame.shape[:2])
            self.text_scale = calculate_optimal_text_scale(frame.shape[:2])

        render_status(frame, fps, thickness=self.thickness, text_scale=self.text_scale)
        render_resolution(frame, thickness=self.thickness, text_scale=self.text_scale)

        output_bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.dest:
            if is_capture:
                if self.video_writer is None:
                    make_dirs(self.dest)
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    self.video_writer = cv2.VideoWriter(self.dest, fourcc, self.source_fps, (width, height))
                if self.video_writer:
                    self.video_writer.write(output_bgr_frame)
            else:
                path = Path(self.dest)
                out_path = path.parent / f"{path.stem}_{self.image_index}{path.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), output_bgr_frame)
                self.image_index += 1
        else:
            cv2.imshow(self.window_name, output_bgr_frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                return False
        return True

    def start(self):
        self._start_time = time.time()

    def increment(self, n: int = 1):
        self._count += n

    @property
    def count(self) -> int:
        return self._count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        return f"Processed {self.count} frames at {self.fps:.2f} FPS, Total time: {self.elapsed:.2f} seconds"

    def visualize(self, output_queue: queue.Queue, callback: Callable, is_capture: bool = True, **kwargs) -> None:
        self.start()
        with self:
            while True:
                try:
                    result = output_queue.get()
                    if result is None:
                        break
                    original_frame, inference_result = result
                    if self.stop_event.is_set():
                        continue
                    frame_with_detections = callback(original_frame, inference_result, **kwargs)
                    self.increment()
                    if not self.show(frame_with_detections, self.fps, is_capture=is_capture):
                        self.stop_event.set()
                finally:
                    output_queue.task_done()
        self.stop_event.set()


class Video:
    def __init__(self, src=0, resolution=(1920, 1080), fps=30, dest=None):
        self.out = None
        self.quit = False
        self.win_name = ''
        self.picam2 = None
        if src == 0 and is_raspberry_pi():
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_video_configuration(
                main={"format": 'RGB888', "size": resolution},
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
            self.duration = round(self.frames / self.fps, 3) if self.fps > 0 else 0
        self.frame_rate = self.fps
        if dest:
            make_dirs(dest)
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
        render_status(frame, fps=1 / (t1 - self.t0) if t1 > self.t0 else 0)
        render_resolution(frame)
        self.t0 = t1
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.out:
            self.out.write(out)
        else:
            if not self.win_name:
                self.win_name = 'Video'
                cv2.namedWindow(self.win_name, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(self.win_name, out)
            ch = cv2.waitKey(1) & 0xFF
            if (ch == 27 or ch == ord('q')) or cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True


if __name__ == "__main__":
    video = Video()
    for frame in video:
        video.show(frame)
