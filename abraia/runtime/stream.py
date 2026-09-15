import cv2
import time
import queue
import logging
import threading
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple, Callable, Any

from ..utils.draw import (
    render_resolution,
    render_status,
    calculate_optimal_thickness,
    calculate_optimal_text_scale,
)
from .video import (
    is_raspberry_pi,
    infer_source_type,
    Camera,
    CAMERA_RESOLUTION_MAP,
    FrameSource,
    load_images,
    open_capture,
)
from ..utils.filesystem import make_dirs

logger = logging.getLogger(__name__)

def get_input_type(src: Any) -> Optional[str]:
    """Determine the type of input source."""
    source_type = infer_source_type(src)
    if source_type == "camera":
        return "rpi_camera" if is_raspberry_pi() else "usb_camera"
    return source_type


def open_cv_capture(src: Any, source_type: str, resolution=(1280, 720), fps=30) -> Any:
    """Open an OpenCV-compatible capture source."""
    cap = open_capture(src, source_type, resolution=resolution, fps=fps)
    logger.info("Using %s input: %s", source_type, src)
    return cap


def open_rpi_camera(resolution=(1280, 720), fps=30) -> Optional[Any]:
    """Open camera using Camera."""
    cam = Camera(src=0, resolution=resolution, fps=fps)
    if cam.isOpened():
        return cam
    cam.release()
    return None


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    return load_images(images_path)


class VideoInput(FrameSource):
    def __init__(
        self,
        input_src: str,
        batch_size: int = 1,
        resolution: Optional[str] = None,
        frame_rate: Optional[float] = None,
        video_unpaced: bool = False,
        stop_event: Optional[threading.Event] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.frame_rate = frame_rate
        source_type = get_input_type(input_src)
        if not source_type:
            raise ValueError(f"Invalid input source: {input_src}")
        super().__init__(
            input_src,
            source_type=source_type,
            resolution=resolution or (1280, 720),
            fps=frame_rate or 30,
            target_fps=frame_rate,
            video_unpaced=video_unpaced,
            stop_event=stop_event,
        )
        self.frame_rate = frame_rate
        self.input_type = self.source_type

    @staticmethod
    def _camera_resolution(resolution: Any) -> Tuple[int, int]:
        return FrameSource._camera_resolution(resolution or (1280, 720))

    def preprocess(self, input_queue: queue.Queue, preprocess_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        raw_frames, processed_frames = [], []
        try:
            for frame in self._generate_frames():
                raw_frames.append(frame)
                processed_frames.append(preprocess_fn(frame))
                if len(raw_frames) >= self.batch_size:
                    input_queue.put((raw_frames, processed_frames))
                    raw_frames, processed_frames = [], []
            if raw_frames:
                input_queue.put((raw_frames, processed_frames))
        finally:
            # Consumers must always be released, including when capture or
            # preprocessing fails in a worker thread.
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
        self._display_enabled = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Release the writer and any preview window."""
        self._close_writer()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def show(self, frame: np.ndarray, fps: float, is_capture: bool = True) -> bool:
        frame = frame.copy()
        if not hasattr(self, "thickness"):
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
                    self.video_writer = cv2.VideoWriter(
                        self.dest,
                        fourcc,
                        self.source_fps or 30.0,
                        (width, height),
                    )
                    if not self.video_writer.isOpened():
                        self.video_writer.release()
                        self.video_writer = None
                        raise RuntimeError(
                            f"Unable to open video destination: {self.dest}"
                        )
                if self.video_writer is not None:
                    self.video_writer.write(output_bgr_frame)
            else:
                path = Path(self.dest)
                out_path = path.parent / f"{path.stem}_{self.image_index}{path.suffix}"
                make_dirs(out_path)
                cv2.imwrite(str(out_path), output_bgr_frame)
                self.image_index += 1
        if not self._display_enabled:
            return True
        try:
            cv2.imshow(self.window_name, output_bgr_frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                return False
        except cv2.error:
            logger.warning(
                "OpenCV display unavailable; continuing without preview",
                exc_info=True,
            )
            self._display_enabled = False
        return True

    def _close_writer(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

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
        try:
            with self:
                while True:
                    result = output_queue.get()
                    try:
                        if result is None:
                            break
                        original_frame, inference_result = result
                        if self.stop_event.is_set():
                            continue
                        frame_with_detections = callback(
                            original_frame, inference_result, **kwargs
                        )
                        self.increment()
                        if not self.show(
                            frame_with_detections, self.fps, is_capture=is_capture
                        ):
                            self.stop_event.set()
                    finally:
                        output_queue.task_done()
        finally:
            self.stop_event.set()
