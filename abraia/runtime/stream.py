import os
import cv2
import time
import queue
import logging
import threading
import numpy as np

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

from ..utils.draw import (
    render_resolution,
    render_status,
    calculate_optimal_thickness,
    calculate_optimal_text_scale,
)
from .video import (
    is_raspberry_pi,
    is_stream_url,
    is_video,
    is_image,
    is_camera,
    Camera,
    open_capture,
    read_rgb,
)
from ..utils.filesystem import make_dirs

logger = logging.getLogger(__name__)

CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd": (640, 480), "hd": (1280, 720), "fhd": (1920, 1080)
}


def get_input_type(src: Any) -> Optional[str]:
    """Determine the type of input source."""
    src_str = str(src)
    if is_camera(src):
        return "rpi_camera" if is_raspberry_pi() else "usb_camera"
    if is_stream_url(src_str):
        return "stream"
    if os.path.isdir(src_str):
        return "images"
    if os.path.isfile(src_str):
        if is_video(src_str):
            return "video"
        if is_image(src_str):
            return "images"
    return None


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
    path = Path(images_path)

    def read_image(p: Path):
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    if is_image(str(path)):
        img = read_image(path)
        return [img] if img is not None else []
    elif path.is_dir():
        images = [
            read_image(img)
            for img in sorted(path.glob("*"))
            if is_image(str(img))
        ]
        return [img for img in images if img is not None]
    return []


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
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.resolution = self._camera_resolution(resolution)
        self.frame_rate = frame_rate
        self.video_unpaced = video_unpaced
        self.stop_event = stop_event or threading.Event()

        self.cap = None
        self.images = None

        self.input_type = get_input_type(input_src)
        if not self.input_type:
            raise ValueError(f"Invalid input source: {input_src}")

        self.width = None
        self.height = None
        self.source_fps = None
        if self.input_type == "images":
            self.images = load_images_opencv(input_src)
            if not self.images:
                raise ValueError(f"No readable images found in: {input_src}")
        else:
            width, height = self.resolution
            if self.input_type == "rpi_camera":
                self.cap = open_rpi_camera(resolution=(width, height))
                self.source_fps = 30
            else:
                self.cap = open_cv_capture(
                    input_src,
                    self.input_type,
                    resolution=(width, height),
                    fps=self.frame_rate or 30,
                )
            if self.cap is None:
                raise RuntimeError(f"Unable to open {self.input_type} source: {input_src}")
            if self.cap is not None:
                self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    @staticmethod
    def _camera_resolution(resolution: Any) -> Tuple[int, int]:
        if isinstance(resolution, str):
            return CAMERA_RESOLUTION_MAP.get(resolution, (1280, 720))
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
        return 1280, 720

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

        camera_source = self.input_type in ("usb_camera", "rpi_camera", "stream")
        video_source = self.input_type == "video"
        target_fps = max(float(self.frame_rate or 0), 0)
        source_fps = self.source_fps or 0

        should_drop = target_fps > 0 and (source_fps == 0 or target_fps < source_fps)
        should_pace = video_source and not self.video_unpaced

        keep_period = 1.0 / target_fps if should_drop and camera_source else 0
        video_keep_period_ms = (
            1000.0 / target_fps
            if should_drop and video_source and should_pace
            else 0
        )

        next_keep_timestamp = time.monotonic()
        video_start_ms, wall_start_time = None, None
        next_keep_video_ms = None

        try:
            while not self.stop_event.is_set():
                ret, frame = read_rgb(self.cap)
                if not ret:
                    break

                current_pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                if should_pace:
                    if video_start_ms is None:
                        video_start_ms, wall_start_time = current_pos_ms, time.monotonic()
                    if video_keep_period_ms:
                        if next_keep_video_ms is None:
                            next_keep_video_ms = current_pos_ms
                        if current_pos_ms + 1e-3 < next_keep_video_ms:
                            continue
                        while current_pos_ms + 1e-3 >= next_keep_video_ms:
                            next_keep_video_ms += video_keep_period_ms
                    desired_wall_time = wall_start_time + (current_pos_ms - video_start_ms) / 1000.0
                    if time.monotonic() < desired_wall_time:
                        time.sleep(max(0, desired_wall_time - time.monotonic()))

                if keep_period:
                    if time.monotonic() < next_keep_timestamp:
                        continue
                    next_keep_timestamp = time.monotonic() + keep_period

                yield frame
        finally:
            self.close()

    def close(self) -> None:
        """Release the capture, if this input owns one."""
        capture, self.cap = self.cap, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                logger.debug("Failed to release input capture", exc_info=True)

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
