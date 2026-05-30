from __future__ import annotations
import json
import os
import sys
import cv2
import time
import queue
import threading
import numpy as np

from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Callable, Any

from .core import (
    COCO_LABELS,
    IMAGE_EXTENSIONS,
    VIDEO_SUFFIXES,
    CAMERA_RESOLUTION_MAP,
    is_raspberry_pi
)
import logging
from ..utils.draw import render_resolution, render_status


logger = logging.getLogger(__name__)


class PiCamera2CaptureAdapter:
    """
    Adapter that makes Picamera2 behave like cv2.VideoCapture.
    """

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


class CapProcessingMode(str, Enum):
    """
    Capture processing modes.

    Defines how frames are read from the source and fed into the pipeline,
    based on source type and user options (saving output, target FPS, etc.).
    """

    # Camera modes
    CAMERA_NORMAL = "camera_normal"           # Process camera frames as they arrive (real-time)
    CAMERA_FRAME_DROP = "camera_frame_drop"   # Drop camera frames to match the requested target FPS

    # Video modes
    VIDEO_PACE = "video_pace"                         # Normal video playback pacing (based on original video FPS)
    VIDEO_UNPACED = "video_unpaced"                   # Run video as fast as processing allows (no pacing)
    VIDEO_PACED_AND_FRAME_DROP = "video_paced_and_frame_drop" # Paced playback but skip frames to match a lower requested FPS


def select_cap_processing_mode(
    input_type: str,
    frame_rate: Optional[float],
    source_fps: Optional[float],
    video_unpaced: bool = False,
) -> Optional[CapProcessingMode]:
    """
    Select the capture processing mode based on input type and user settings.
    """
    is_camera = input_type in ("usb_camera", "rpi_camera", "stream")
    is_video = input_type == "video"
    has_target_fps = frame_rate is not None and frame_rate > 0
    has_source_fps = source_fps is not None and source_fps > 0

    if not (is_camera or is_video):
        return None

    if is_video and video_unpaced:
        if has_target_fps:
            logger.warning(
                "--frame-rate is ignored when --video-unpaced is enabled."
            )
        return CapProcessingMode.VIDEO_UNPACED

    if has_target_fps and has_source_fps and frame_rate >= source_fps:
        logger.warning(
            f"Requested frame rate ({frame_rate}) is greater than or equal to "
            f"the source FPS ({source_fps}); no frame dropping will be applied."
        )
        return (
            CapProcessingMode.CAMERA_NORMAL
            if is_camera
            else CapProcessingMode.VIDEO_PACE
        )

    if is_camera:
        return (
            CapProcessingMode.CAMERA_FRAME_DROP
            if has_target_fps
            else CapProcessingMode.CAMERA_NORMAL
        )

    return (
        CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP
        if has_target_fps
        else CapProcessingMode.VIDEO_PACE
    )


def get_source_fps(cap: Any, source_name: str) -> Optional[float]:
    """
    Read FPS from an opened capture source.
    """
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        logger.debug(f"{source_name} FPS not reported by source.")
        return None
    return source_fps


def open_cv_capture(src: Any, source_type: str) -> Any:
    """
    Open an OpenCV-based capture source.
    """
    if source_type == "video" and not os.path.exists(src):
        logger.error(f"File not found: {src}")
        sys.exit(1)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Failed to open {source_type} source: {src}")
        sys.exit(1)

    logger.info(f"Using {source_type} input: {src}")
    return cap


def _apply_resolution_and_validate(
    cap: Any,
    resolution: Optional[str],
) -> Any:
    """
    Apply requested resolution and validate that the capture source
    produces frames.
    """
    if resolution in CAMERA_RESOLUTION_MAP:
        width, height = CAMERA_RESOLUTION_MAP[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.debug(f"Camera resolution forced to {width}x{height}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        logger.error("Camera opened but produced no frames.")
        sys.exit(1)

    return cap


def open_usb_camera(input_src: str, resolution: Optional[str]):
    camera_index = int(str(input_src))
    cap = open_cv_capture(camera_index, "USB camera index")
    return _apply_resolution_and_validate(cap, resolution)


def open_rpi_camera() -> Optional[Any]:
    """
    Open Raspberry Pi camera using Picamera2.
    """
    try:
        from picamera2 import Picamera2
    except Exception as e:
        logger.error(f"Picamera2 not available: {e}")
        return None

    try:
        picam2 = Picamera2()
        width, height = 800, 600
        fps = 30
        main = {"size": (width, height), "format": "RGB888"}
        config = picam2.create_video_configuration(main=main, controls={"FrameRate": fps})

        picam2.configure(config)
        picam2.start()

        logger.debug(f"RPi camera started ({width}x{height}) @ {fps} FPS")

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
    """
    Return True if the input looks like a supported network stream URL.
    """
    src_lower = src.lower()
    return (
        src_lower.startswith("rtsp://")
        or src_lower.startswith("http://")
        or src_lower.startswith("https://")
    )


class InputType(Enum):
    USB_CAMERA = "usb_camera"
    RPI_CAMERA = "rpi_camera"
    STREAM = "stream"
    VIDEO = "video"
    IMAGES = "images"
    UNKNOWN = "unknown"


@dataclass
class VisualizationSettings:
    output_dir: str
    save_stream_output: bool = False
    output_resolution: Optional[Tuple[int, int]] = None
    side_by_side: bool = False


class VideoPipeline:
    def __init__(
        self,
        input_src: str,
        batch_size: int = 1,
        resolution: Optional[str] = None,
        frame_rate: Optional[float] = None,
        video_unpaced: bool = False
    ):
        self.input_src = input_src.strip()
        self.batch_size = batch_size
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.video_unpaced = video_unpaced

        self.input_type = InputType.UNKNOWN
        self.cap = None
        self.images = None
        self.source_fps = None
        self.cap_processing_mode = None
        self.width = None
        self.height = None

        self.stop_event = threading.Event()
        self.fps_tracker = FrameRateTracker()

        self._init_input()

    def _init_input(self):
        src = self.input_src

        if src == "usb" or src.isdigit():
            self.input_type = InputType.USB_CAMERA
            self.cap = open_usb_camera(src, self.resolution)
            self.source_fps = get_source_fps(self.cap, "USB camera")
        elif src == "rpi":
            if not is_raspberry_pi():
                logger.error("RPi camera requested, but this is not a Raspberry Pi system.")
                sys.exit(1)
            self.input_type = InputType.RPI_CAMERA
            self.cap = open_rpi_camera()
            self.source_fps = 30
            if self.cap is None:
                sys.exit(1)
            logger.info("Using Raspberry Pi camera at 800x600, 30 FPS")
        elif is_stream_url(src):
            self.input_type = InputType.STREAM
            self.cap = open_cv_capture(src, "stream")
            self.source_fps = get_source_fps(self.cap, "stream camera")
        elif any(src.lower().endswith(suffix) for suffix in VIDEO_SUFFIXES):
            if not os.path.exists(src):
                logger.error(f"File not found: {src}")
                sys.exit(1)
            self.input_type = InputType.VIDEO
            self.cap = open_cv_capture(src, "video")
            self.source_fps = get_source_fps(self.cap, "video file")
        elif not os.path.exists(src):
            logger.error(f"Invalid input '{src}'.")
            sys.exit(1)
        else:
            self.input_type = InputType.IMAGES
            self.images = load_images_opencv(src)
            try:
                validate_images(self.images, self.batch_size)
            except ValueError as error:
                logger.error(error)
                sys.exit(1)
            logger.info(f"Using image input: {src}")

        if self.cap is not None:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.cap_processing_mode = select_cap_processing_mode(
                input_type=self.input_type.value,
                frame_rate=self.frame_rate,
                source_fps=self.source_fps,
                video_unpaced=self.video_unpaced,
            )

    @property
    def has_capture(self) -> bool:
        return self.cap is not None

    @property
    def has_images(self) -> bool:
        return self.images is not None and len(self.images) > 0

    def preprocess(
        self,
        input_queue: queue.Queue,
        model_input_width: int,
        model_input_height: int,
        preprocess_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
    ) -> None:
        preprocess_fn = preprocess_fn or default_preprocess
        if self.has_images:
            preprocess_images(
                images=self.images,
                batch_size=self.batch_size,
                input_queue=input_queue,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                preprocess_fn=preprocess_fn,
            )
        else:
            preprocess_from_capture(
                cap=self.cap,
                batch_size=self.batch_size,
                input_queue=input_queue,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                processing_mode=self.cap_processing_mode,
                preprocess_fn=preprocess_fn,
                target_fps=self.frame_rate,
                stop_event=self.stop_event,
            )
        input_queue.put(None)

    def visualize(
        self,
        visualization_settings: VisualizationSettings,
        output_queue: queue.Queue,
        callback: Callable[[Any, Any], None],
    ) -> None:
        image_index = 0
        video_writer = None
        writer_frame_width = None
        writer_frame_height = None
        cap = self.cap

        if cap is not None:
            cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
            if visualization_settings.output_resolution is not None:
                target_width, target_height = visualization_settings.output_resolution
            else:
                target_width, target_height = self.width, self.height
            writer_frame_width = target_width * (2 if visualization_settings.side_by_side else 1)
            writer_frame_height = target_height
            if visualization_settings.save_stream_output:
                camera_fps = self.source_fps
                output_fps = self.frame_rate or (camera_fps if camera_fps and camera_fps > 1 else 30.0)
                os.makedirs(visualization_settings.output_dir, exist_ok=True)
                output_video_path = os.path.join(visualization_settings.output_dir, "output.avi")
                video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"XVID"), output_fps, (writer_frame_width, writer_frame_height))

        self.fps_tracker.start()
        while True:
            result = output_queue.get()
            try:
                if result is None:
                    break
                original_frame, inference_result, *metadata = result
                if self.stop_event.is_set():
                    continue
                if isinstance(inference_result, list) and len(inference_result) == 1:
                    inference_result = inference_result[0]
                if metadata:
                    frame_with_detections = callback(original_frame, inference_result, metadata[0])
                else:
                    frame_with_detections = callback(original_frame, inference_result)
                
                self.fps_tracker.increment()
                render_status(frame_with_detections, self.fps_tracker.fps)
                render_resolution(frame_with_detections)

                output_bgr_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
                frame_to_show = resize_frame_for_output(output_bgr_frame, visualization_settings.output_resolution)

                if cap is not None:
                    cv2.imshow("Output", frame_to_show)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop_event.set()
                        continue
                    if visualization_settings.save_stream_output and video_writer is not None:
                        frame_to_write = cv2.resize(frame_to_show, (writer_frame_width, writer_frame_height))
                        video_writer.write(frame_to_write)
                else:
                    os.makedirs(visualization_settings.output_dir, exist_ok=True)
                    output_image_path = os.path.join(visualization_settings.output_dir, f"output_{image_index}.png")
                    cv2.imwrite(output_image_path, frame_to_show)
                image_index += 1
            finally:
                output_queue.task_done()

        if video_writer is not None:
            video_writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        self.stop_event.set()


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


def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    if not images:
        raise ValueError('No valid images found in the specified path.')
    if len(images) % batch_size != 0:
        raise ValueError('The number of input images should be divisible by the batch size without any remainder.')


def get_labels(labels_path: str) -> list:
    if labels_path is None or not os.path.exists(labels_path):
        return COCO_LABELS
    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()
    return class_names


def preprocess_from_capture(cap, batch_size, input_queue, model_input_width, model_input_height, processing_mode, preprocess_fn, target_fps=None, stop_event=None):
    def should_stop() -> bool:
        return stop_event is not None and stop_event.is_set()
    if processing_mode in (CapProcessingMode.CAMERA_FRAME_DROP, CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP):
        if not target_fps or target_fps <= 0:
            raise ValueError(f"{processing_mode.value} requires a positive target_fps")
    next_keep_timestamp = time.monotonic()
    keep_period = (1.0 / float(target_fps) if processing_mode == CapProcessingMode.CAMERA_FRAME_DROP else None)
    video_start_ms, wall_start_time = None, None
    next_keep_video_ms = None
    video_keep_period_ms = (1000.0 / float(target_fps) if processing_mode == CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP else None)
    raw_frames, processed_frames = [], []
    while not should_stop():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        current_pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if processing_mode in (CapProcessingMode.VIDEO_PACE, CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP):
            if video_start_ms is None:
                video_start_ms, wall_start_time = current_pos_ms, time.monotonic()
        if processing_mode == CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP:
            if next_keep_video_ms is None:
                next_keep_video_ms = current_pos_ms
            if current_pos_ms + 1e-3 < next_keep_video_ms:
                continue
            while current_pos_ms + 1e-3 >= next_keep_video_ms:
                next_keep_video_ms += video_keep_period_ms
        if processing_mode in (CapProcessingMode.VIDEO_PACE, CapProcessingMode.VIDEO_PACED_AND_FRAME_DROP):
            desired_wall_time = wall_start_time + (current_pos_ms - video_start_ms) / 1000.0
            current_wall_time = time.monotonic()
            if current_wall_time < desired_wall_time:
                time.sleep(desired_wall_time - current_wall_time)
        if processing_mode == CapProcessingMode.CAMERA_FRAME_DROP:
            current_time = time.monotonic()
            if current_time < next_keep_timestamp:
                continue
            next_keep_timestamp += keep_period
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
        processed_frames.append(preprocess_fn(frame_rgb, model_input_width, model_input_height))
        if len(raw_frames) >= batch_size:
            input_queue.put((raw_frames, processed_frames))
            raw_frames, processed_frames = [], []
    if raw_frames and not should_stop():
        input_queue.put((raw_frames, processed_frames))


def preprocess_images(images, batch_size, input_queue, model_input_width, model_input_height, preprocess_fn):
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        batch_tuple = ([image for image in batch], [preprocess_fn(image, model_input_width, model_input_height) for image in batch])
        input_queue.put(batch_tuple)


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)
    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset, y_offset = (model_w - new_img_w) // 2, (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image
    return padded_image


def resize_frame_for_output(frame: np.ndarray, resolution: Optional[Tuple[int, int]]) -> np.ndarray:
    if resolution is None:
        return frame
    _, target_h = resolution
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    scale = target_h / float(h)
    new_w, new_h = int(round(w * scale)), target_h
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class FrameRateTracker:
    def __init__(self):
        self._count = 0
        self._start_time = None
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
