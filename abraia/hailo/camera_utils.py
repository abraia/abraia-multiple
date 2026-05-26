import os
import sys
import cv2
import threading
from enum import Enum
from typing import Any, Optional

from .defines import CAMERA_RESOLUTION_MAP
from .hailo_logger import get_logger

hailo_logger = get_logger(__name__)


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

    Camera inputs:
        - CAMERA_NORMAL
        - CAMERA_FRAME_DROP

    Video inputs:
        - VIDEO_PACE
        - VIDEO_UNPACED
        - VIDEO_PACED_AND_FRAME_DROP
    """
    is_camera = input_type in ("usb_camera", "rpi_camera", "stream")
    is_video = input_type == "video"
    has_target_fps = frame_rate is not None and frame_rate > 0
    has_source_fps = source_fps is not None and source_fps > 0

    if not (is_camera or is_video):
        return None

    if is_video and video_unpaced:
        if has_target_fps:
            hailo_logger.warning(
                "--frame-rate is ignored when --video-unpaced is enabled."
            )
        return CapProcessingMode.VIDEO_UNPACED

    if has_target_fps and has_source_fps and frame_rate >= source_fps:
        hailo_logger.warning(
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

    Args:
        cap: Opened capture object.
        source_name: Human-readable source name for logging.

    Returns:
        Optional[float]: Reported FPS, or None if unavailable.
    """
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        hailo_logger.debug(f"{source_name} FPS not reported by source.")
        return None
    return source_fps


def open_cv_capture(src: Any, source_type: str) -> Any:
    """
    Open an OpenCV-based capture source.

    Args:
        src: Video file path, stream URL, device path, or camera index.
        source_type: Human-readable source type for logging.

    Returns:
        Opened capture object.
    """
    if source_type == "video" and not os.path.exists(src):
        hailo_logger.error(f"File not found: {src}")
        sys.exit(1)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        hailo_logger.error(f"Failed to open {source_type} source: {src}")
        sys.exit(1)

    hailo_logger.info(f"Using {source_type} input: {src}")
    return cap


def _apply_resolution_and_validate(
    cap: Any,
    resolution: Optional[str],
) -> Any:
    """
    Apply requested resolution and validate that the capture source
    produces frames.

    Args:
        cap: Opened capture object.
        resolution: Optional named resolution key.

    Returns:
        The validated capture object.
    """
    if resolution in CAMERA_RESOLUTION_MAP:
        width, height = CAMERA_RESOLUTION_MAP[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        hailo_logger.debug(f"Camera resolution forced to {width}x{height}")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        hailo_logger.error("Camera opened but produced no frames.")
        sys.exit(1)

    return cap


def open_usb_camera(input_src: str, resolution: Optional[str]):
    camera_index = int(str(input_src))
    cap = open_cv_capture(camera_index, "USB camera index")
    return _apply_resolution_and_validate(cap, resolution)


def open_rpi_camera() -> Optional[Any]:
    """
    Open Raspberry Pi camera using Picamera2.

    Returns:
        PiCamera2CaptureAdapter | None:
            Camera adapter if successful, otherwise None.
    """
    try:
        from picamera2 import Picamera2
    except Exception as e:
        hailo_logger.error(f"Picamera2 not available: {e}")
        return None

    try:
        picam2 = Picamera2()
        width, height = 800, 600
        fps = 30
        main = {"size": (width, height), "format": "RGB888"}
        config = picam2.create_video_configuration(main=main, controls={"FrameRate": fps})

        picam2.configure(config)
        picam2.start()

        hailo_logger.debug(f"RPi camera started ({width}x{height}) @ {fps} FPS")

        return PiCamera2CaptureAdapter(picam2)

    except Exception as e:
        hailo_logger.error(f"Failed to open RPi camera: {e}")
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
