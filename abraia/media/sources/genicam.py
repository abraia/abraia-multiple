"""Optional GenICam image acquisition through GenTL Producers."""

import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, Iterator, Mapping


class GenICamSource:
    """Acquire RGB frames from a GenICam device using Harvester.

    Harvester and a compatible vendor GenTL Producer are optional runtime
    requirements. ``src`` selects a device by serial number, or by numeric
    index when it contains only digits. Source options support ``producer``
    (a CTI file or directory), ``selector`` (Harvester device properties),
    ``features`` (writable GenICam feature values), and ``timeout`` seconds.
    """

    _OPTION_NAMES = {"producer", "selector", "features", "timeout"}

    def __init__(self, src, fps=30, options=None):
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("GenICam source options must be an object")
        options = {} if options is None else dict(options)
        unknown_options = sorted(set(options) - self._OPTION_NAMES)
        if unknown_options:
            raise ValueError(
                "Unsupported GenICam source option(s): "
                + ", ".join(unknown_options)
            )

        selector = options.get("selector")
        if selector is not None and not isinstance(selector, Mapping):
            raise ValueError("GenICam 'selector' must be an object")
        if selector is not None and not selector:
            raise ValueError("GenICam 'selector' cannot be empty")
        features = options.get("features", {})
        if not isinstance(features, Mapping):
            raise ValueError("GenICam 'features' must be an object")
        try:
            self.timeout = float(options.get("timeout", 0.2))
        except (TypeError, ValueError):
            raise ValueError("GenICam 'timeout' must be a positive number")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("GenICam 'timeout' must be a positive number")

        self.src = src
        self.frame_rate = float(fps or 0)
        self.width = 0
        self.height = 0
        self.device_info = {}
        self._stop_event = threading.Event()
        self._harvester = None
        self._acquirer = None
        self._acquiring = False
        self._closed = False

        try:
            from harvesters.core import Harvester
        except ImportError as error:
            raise ImportError(
                "GenICam input requires the optional 'harvesters' package and "
                "a compatible vendor GenTL Producer. Install Abraia with "
                "'pip install abraia[genicam]' and configure source.options.producer."
            ) from error

        try:
            self._harvester = Harvester()
            producer_files = self._producer_files(options.get("producer"))
            if not producer_files:
                raise RuntimeError(
                    "No GenTL Producer was configured. Set source.options.producer "
                    "to a CTI file or directory, or set GENICAM_GENTL64_PATH."
                )
            for producer_file in producer_files:
                self._harvester.add_file(str(producer_file))
            self._harvester.update()

            selector = self._device_selector(src, selector)
            try:
                self._acquirer = self._harvester.create(selector)
            except Exception as error:
                available = self._format_devices(self._harvester.device_info_list)
                raise RuntimeError(
                    f"Unable to open GenICam device {selector!r}. Available devices: "
                    f"{available or 'none'}"
                ) from error

            self.device_info = self._selected_device_info(selector)
            self._apply_features(features)
            node_map = self._acquirer.remote_device.node_map
            self.width = self._node_value(node_map, "Width", 0)
            self.height = self._node_value(node_map, "Height", 0)
            self.frame_rate = float(
                self._node_value(node_map, "AcquisitionFrameRate", self.frame_rate)
                or self.frame_rate
            )
            self._acquirer.start()
            self._acquiring = True
        except Exception:
            self.close()
            raise

    @staticmethod
    def _producer_files(configured) -> list:
        """Resolve configured or environment-provided CTI producer files."""
        paths = []
        if configured is not None:
            configured_paths = configured if isinstance(configured, (list, tuple)) else [configured]
        else:
            env_paths = os.environ.get("GENICAM_GENTL64_PATH", "")
            configured_paths = [path for path in env_paths.split(os.pathsep) if path]

        for value in configured_paths:
            path = Path(str(value)).expanduser()
            if path.is_dir():
                paths.extend(sorted(path.glob("*.cti")))
            elif path.is_file() and path.suffix.lower() == ".cti":
                paths.append(path)
            elif configured is not None:
                raise FileNotFoundError(f"GenTL Producer not found: {path}")
        if configured is None and not paths:
            return []
        if not paths:
            raise FileNotFoundError("No .cti GenTL Producer files found")
        return paths

    @staticmethod
    def _device_selector(src, configured):
        if configured is not None:
            return dict(configured)
        if isinstance(src, int) or str(src).strip().isdigit():
            return int(src)
        return {"serial_number": str(src).strip()}

    @staticmethod
    def _format_devices(devices: Iterable) -> str:
        return ", ".join(
            str(getattr(device, "display_name", None)
                or getattr(device, "serial_number", None)
                or getattr(device, "id_", "unknown"))
            for device in devices
        )

    def _selected_device_info(self, selector: Any) -> Dict[str, Any]:
        devices = getattr(self._harvester, "device_info_list", ())
        if isinstance(selector, int):
            selected = devices[selector]
            return self._device_info_dict(selected)
        for device in devices:
            if all(getattr(device, key, None) == value for key, value in selector.items()):
                return self._device_info_dict(device)
        return {}

    @staticmethod
    def _device_info_dict(device) -> Dict[str, Any]:
        fields = (
            "display_name", "id_", "model", "serial_number", "tl_type",
            "user_defined_name", "vendor", "version",
        )
        return {
            field: getattr(device, field)
            for field in fields
            if getattr(device, field, None) is not None
        }

    @staticmethod
    def _node_value(node_map, name, default=None):
        try:
            return getattr(node_map, name).value
        except Exception:
            return default

    def _apply_features(self, features: Mapping) -> None:
        node_map = self._acquirer.remote_device.node_map
        for name, value in features.items():
            try:
                node = getattr(node_map, str(name))
                node.value = value
            except Exception as error:
                raise ValueError(
                    f"Unable to set GenICam feature {name!r} to {value!r}"
                ) from error

    @staticmethod
    def _frame_from_component(component):
        """Copy common unpacked Mono/Bayer/RGB/BGR PFNC images into RGB."""
        import numpy as np
        import cv2

        pixel_format = str(component.data_format)
        width, height = int(component.width), int(component.height)
        raw = np.asarray(component.data)
        if raw.ndim != 1:
            raw = raw.reshape(-1)

        if "Packed" in pixel_format or re.search(r"\d+p", pixel_format):
            raise ValueError(
                f"Packed GenICam pixel format {pixel_format!r} is not supported; "
                "select an unpacked Mono, Bayer, RGB, or BGR format."
            )
        bayer = re.match(r"^Bayer(RG|GR|GB|BG)", pixel_format)
        if pixel_format.startswith("Mono"):
            channels = 1
        elif bayer:
            channels = 1
        elif pixel_format.startswith(("RGBA", "BGRA")):
            channels = 4
        elif pixel_format.startswith(("RGB", "BGR")):
            channels = 3
        else:
            raise ValueError(
                f"Unsupported GenICam pixel format {pixel_format!r}; configure "
                "an unpacked Mono, Bayer, RGB, or BGR pixel format in "
                "source.options.features."
            )

        row_values = width * channels
        if height <= 0 or width <= 0 or raw.size % height:
            raise ValueError(
                f"Invalid GenICam image buffer for {width}x{height} {pixel_format}"
            )
        row_stride = raw.size // height
        if row_stride < row_values:
            raise ValueError(
                f"GenICam image buffer is too small for {width}x{height} {pixel_format}"
            )
        raw = raw.reshape(height, row_stride)[:, :row_values]
        if channels == 1:
            frame = raw.reshape(height, width)
        else:
            frame = raw.reshape(height, width, channels)

        if frame.dtype != np.uint8:
            match = re.search(r"(\d+)", pixel_format)
            bit_depth = int(match.group(1)) if match else frame.dtype.itemsize * 8
            maximum = float((1 << min(bit_depth, 16)) - 1)
            frame = np.clip(frame.astype(np.float32) * (255.0 / maximum), 0, 255)
            frame = frame.astype(np.uint8)

        if channels == 1:
            if bayer:
                pattern = {
                    "RG": "RGGB",
                    "GR": "GRBG",
                    "GB": "GBRG",
                    "BG": "BGGR",
                }[bayer.group(1)]
                frame = cv2.cvtColor(
                    frame,
                    getattr(cv2, f"COLOR_Bayer{pattern}2RGB"),
                )
            else:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif channels == 4:
            frame = frame[:, :, :3]
        if pixel_format.startswith(("BGR", "BGRA")):
            frame = frame[:, :, ::-1]
        return np.array(frame, copy=True, order="C"), pixel_format

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._closed or self._acquirer is None:
            return
        while not self._stop_event.is_set():
            try:
                buffer = self._acquirer.try_fetch(timeout=self.timeout)
            except Exception as error:
                if self._stop_event.is_set():
                    break
                if "timeout" in type(error).__name__.lower():
                    continue
                raise RuntimeError("GenICam image acquisition failed") from error
            if buffer is None:
                continue

            try:
                with buffer:
                    if getattr(buffer, "is_incomplete", False):
                        raise RuntimeError("GenICam returned an incomplete image buffer")
                    components = buffer.payload.components
                    if not components:
                        raise RuntimeError("GenICam buffer contains no image component")
                    frame, pixel_format = self._frame_from_component(components[0])
                    data = {
                        "source_type": "genicam",
                        "device": dict(self.device_info),
                        "pixel_format": pixel_format,
                    }
                    for name in ("timestamp_ns", "timestamp", "frame_id"):
                        value = getattr(buffer, name, None)
                        if value is not None:
                            data[name] = value
            except Exception as error:
                if self._stop_event.is_set():
                    break
                raise RuntimeError("Unable to convert GenICam image buffer") from error
            yield {"frame": frame, "data": data}

    def stop(self) -> None:
        """Stop acquisition so a concurrent frame fetch can finish."""
        self._stop_event.set()
        if self._acquiring and self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass
            self._acquiring = False

    def close(self) -> None:
        """Release the device and producer, safely and repeatedly."""
        if self._closed:
            return
        self._closed = True
        self.stop()
        if self._acquirer is not None:
            try:
                self._acquirer.destroy()
            except Exception:
                pass
            self._acquirer = None
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
            self._harvester = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["GenICamSource"]
