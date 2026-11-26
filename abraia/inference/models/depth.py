"""ONNX inference for Depth Anything and YOLO26n-Depth."""

import cv2
import numpy as np

from ...utils.remote import resolve_model_file
from ..geometry import normalize
from ..session import OnnxSessionMixin


_DA3_PROCESS_RES = 504
_DAV2_PROCESS_RES = 518
_DA3_PATCH_SIZE = 14
_YOLO_PROCESS_RES = 768


def _resize_da3(image, input_size):
    height, width = image.shape[:2]
    scale = _DA3_PROCESS_RES / max(height, width)
    scaled = (max(1, round(width * scale)), max(1, round(height * scale)))
    if scaled != (width, height):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, scaled, interpolation=interpolation)
    size = tuple(
        dim if isinstance(dim, int) else (
            max(1, (value + _DA3_PATCH_SIZE // 2) // _DA3_PATCH_SIZE) * _DA3_PATCH_SIZE
        )
        for dim, value in zip(input_size[::-1], scaled)
    )
    if size != scaled:
        interpolation = (
            cv2.INTER_CUBIC if any(a > b for a, b in zip(size, scaled))
            else cv2.INTER_AREA
        )
        image = cv2.resize(image, size, interpolation=interpolation)
    return image


def _resize_dav2(image, input_size):
    """Resize Depth Anything V2 input with a 518-pixel long edge."""
    if all(isinstance(dim, int) for dim in input_size):
        return cv2.resize(image, input_size[::-1], interpolation=cv2.INTER_CUBIC)
    height, width = image.shape[:2]
    scale = _DAV2_PROCESS_RES / max(height, width)
    scaled = (max(1, round(width * scale)), max(1, round(height * scale)))
    if scaled != (width, height):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, scaled, interpolation=interpolation)
    size = tuple(
        dim if isinstance(dim, int) else (value // _DA3_PATCH_SIZE) * _DA3_PATCH_SIZE
        for dim, value in zip(input_size[::-1], scaled)
    )
    size = tuple(max(_DA3_PATCH_SIZE, dim) for dim in size)
    if size != scaled:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    return image


def _resize_yolo_depth(depth, shape):
    """Resize using the same corner alignment as the YOLO depth predictor."""
    if depth.shape == shape:
        return depth
    y, x = (
        np.linspace(0, length - 1, size, dtype=np.float32)
        for length, size in zip(depth.shape, shape)
    )
    x0, y0 = x.astype(np.intp), y.astype(np.intp)
    x1 = np.minimum(x0 + 1, depth.shape[1] - 1)
    y1 = np.minimum(y0 + 1, depth.shape[0] - 1)
    wx, wy = x - x0.astype(np.float32), (y - y0.astype(np.float32))[:, None]
    rows = depth[:, x0] * (1 - wx) + depth[:, x1] * wx
    return rows[y0] * (1 - wy) + rows[y1] * wy


def restore_yolo_depth(depth, input_shape, image_shape):
    """Restore the model input resolution, then resize to the original image."""
    return _resize_yolo_depth(_resize_yolo_depth(depth, input_shape), image_shape)


class DepthModel(OnnxSessionMixin):
    """Predict relative or metric depth from a single uint8 RGB image."""

    def __init__(self, model_uri, *, kind, providers=None):
        self.is_da3 = kind == "depth_anything"
        self.is_dav2 = kind == "depth_anything_v2"
        self.is_relative = self.is_da3 or self.is_dav2
        self.depth_units = "relative" if self.is_relative else "m"
        session_options = None
        if self.is_da3:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = False
        self._init_onnx_session(
            resolve_model_file(model_uri),
            providers=providers,
            session_options=session_options,
        )
        try:
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            layout = (1, 1, 3) if self.is_da3 else (1, 3)
            rank = len(layout) + 2
            if len(inputs) != 1 or len(inputs[0].shape or ()) != rank:
                raise ValueError(f"{kind} requires a single rank-{rank} ONNX input")
            model_input = inputs[0]
            shape = model_input.shape
            if model_input.type != "tensor(float)" or any(
                isinstance(actual, int) and actual != expected
                for actual, expected in zip(shape[:-2], layout)
            ) or any(isinstance(dim, int) and dim <= 0 for dim in shape[-2:]):
                raise ValueError(f"{kind} requires a batch-one FP32 RGB ONNX input")
            self.input_name = model_input.name
            self.input_size = tuple(shape[-2:])
            if self.is_da3:
                if not any(output.name == "predicted_depth" for output in outputs):
                    raise ValueError("DA3 ONNX must expose a 'predicted_depth' output")
                self.output_name = "predicted_depth"
            elif self.is_dav2:
                if not any(output.name == "predicted_depth" for output in outputs):
                    raise ValueError("DA2 ONNX must expose a 'predicted_depth' output")
                self.output_name = "predicted_depth"
            else:
                if len(outputs) != 1 or len(outputs[0].shape or ()) not in (3, 4):
                    raise ValueError("YOLO26-Depth ONNX must expose one dense depth output")
                task = self.session.get_modelmeta().custom_metadata_map.get("task")
                if task and task != "depth":
                    raise ValueError("Select a YOLO26-Depth ONNX export")
                self.output_name = outputs[0].name
                self.input_size = tuple(
                    dim if isinstance(dim, int) else _YOLO_PROCESS_RES
                    for dim in self.input_size
                )
        except Exception:
            self.close()
            raise

    def predict(self, image):
        """Return an image-sized float32 depth map and its units."""
        self._ensure_open()
        image = np.asarray(image)
        if (
            image.ndim != 3 or image.shape[2] != 3
            or not image.size or image.dtype != np.uint8
        ):
            raise ValueError("Depth inference requires a nonempty uint8 RGB image")
        shape = image.shape[:2]
        if self.is_da3:
            resized = _resize_da3(image, self.input_size)
            tensor = np.ascontiguousarray(normalize(resized).transpose(2, 0, 1)[None, None])
        elif self.is_dav2:
            resized = _resize_dav2(image, self.input_size)
            tensor = np.ascontiguousarray(normalize(resized).transpose(2, 0, 1)[None])
        else:
            resized = cv2.resize(image, self.input_size[::-1], interpolation=cv2.INTER_LINEAR)
            tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
            tensor /= 255.0
        depth, = self.session.run([self.output_name], {self.input_name: tensor})
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim not in (3, 4) or any(dim != 1 for dim in depth.shape[:-2]):
            raise ValueError(f"Unexpected ONNX depth output shape: {depth.shape}")
        depth = depth.reshape(depth.shape[-2:])
        if self.is_da3:
            if depth.shape != shape:
                depth = cv2.resize(depth, shape[::-1], interpolation=cv2.INTER_LINEAR)
        elif self.is_dav2:
            if depth.shape != shape:
                depth = cv2.resize(depth, shape[::-1], interpolation=cv2.INTER_LINEAR)
            finite = np.isfinite(depth)
            if finite.any():
                minimum = float(depth[finite].min())
                maximum = float(depth[finite].max())
                depth[finite] = minimum + maximum - depth[finite]
        else:
            depth = restore_yolo_depth(depth, self.input_size, shape)
        return {"depth_map": depth, "depth_units": self.depth_units}

    def run(self, image):
        """Publish a dense depth result through the pipeline record interface."""
        return [self.predict(image)]
