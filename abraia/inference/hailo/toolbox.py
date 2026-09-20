import os
import logging
import collections
import numpy as np

from functools import partial
from typing import Dict, List, Optional, Tuple

from ...tasks import normalize_task, to_hailo_task
from ..postprocess.common import nms, sigmoid
from . import models, postprocess

logger = logging.getLogger(__name__)

MAX_ASYNC_INFER_JOBS = 20


def format_order_name(order):
    """Normalize HailoRT format-order values across SDK versions."""
    return str(getattr(order, "name", order)).rsplit(".", 1)[-1].upper()


def is_nms_format_order(order):
    """Return whether a HailoRT output uses a host-readable NMS format."""
    return format_order_name(order) in {
        "HAILO_NMS",
        "HAILO_NMS_BY_CLASS",
        "HAILO_NMS_BY_SCORE",
        "HAILO_NMS_WITH_BYTE_MASK",
    }

try:
    from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


if HAILO_AVAILABLE:
    class HailoInfer:
        def __init__(
            self, hef_path: str, batch_size: int = 1,
                input_type: Optional[str] = None, output_type: Optional[str] = None,
                priority: Optional[int] = 0) -> None:

            """
            Initialize the HailoAsyncInference class to perform asynchronous inference using a Hailo HEF model.

            Args:
                hef_path (str): Path to the HEF model file.
                batch_size (optional[int]): Number of inputs processed per inference. Defaults to 1.
                input_type (Optional[str], optional): Input data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
                output_type (Optional[str], optional): Output data type format. Common values: 'UINT8', 'UINT16', 'FLOAT32'.
                priority (optional[int]): Scheduler priority value for the model within the shared VDevice context. Defaults to 0.
            """
            params = VDevice.create_params()
            # Set the scheduling algorithm to round-robin to activate the scheduler
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            params.group_id = "SHARED"
            self.target = None
            self.hef = None
            self.infer_model = None
            self.config_ctx = None
            self.configured_model = None
            self.pending_jobs = collections.deque()
            self._closed = True
            self._config_entered = False
            try:
                self.target = VDevice(params)
                hef_path = os.fspath(hef_path)
                self.hef = HEF(hef_path)

                self.infer_model = self.target.create_infer_model(hef_path)
                self.infer_model.set_batch_size(batch_size)

                self._set_input_type(input_type)
                self._set_output_type(output_type)

                self.config_ctx = self.infer_model.configure()
                self.configured_model = self.config_ctx.__enter__()
                self._config_entered = True
                self.configured_model.set_scheduler_priority(priority)
            except Exception:
                self._release_resources()
                raise
            self.last_infer_job = None
            self._closed = False

        def _set_input_type(self, input_type: Optional[str] = None) -> None:
            """
            Set the input type for the HEF model. If the model has multiple inputs,
            it will set the same type of all of them.

            Args:
                input_type (Optional[str]): Format type of the input stream.
            """

            if input_type is not None:
                self.infer_model.input().set_format_type(getattr(FormatType, input_type))

        def _set_output_type(self, output_type: Optional[str] = None) -> None:
            """
            Set the output type for each model output.

            Args:
                output_type (Optional[str]): Desired output data type. Common values:
                    'UINT8', 'UINT16', 'FLOAT32'.
            """

            self.nms_postprocess_enabled = False

            output_orders = [
                format_order_name(output.format.order)
                for output in self.infer_model.outputs
            ]

            # NMS outputs are transformed by HailoRT when the completed
            # binding buffer is read. Keep their native output order instead
            # of forcing every layer through the raw tensor path.
            if any(
                order in {"HAILO_NMS_WITH_BYTE_MASK", "HAILO_NMS_BY_SCORE"}
                for order in output_orders
            ):
                # Use UINT8 and skip setting output formats
                self.nms_postprocess_enabled = True
                self.output_type = self._output_data_type2dict("UINT8")
                return

            if any(is_nms_format_order(order) for order in output_orders):
                self.nms_postprocess_enabled = True
                self.output_type = self._output_data_type2dict("FLOAT32")
                return

            # Otherwise, set the format type based on the provided output_type argument
            self.output_type = self._output_data_type2dict(output_type)

            # Apply format to each output layer
            for name, dtype in self.output_type.items():
                self.infer_model.output(name).set_format_type(getattr(FormatType, dtype))

        def get_vstream_info(self) -> Tuple[list, list]:
            """
            Get information about input and output stream layers.

            Returns:
                Tuple[list, list]: List of input stream layer information, List of 
                                   output stream layer information.
            """
            return (
                self.hef.get_input_vstream_infos(), 
                self.hef.get_output_vstream_infos()
            )

        def get_hef(self) -> HEF:
            """
            Get a HEF instance
            
            Returns:
                HEF: A HEF (Hailo Executable File) containing the model.
            """
            return self.hef

        def get_input_shape(self) -> Tuple[int, ...]:
            """
            Get the shape of the model's input layer.

            Returns:
                Tuple[int, ...]: Shape of the model's input layer.
            """
            return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

        def run(self, input_batch: List[np.ndarray], inference_callback_fn) -> object:
            """
            Run an asynchronous inference job on a batch of preprocessed inputs.

            This method reuses a preconfigured model (no reconfiguration overhead),
            prepares input/output bindings, launches async inference, and returns
            the job handle so that the caller can wait on it if needed.

            Args:
                input_batch (List[np.ndarray]): A batch of preprocessed model inputs.
                inference_callback_fn (Callable): Function to be invoked when inference is complete.
                                                  It receives `bindings_list` and additional context.

            Returns:
                Async job handle returned by `run_async`, which can be used to wait for completion or check status.
            """
            if self._closed:
                raise RuntimeError("Hailo inference has already been closed")

            while len(self.pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                self.pending_jobs.popleft().wait(10000)

            bindings_list = self._create_bindings(self.configured_model, input_batch)
            self.configured_model.wait_for_async_ready(timeout_ms=10000)

            # Launch async inference and attach the result handler
            self.last_infer_job = self.configured_model.run_async(
                bindings_list,
                partial(inference_callback_fn, bindings_list=bindings_list)
            )
            self.pending_jobs.append(self.last_infer_job)
            return self.last_infer_job

        def _create_bindings(self, configured_model, input_batch):
            """
            Create a list of input-output bindings for a batch of frames.

            Args:
                configured_model: The configured inference model.
                input_batch (List[np.ndarray]): List of input frames, preprocessed and ready.

            Returns:
                List[Bindings]: A list of bindings for each frame's input and output buffers.
            """

            def _frame_binding(frame: np.ndarray):
                output_buffers = {
                    name: np.empty(
                        self.infer_model.output(name).shape,
                        dtype=(getattr(np, self.output_type[name].lower()))
                    )
                    for name in self.output_type
                }

                binding = configured_model.create_bindings(output_buffers=output_buffers)
                binding.input().set_buffer(np.array(frame))
                return binding

            return [_frame_binding(frame) for frame in input_batch]

        def is_nms_postprocess_enabled(self) -> bool:
            """
            Returns True if the HEF model includes an NMS postprocess node.
            """
            return self.nms_postprocess_enabled

        def _output_data_type2dict(self, data_type: Optional[str]) -> Dict[str, str]:
            """
            Generate a dictionary mapping each output layer name to its corresponding
            data type. If no data type is provided, use the type defined in the HEF.

            Args:
                data_type (Optional[str]): The desired data type for all output layers.
                                           Valid values: 'float32', 'uint8', 'uint16'.
                                           If None, uses types from the HEF metadata.

            Returns:
                Dict[str, str]: A dictionary mapping output layer names to data types.
            """
            valid_types = {"float32", "uint8", "uint16"}
            data_type_dict = {}

            for output_info in self.hef.get_output_vstream_infos():
                name = output_info.name
                if data_type is None:
                    # Extract type from HEF metadata
                    hef_type = str(output_info.format.type).split(".")[-1]
                    data_type_dict[name] = hef_type
                else:
                    if data_type.lower() not in valid_types:
                        raise ValueError(f"Invalid data_type: {data_type}. Must be one of {valid_types}")
                    data_type_dict[name] = data_type

            return data_type_dict

        def close(self):
            """Wait for all submitted jobs and release the configured model."""
            if self._closed:
                return
            self._closed = True
            try:
                self.wait()
            finally:
                self._release_resources()

        def _release_resources(self):
            """Release partially or fully initialized Hailo resources."""
            if self.config_ctx is not None and self._config_entered:
                try:
                    self.config_ctx.__exit__(None, None, None)
                except Exception:
                    logger.warning("Failed to close Hailo configuration", exc_info=True)
                finally:
                    self.config_ctx = None
                    self.configured_model = None
                    self._config_entered = False

            release = getattr(self.target, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    logger.warning("Failed to release Hailo device", exc_info=True)
            self.target = None

        def wait(self):
            """Wait for all submitted jobs without releasing the backend."""
            while self.pending_jobs:
                self.pending_jobs.popleft().wait(10000)


if HAILO_AVAILABLE:
    class ModelInference(HailoInfer):
        def __init__(self, hef_path: str, task: str, labels: list, batch_size: int = 1, score_threshold: float = 0.25, mask_threshold: float = 0.45, model_type: str = 'v8'):
            task = normalize_task(task)
            backend_task = to_hailo_task(task)
            hef_path = models.resolve_hef_path(hef_path, backend_task)
            if hef_path is None:
                raise FileNotFoundError(f"Unable to resolve Hailo model for task '{task}'")
            # The v5/v8/pose postprocessors consume logits and DFL
            # distributions. Request dequantized output buffers from HailoRT;
            # using a HEF's native UINT8/UINT16 stream type makes those values
            # invalid for host-side softmax, box decoding, and NMS. Hailo's
            # byte-mask NMS path is still handled by HailoInfer itself.
            super().__init__(hef_path, batch_size, output_type="FLOAT32")
            self.task = backend_task
            self.labels = labels
            self.score_threshold = score_threshold
            self.mask_threshold = mask_threshold
            self.model_type = model_type
            self._postprocessors = {
                "detect": self._process_detect_results,
                "segment": self._process_segment_results,
                "pose": self._process_pose_results,
            }

        def _label(self, class_id):
            class_id = int(class_id)
            if self.labels and 0 <= class_id < len(self.labels):
                return self.labels[class_id]
            return str(class_id)

        def _get_results(self, bindings):
            if len(bindings._output_names) == 1:
                return bindings.output().get_buffer()
            return {name: np.expand_dims(bindings.output(name).get_buffer(), axis=0) for name in bindings._output_names}

        def _process_nms_results(self, result, image):
            if isinstance(result, np.ndarray):
                # Standard Hailo NMS buffers are either one [N, 5] array or
                # one [N, 5] array per class.  HailoRT may return either form
                # depending on the runtime version and output binding.
                infer_results = (
                    list(result)
                    if result.ndim >= 3
                    else [result]
                )
            else:
                infer_results = (
                    list(result)
                    if isinstance(result, (list, tuple))
                    else [result]
                )
            img_height, img_width = image.shape[:2]
            model_height, model_width, _ = self.get_input_shape()
            detections = []
            for class_id, det_group in enumerate(infer_results):
                if hasattr(det_group, "score"):
                    candidates = [det_group]
                else:
                    candidates = np.asarray(det_group)
                    if candidates.ndim == 1:
                        candidates = candidates.reshape(1, -1)
                    if candidates.ndim != 2:
                        continue

                for det in candidates:
                    if hasattr(det, "score"):
                        score = float(det.score)
                        class_value = int(det.class_id)
                        box = [det.x_min, det.y_min, det.x_max, det.y_max]
                        mask_value = getattr(det, "mask", None)
                    else:
                        if len(det) < 5:
                            continue
                        score = float(det[4])
                        class_value = class_id
                        box = [det[1], det[0], det[3], det[2]]
                        mask_value = None
                    if score < self.score_threshold:
                        continue
                    box_on_input_image, box_on_padded_image = postprocess.convert_nms_box_from_normalized(
                        box,
                        (model_height, model_width),
                        (img_height, img_width),
                    )
                    xmin, ymin, xmax, ymax = box_on_input_image
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    detection = {
                        'label': self._label(class_value),
                        'score': score,
                        'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'class_id': class_value,
                    }
                    if self.task == 'segment' and mask_value is not None:
                        mask = postprocess.resize_mask_to_unpadded_box(
                            mask_value, box_on_input_image, box_on_padded_image
                        )
                        if mask is not None:
                            detection['mask'] = mask
                    detections.append(detection)
            return detections

        def _process_detect_results(self, result, image):
            infer_results = result if isinstance(result, list) else [result]
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            detections = []
            for class_id, detection in enumerate(infer_results):
                for det in detection:
                    bbox, score = det[:4], det[4]
                    if score >= self.score_threshold:
                        # Hailo's raw detection output is normalized as
                        # [ymin, xmin, ymax, xmax, score].
                        xmin, ymin, xmax, ymax = postprocess.map_box_to_orig(
                            [bbox[1] * mw, bbox[0] * mh, bbox[3] * mw, bbox[2] * mh],
                            (oh, ow),
                            (mh, mw),
                        )
                        detections.append({'label': self._label(class_id), 'score': float(score), 'box': [xmin, ymin, xmax - xmin, ymax - ymin], 'class_id': class_id})
            return detections
        
        def _process_segment_results(self, raw_detections, image):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            arch_cfg = postprocess.SEGMENT_CONFIG[self.model_type]
            expected_shapes = [
                postprocess.resolve_shape(layer, self.model_type, arch_cfg)
                for layer in arch_cfg["layers"]
            ]
            endnodes = postprocess.select_output_layers(raw_detections, expected_shapes)
            # Use the runtime threshold during host-side NMS. The model-zoo
            # default (0.001) is useful for evaluation but allows almost every
            # low-confidence class through multi-label NMS, which can make the
            # callback appear stalled before results reach the renderer.
            postprocess_cfg = {**arch_cfg, "score_threshold": self.score_threshold}
            if self.model_type == "v5": result = postprocess.segment_yolov5_postprocess(endnodes, **postprocess_cfg)[0]
            elif self.model_type == "v8": result = postprocess.segment_yolov8_postprocess(endnodes, **postprocess_cfg)[0]
            else: raise ValueError(f"Unsupported architecture key: {self.model_type}")
            boxes, masks, scores, classes = result['detection_boxes'], result['mask'], result['detection_scores'], result['detection_classes']
            detections = []
            for i in range(len(boxes)):
                if scores[i] > self.score_threshold:
                    x1, y1, x2, y2 = boxes[i]
                    xmin, ymin, xmax, ymax = postprocess.map_box_to_orig(
                        [x1 * mw, y1 * mh, x2 * mw, y2 * mh],
                        (oh, ow),
                        (mh, mw),
                    )
                    mask = postprocess.map_mask_to_orig(masks[i], (oh, ow), (mh, mw))
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    detections.append({
                        'label': self._label(classes[i]), 'score': float(scores[i]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'mask': (mask[ymin:ymax, xmin:xmax] > self.mask_threshold).astype(np.uint8), 'class_id': int(classes[i])
                    })
            return detections

        def _process_pose_results(self, result, image, iou_thres=0.7, max_det=300):
            oh, ow = image.shape[:2]
            mh, mw, _ = self.get_input_shape()
            raw_detections = result
            reg_len = 15
            detection_out_channels = (reg_len + 1) * 4
            expected_shapes = [
                (1, h, w, c)
                for h, w, c in (
                    (20, 20, detection_out_channels),
                    (20, 20, 1),
                    (20, 20, 51),
                    (40, 40, detection_out_channels),
                    (40, 40, 1),
                    (40, 40, 51),
                    (80, 80, detection_out_channels),
                    (80, 80, 1),
                    (80, 80, 51),
                )
            ]
            endnodes = postprocess.select_output_layers(raw_detections, expected_shapes)
            batch_size = endnodes[0].shape[0]
            strides = [32, 16, 8]
            raw_boxes = endnodes[:7:3]
            scores = np.concatenate([np.reshape(s, (-1, s.shape[1] * s.shape[2], 1)) for s in endnodes[1:8:3]], axis=1)
            scores = postprocess.normalize_yolov8_scores(scores)
            kpts = [np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]]
            decoded_boxes, decoded_kpts = postprocess.decode_pose_results(
                raw_boxes, kpts, strides, (mh, mw), reg_len
            )
            predictions = np.concatenate([decoded_boxes, scores, np.reshape(decoded_kpts, (batch_size, -1, 51))], axis=2)
            detections = []
            x = predictions[0]
            x = x[
                (x[:, 4] > self.score_threshold)
                & np.isfinite(x).all(axis=1)
            ]
            if x.shape[0] > 0:
                boxes = np.copy(x[:, :4])
                boxes[:, 0], boxes[:, 1] = x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2
                boxes[:, 2], boxes[:, 3] = x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2
                finite_boxes = np.isfinite(boxes).all(axis=1)
                if not finite_boxes.all():
                    logger.debug("Discarding %d non-finite pose boxes", np.count_nonzero(~finite_boxes))
                    boxes = boxes[finite_boxes]
                    x = x[finite_boxes]
                indices = nms(np.concatenate((boxes, x[:, 4:5]), axis=1), iou_thres)[:max_det]
                for idx in indices:
                    xmin, ymin, xmax, ymax = postprocess.map_box_to_orig(
                        boxes[idx], (oh, ow), (mh, mw)
                    )
                    mapped_kpts = postprocess.map_keypoints_to_orig(
                        x[idx, 5:].reshape(17, 3)[..., :2], (oh, ow), (mh, mw)
                    )
                    detections.append({
                        'label': self._label(0) if self.labels else 'person', 'score': float(x[idx, 4]), 'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'keypoints': mapped_kpts, 'joint_scores': sigmoid(x[idx, 5:].reshape(17, 3)[..., 2]), 'class_id': 0
                    })
            return detections

        def _inference_callback(
            self,
            completion_info,
            bindings_list: list,
            input_batch: list,
            output_callback,
            image_batch=None,
            stop_event=None,
        ) -> None:
            image_batch = image_batch if image_batch is not None else input_batch

            def emit_result(index, result=None, error=None):
                if stop_event is not None and stop_event.is_set():
                    return False
                return output_callback(index, result, error) is not False

            if completion_info.exception:
                logger.error("Hailo inference failed: %s", completion_info.exception)
                for index in range(len(input_batch)):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if not emit_result(index, error=completion_info.exception):
                        break
                return

            for i, bindings in enumerate(bindings_list):
                try:
                    result = self._get_results(bindings)
                    image = image_batch[i]
                    if self.is_nms_postprocess_enabled():
                        processed_result = self._process_nms_results(result, image)
                    else:
                        processor = self._postprocessors.get(self.task)
                        processed_result = (
                            processor(result, image)
                            if processor is not None
                            else result
                        )
                except Exception as exc:
                    logger.exception("Failed to post-process Hailo %s result", self.task)
                    if not emit_result(i, error=exc):
                        break
                    continue
                if not emit_result(i, processed_result):
                    break

        def infer_batch(
            self, input_batch, output_callback, stop_event=None, image_batch=None
        ):
            """Submit one batch and emit decoded results by frame index."""
            if stop_event is not None and stop_event.is_set():
                return
            if image_batch is not None and len(image_batch) != len(input_batch):
                raise ValueError("image_batch must match input_batch length")
            inference_callback_fn = partial(
                self._inference_callback,
                input_batch=input_batch,
                image_batch=image_batch,
                output_callback=output_callback,
                stop_event=stop_event,
            )
            self.run(input_batch, inference_callback_fn)

else:
    HailoInfer = None
    ModelInference = None
