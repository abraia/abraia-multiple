"""Inference for YOLOv9 Hailo exports produced by Multiple."""

import numpy as np
from PIL import Image


def preprocess_yolov9_image(image, input_height, input_width):
    """Letterbox an RGB image exactly as the native YOLOv9 backend does."""
    if isinstance(image, Image.Image):
        source = image.convert("RGB")
    else:
        source = Image.fromarray(np.asarray(image)).convert("RGB")

    width, height = source.size
    scale = min(input_width / width, input_height / height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized = source.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    padded = Image.new(
        "RGB", (input_width, input_height), (114, 114, 114)
    )
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    padded.paste(resized, (pad_x, pad_y))
    return np.asarray(padded, dtype=np.float32) / 255.0


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -88.0, 88.0)))


def _softmax(values, axis):
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _output_hwc(value, channels, height, width, name):
    """Return one Hailo output as HWC, validating its compiled dimensions."""
    output = np.asarray(value, dtype=np.float32)
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    if output.shape == (height, width, channels):
        return output
    if output.shape == (channels, height, width):
        return output.transpose(1, 2, 0)
    raise ValueError(
        f"YOLOv9 Hailo output {name!r} has shape {output.shape}; "
        f"expected {(height, width, channels)}."
    )


def decode_yolov9_outputs(
    outputs,
    metadata,
    image_shape,
    *,
    conf_threshold=None,
    iou_threshold=None,
    labels=None,
):
    """Decode YOLOv9 class and DFL outputs into source-image detections."""
    from abraia.inference.geometry import nms
    from abraia.inference.hailo.postprocess import LetterboxTransform

    classes = list(metadata["classes"])
    input_height, input_width = map(int, metadata["inputShape"][-2:])
    reg_max = int(metadata["regMax"])
    roles = metadata["outputRoles"]
    conf_threshold = (
        float(metadata.get("confidenceThreshold", 0.25))
        if conf_threshold is None else float(conf_threshold)
    )
    iou_threshold = (
        float(metadata.get("iouThreshold", 0.7))
        if iou_threshold is None else float(iou_threshold)
    )

    candidates_by_class = [[] for _ in classes]
    bins = np.arange(reg_max, dtype=np.float32)
    for stride in metadata["strides"]:
        stride = int(stride)
        height, width = input_height // stride, input_width // stride
        cls_name = roles[f"classes_{stride}"]
        box_name = roles[f"box_distributions_{stride}"]
        class_count = len(classes)
        cls_logits = _output_hwc(
            outputs[cls_name], class_count, height, width, cls_name
        )
        box_channels = 4 * reg_max
        box_logits = _output_hwc(
            outputs[box_name], box_channels, height, width, box_name
        ).reshape(height, width, 4, reg_max)

        distances = (
            _softmax(box_logits, axis=3) * bins[None, None, None, :]
        ).sum(axis=3) * stride
        grid_x, grid_y = np.meshgrid(
            (np.arange(width, dtype=np.float32) * stride + stride // 2),
            (np.arange(height, dtype=np.float32) * stride + stride // 2),
        )
        left, top, right, bottom = distances.transpose(2, 0, 1)
        boxes = np.stack((
            grid_x - left,
            grid_y - top,
            grid_x + right,
            grid_y + bottom,
        ), axis=-1).reshape(-1, 4)
        scores = _sigmoid(cls_logits).reshape(-1, class_count)

        for class_id in range(class_count):
            selected = np.flatnonzero(scores[:, class_id] >= conf_threshold)
            if not len(selected):
                continue
            selected_scores = scores[selected, class_id]
            if len(selected) > 300:
                top_indices = np.argpartition(selected_scores, -300)[-300:]
                selected = selected[top_indices]
                selected_scores = selected_scores[top_indices]
            rows = np.column_stack((
                boxes[selected], selected_scores,
            )).astype(np.float32)
            keep = nms(rows, iou_threshold)[:300]
            candidates_by_class[class_id].extend(
                (rows[index, :4], float(rows[index, 4]))
                for index in keep
            )

    image_height, image_width = image_shape[:2]
    transform = LetterboxTransform.from_dimensions(
        (image_height, image_width), (input_height, input_width)
    )
    results = []
    for class_id, candidates in enumerate(candidates_by_class):
        label = classes[class_id]
        if labels and label not in labels:
            continue
        for box, score in candidates:
            left, top, right, bottom = transform.box_to_original(box.tolist())
            if right <= left or bottom <= top:
                continue
            results.append({
                "label": label,
                "score": score,
                "class_id": class_id,
                "box": [left, top, right - left, bottom - top],
            })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:300]


class HailoYOLOv9Model:
    """Run a Multiple YOLOv9 HEF using the shared Abraia Hailo runtime."""

    def __init__(self, model_uri, *, score_threshold=None, iou_threshold=None):
        from abraia.inference.hailo.models import (
            load_hailo_model_config,
            resolve_hef_path,
        )
        from abraia.inference.hailo.toolbox import HAILO_AVAILABLE, HailoInfer

        self.inference = None
        self._closed = False
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        try:
            if not HAILO_AVAILABLE or HailoInfer is None:
                raise RuntimeError(
                    "Hailo support is unavailable; install hailo_platform on a "
                    "Hailo-enabled host"
                )
            self.metadata = load_hailo_model_config(model_uri)
            if (
                self.metadata.get("format") != "hailo"
                or self.metadata.get("backend") != "yolov9"
                or self.metadata.get("task") != "detection"
            ):
                raise ValueError("Select a compiled Multiple YOLOv9 Hailo model.")
            self.classes = list(self.metadata["classes"])
            self.input_shape = self.metadata["inputShape"]
            self.accelerator = "HAILO"

            hef_path = resolve_hef_path(model_uri, "detection")
            if hef_path is None:
                raise FileNotFoundError(
                    f"Unable to resolve YOLOv9 Hailo model: {model_uri}"
                )
            self.inference = HailoInfer(
                str(hef_path), input_type="FLOAT32", output_type="FLOAT32"
            )
            input_infos, output_infos = self.inference.get_vstream_info()
            if len(input_infos) != 1:
                raise ValueError(
                    "YOLOv9 Hailo models require one image input; "
                    f"found {len(input_infos)}."
                )
            input_height, input_width, channels = input_infos[0].shape
            if (
                channels != 3
                or (input_height, input_width)
                != tuple(int(value) for value in self.input_shape[-2:])
            ):
                raise ValueError(
                    "The Hailo input shape does not match YOLOv9 metadata."
                )
            self.input_height = int(input_height)
            self.input_width = int(input_width)
            self.output_names = tuple(info.name for info in output_infos)
            self.output_name_map = {}
            for expected in self.metadata["outputRoles"].values():
                matches = [
                    name for name in self.output_names
                    if name == expected
                    or str(name).rsplit("/", 1)[-1].split(":", 1)[0] == expected
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Could not resolve YOLOv9 Hailo output {expected!r}."
                    )
                self.output_name_map[expected] = matches[0]
        except Exception:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Inference model has already been closed")

    def _infer(self, tensor):
        outputs = []

        def receive(completion_info, bindings_list, **_kwargs):
            if completion_info.exception:
                outputs.append(RuntimeError(str(completion_info.exception)))
                return
            try:
                binding = bindings_list[0]
                buffers = {
                    name: np.array(
                        binding.output(name).get_buffer(),
                        dtype=np.float32,
                        copy=True,
                    )
                    for name in self.output_names
                }
                outputs.append({
                    expected: buffers[actual]
                    for expected, actual in self.output_name_map.items()
                })
            except Exception as error:
                outputs.append(error)

        self.inference.run([tensor], receive)
        self.inference.wait()
        if not outputs:
            raise RuntimeError("Hailo inference completed without outputs.")
        if isinstance(outputs[0], Exception):
            raise outputs[0]
        return outputs[0]

    def run(
        self, image, *, conf_threshold=None, iou_threshold=None, labels=None,
        **_options,
    ):
        """Return decoded detections for one RGB source image."""
        self._ensure_open()
        tensor = preprocess_yolov9_image(
            image, self.input_height, self.input_width
        )
        outputs = self._infer(np.ascontiguousarray(tensor))
        return decode_yolov9_outputs(
            outputs,
            self.metadata,
            np.asarray(image).shape,
            conf_threshold=(
                self.score_threshold
                if conf_threshold is None else conf_threshold
            ),
            iou_threshold=(
                self.iou_threshold
                if iou_threshold is None else iou_threshold
            ),
            labels=labels,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        inference, self.inference = self.inference, None
        if inference is not None:
            inference.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "HailoYOLOv9Model",
    "decode_yolov9_outputs",
    "preprocess_yolov9_image",
]
