"""ONNX inference adapters for RF-DETR and D-FINE object models."""

import numpy as np

from abraia.inference.session import OnnxSessionMixin, load_onnx_model_config


_COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)


def _class_name(classes, class_id):
    return (
        classes[class_id]
        if 0 <= class_id < len(classes)
        else str(class_id)
    )


def _is_selected_label(label, labels):
    return not labels or label in labels


def _crop_mask_to_box(mask, box):
    """Return a full-image mask cropped to the integer box used by renderers."""
    x, y, width, height = (int(value) for value in box)
    if width <= 0 or height <= 0:
        return np.zeros((max(height, 0), max(width, 0)), dtype=np.uint8)
    return np.asarray(mask[y:y + height, x:x + width], dtype=np.uint8)


def _to_rgb_uint8(image):
    """Normalize grayscale or RGB input to an RGB uint8 image."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image[:, :, :3]


def _rfdetr_class_slots(output_count, class_count):
    """Map RF-DETR output slots to sidecar class indices."""
    if output_count in (class_count, class_count + 1):
        slots = np.arange(class_count)
        return slots, slots
    if output_count == 91 and class_count == 80:
        return np.asarray(_COCO_CATEGORY_IDS), np.arange(class_count)
    raise ValueError(
        "RF-DETR ONNX class output does not match its metadata: "
        f"{output_count} outputs for {class_count} classes."
    )


def _decode_rfdetr_outputs(
    boxes,
    logits,
    masks,
    image_size,
    classes,
    conf_threshold,
    labels,
):
    """Convert RF-DETR's raw ONNX outputs into Abraia detection records."""
    boxes = np.asarray(boxes, dtype=np.float32)[0]
    logits = np.asarray(logits, dtype=np.float32)[0]
    output_slots, class_indices = _rfdetr_class_slots(
        logits.shape[-1], len(classes)
    )
    if not len(output_slots):
        return []
    probabilities = 1 / (
        1 + np.exp(-np.clip(logits[:, output_slots], -88, 88))
    )
    num_classes = len(output_slots)
    flat_scores = probabilities.reshape(-1)
    top_indices = np.argsort(-flat_scores, kind="stable")[
        :min(300, flat_scores.size)
    ]
    scores = flat_scores[top_indices]
    query_indices = top_indices // num_classes
    class_ids = class_indices[top_indices % num_classes]
    threshold = 0.0 if conf_threshold is None else float(conf_threshold)
    keep = scores >= threshold
    scores = scores[keep]
    query_indices = query_indices[keep]
    class_ids = class_ids[keep]

    height, width = image_size
    selected_boxes = boxes[query_indices]
    cx, cy, box_width, box_height = selected_boxes.T
    xyxy = np.stack((
        cx - box_width / 2,
        cy - box_height / 2,
        cx + box_width / 2,
        cy + box_height / 2,
    ), axis=1)
    xyxy *= np.asarray([width, height, width, height], dtype=np.float32)
    np.clip(xyxy, [0, 0, 0, 0], [width, height, width, height], out=xyxy)

    mask_outputs = None
    if masks is not None:
        mask_outputs = np.asarray(masks, dtype=np.float32)[0]
        if mask_outputs.ndim != 3:
            raise ValueError(
                "RF-DETR ONNX masks must have shape (batch, queries, height, width)."
            )

    objects = []
    for index, (box, score, class_id) in enumerate(zip(xyxy, scores, class_ids)):
        label = _class_name(classes, int(class_id))
        if not _is_selected_label(label, labels):
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        item = {
            "label": label,
            "score": float(score),
            "box": [x1, y1, x2 - x1, y2 - y1],
        }
        if mask_outputs is not None:
            import cv2

            resized_mask = cv2.resize(
                mask_outputs[query_indices[index]],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            full_mask = (resized_mask > 0).astype(np.uint8)
            item["mask"] = _crop_mask_to_box(
                full_mask, item["box"]
            )
        objects.append(item)
    return objects


class RFDETRONNXModel(OnnxSessionMixin):
    """Run RF-DETR ONNX models without loading the training package."""

    def __init__(self, model_uri, *, providers=None):
        model_path, metadata = load_onnx_model_config(model_uri)
        self.task = metadata["task"]
        self.classes = list(metadata["classes"])
        self.input_shape = list(metadata["inputShape"])
        self._init_onnx_session(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        output_names = [
            output.name.lower() for output in self.session.get_outputs()
        ]
        self.boxes_index = next(
            (index for index, name in enumerate(output_names) if "dets" in name),
            None,
        )
        self.logits_index = next(
            (index for index, name in enumerate(output_names) if "labels" in name),
            None,
        )
        self.masks_index = next(
            (index for index, name in enumerate(output_names) if "masks" in name),
            None,
        )
        if self.boxes_index is None or self.logits_index is None:
            self.close()
            raise ValueError(
                "RF-DETR ONNX must expose 'dets' and 'labels' outputs."
            )
        if self.task == "segmentation" and self.masks_index is None:
            self.close()
            raise ValueError(
                "RF-DETR segmentation ONNX must expose a 'masks' output."
            )

    def run(
        self, image, conf_threshold=0.35, iou_threshold=0.7, labels=None, **kwargs
    ):
        del iou_threshold, kwargs
        self._ensure_open()
        import cv2

        image = _to_rgb_uint8(image)
        input_height, input_width = self.input_shape[2:]
        resized = cv2.resize(
            image,
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = (
            input_tensor - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        input_tensor = np.ascontiguousarray(input_tensor.transpose(2, 0, 1)[None])
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return _decode_rfdetr_outputs(
            outputs[self.boxes_index],
            outputs[self.logits_index],
            outputs[self.masks_index] if self.masks_index is not None else None,
            image.shape[:2],
            self.classes,
            conf_threshold,
            labels,
        )


def _nms_indices(boxes, scores, threshold, class_ids):
    """Return score-ordered, class-aware NMS indices using NumPy."""
    order = np.argsort(-scores, kind="stable")
    kept = []
    while order.size:
        selected = order[0]
        kept.append(selected)
        remaining = order[1:]
        if not remaining.size:
            break
        x1 = np.maximum(boxes[selected, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[selected, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[selected, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[selected, 3], boxes[remaining, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area_selected = (
            max(boxes[selected, 2] - boxes[selected, 0], 0)
            * max(boxes[selected, 3] - boxes[selected, 1], 0)
        )
        area_remaining = (
            np.maximum(boxes[remaining, 2] - boxes[remaining, 0], 0)
            * np.maximum(boxes[remaining, 3] - boxes[remaining, 1], 0)
        )
        union = area_selected + area_remaining - intersection
        overlaps = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        same_class = class_ids[remaining] == class_ids[selected]
        order = remaining[(overlaps <= threshold) | ~same_class]
    return np.asarray(kept, dtype=np.int64)


class DFineONNXModel(OnnxSessionMixin):
    """Run D-FINE's exported ONNX graph without importing its Torch wrapper."""

    def __init__(self, model_uri, *, providers=None):
        model_path, metadata = load_onnx_model_config(model_uri)
        self.task = metadata["task"]
        self.classes = list(metadata["classes"])
        self.input_shape = list(metadata["inputShape"])
        self._init_onnx_session(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        output_names = [
            output.name.lower() for output in self.session.get_outputs()
        ]
        self.labels_index = next(
            (i for i, name in enumerate(output_names) if "label" in name),
            0,
        )
        self.boxes_index = next(
            (i for i, name in enumerate(output_names) if "box" in name),
            1,
        )
        self.scores_index = next(
            (i for i, name in enumerate(output_names) if "score" in name),
            2,
        )
        self.masks_index = next(
            (i for i, name in enumerate(output_names) if "mask" in name),
            None,
        )
        if len(output_names) < 3:
            self.close()
            raise ValueError(
                "D-FINE ONNX must expose labels, boxes, and scores outputs."
            )
        if self.task == "segmentation" and self.masks_index is None:
            self.close()
            raise ValueError("D-FINE segmentation ONNX must expose masks.")

    def run(
        self, image, conf_threshold=0.35, iou_threshold=0.7, labels=None, **kwargs
    ):
        del kwargs
        self._ensure_open()
        import cv2

        image = _to_rgb_uint8(image)
        original_height, original_width = image.shape[:2]
        input_height, input_width = self.input_shape[2:]
        resized = cv2.resize(
            image,
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_tensor = np.ascontiguousarray(
            (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes = np.asarray(outputs[self.boxes_index], dtype=np.float32)[0].copy()
        scores = np.asarray(outputs[self.scores_index], dtype=np.float32)[0]
        class_ids = np.asarray(outputs[self.labels_index])[0].astype(int)
        boxes[:, [0, 2]] *= original_width / input_width
        boxes[:, [1, 3]] *= original_height / input_height
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_height)

        threshold = 0.0 if conf_threshold is None else float(conf_threshold)
        selected = scores >= threshold
        if labels is not None:
            selected &= np.asarray([
                _is_selected_label(_class_name(self.classes, class_id), labels)
                for class_id in class_ids
            ])
        selected_indices = np.flatnonzero(selected)
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        class_ids = class_ids[selected_indices]
        if boxes.size:
            nms_indices = _nms_indices(
                boxes,
                scores,
                0.7 if iou_threshold is None else float(iou_threshold),
                class_ids,
            )
            selected_indices = selected_indices[nms_indices]
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]
            class_ids = class_ids[nms_indices]

        mask_outputs = None
        if self.masks_index is not None:
            mask_outputs = np.asarray(
                outputs[self.masks_index], dtype=np.float32
            )[0]
            if mask_outputs.ndim != 3:
                raise ValueError(
                    "D-FINE ONNX masks must have shape "
                    "(batch, detections, height, width)."
                )
        objects = []
        for index, (box, score, class_id) in enumerate(
            zip(boxes, scores, class_ids)
        ):
            label = _class_name(self.classes, class_id)
            x1, y1, x2, y2 = (float(value) for value in box[:4])
            item = {
                "label": label,
                "score": float(score),
                "box": [x1, y1, x2 - x1, y2 - y1],
            }
            if mask_outputs is not None:
                mask = cv2.resize(
                    mask_outputs[selected_indices[index]],
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = mask >= 0.5
                inside_box = (
                    (np.arange(original_width)[None, :] >= x1)
                    & (np.arange(original_width)[None, :] < x2)
                    & (np.arange(original_height)[:, None] >= y1)
                    & (np.arange(original_height)[:, None] < y2)
                )
                full_mask = (mask & inside_box).astype(np.uint8)
                item["mask"] = _crop_mask_to_box(
                    full_mask, item["box"]
                )
            objects.append(item)
        return objects


__all__ = ["DFineONNXModel", "RFDETRONNXModel"]
