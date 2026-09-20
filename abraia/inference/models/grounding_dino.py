"""Grounding DINO inference through ONNX Runtime.

The exported model is deliberately kept separate from the generic YOLO-style
ONNX detector. Grounding DINO consumes tokenized text as well as an image and
returns token-level scores, so its labels are determined at inference time.
"""

import os
import re

import cv2
import numpy as np

from ..session import OnnxSessionMixin
from ...utils import get_providers, load_json, resolve_model_file


DEFAULT_MODEL_URI = "multiple/models/grounding_dino_tiny.onnx"
DEFAULT_TOKENIZER_URI = "multiple/models/grounding_dino_vocab.txt"
DEFAULT_INPUT_SHAPE = (1, 3, 800, 800)
DEFAULT_MAX_TEXT_LENGTH = 256
COREML_PROVIDER = "CoreMLExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _grounding_dino_providers(providers):
    """Return providers that do not route this graph through CoreML.

    ONNX Runtime's CoreML provider can hang Grounding DINO on macOS while
    repeatedly reporting ``Context leak detected``.  Keep CUDA when it is
    explicitly available, but never leave CoreML in the provider chain,
    including as a partial-execution fallback.
    """
    if providers is None:
        providers = get_providers()
    safe_providers = [
        provider for provider in providers
        if str(provider) != COREML_PROVIDER
    ]
    if not safe_providers:
        return [CPU_PROVIDER]
    if CPU_PROVIDER not in safe_providers:
        safe_providers.append(CPU_PROVIDER)
    return safe_providers


def _resolve_file(uri):
    """Resolve a local path or fetch an SDK-managed remote asset."""
    return resolve_model_file(uri)


def _dtype_for_input(type_name):
    """Return the NumPy dtype corresponding to an ONNX input type."""
    type_name = str(type_name).lower()
    if "int64" in type_name:
        return np.int64
    if "int32" in type_name:
        return np.int32
    if "uint8" in type_name:
        return np.uint8
    if "bool" in type_name:
        return np.bool_
    return np.float32


def _input_kind(name):
    """Classify an exported input name without depending on one exporter."""
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if normalized in {"image", "images", "pixelvalues", "img"}:
        return "image"
    if normalized in {"inputids", "inputid", "ids"}:
        return "input_ids"
    if normalized in {"attentionmask", "textattentionmask"}:
        return "attention_mask"
    if normalized in {"tokentypeids", "tokenids", "segmentids"}:
        return "token_type_ids"
    if normalized in {"pixelmask", "imagemask"}:
        return "pixel_mask"
    if normalized in {"positionids", "positionid"}:
        return "position_ids"
    if normalized in {"textselfattentionmasks", "textselfattentionmask"}:
        return "text_self_attention_masks"
    return None


def prepare_image(
    image,
    input_shape=DEFAULT_INPUT_SHAPE,
    letterbox=True,
    force_full_pixel_mask=False,
):
    """Resize a BGR image into normalized, letterboxed model inputs.

    Returns ``(pixel_values, pixel_mask, scale, padding)``. The transform
    maps original coordinates to model coordinates. ``letterbox=False`` is
    useful for fixed-square exports that require every pixel-mask value to be
    valid.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Grounding DINO expects an image with three channels")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    target_height, target_width = map(int, input_shape[2:4])
    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Grounding DINO cannot process an empty image")
    scale = min(target_width / image_width, target_height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    if not letterbox:
        resized_width, resized_height = target_width, target_height
        scale = (target_width / image_width, target_height / image_height)
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0
    resized = (resized - IMAGE_MEAN) / IMAGE_STD

    pad_left = (target_width - resized_width) // 2 if letterbox else 0
    pad_top = (target_height - resized_height) // 2 if letterbox else 0
    pixel_values = np.zeros(
        (1, 3, target_height, target_width), dtype=np.float32
    )
    pixel_values[0, :, pad_top:pad_top + resized_height,
                 pad_left:pad_left + resized_width] = resized.transpose(2, 0, 1)
    pixel_mask = np.zeros((1, target_height, target_width), dtype=np.bool_)
    pixel_mask[0, pad_top:pad_top + resized_height,
               pad_left:pad_left + resized_width] = True
    if force_full_pixel_mask:
        pixel_mask.fill(True)
    if isinstance(scale, tuple):
        scale = tuple(float(value) for value in scale)
    else:
        scale = float(scale)
    return pixel_values, pixel_mask, scale, (pad_left, pad_top)


def _clean_phrase(value):
    value = re.sub(r"\s+", " ", str(value)).strip(" .,")
    return value


def _phrase_from_tokens(
    tokenizer, input_ids, token_mask, prompt, offsets=None, token_scores=None
):
    """Recover a human-readable prompt phrase from active token positions."""
    indices = np.flatnonzero(token_mask)
    if not len(indices):
        return ""
    if offsets is not None:
        spans = [
            (index, int(offsets[index][0]), int(offsets[index][1]))
            for index in indices
            if index < len(offsets) and offsets[index][1] > offsets[index][0]
        ]
        if spans:
            # A query can activate tokens from more than one dot-separated
            # candidate. Select the candidate with the strongest token.
            groups = []
            current = []
            for item in spans:
                if current and "." in prompt[current[-1][2]:item[1]]:
                    groups.append(current)
                    current = []
                current.append(item)
            if current:
                groups.append(current)
            if token_scores is None:
                group = groups[0]
            else:
                group = max(
                    groups,
                    key=lambda values: max(
                        float(token_scores[index]) for index, _, _ in values
                    ),
                )
            return _clean_phrase(prompt[group[0][1]:group[-1][2]])
    token_ids = [int(input_ids[index]) for index in indices]
    return _clean_phrase(tokenizer.decode(token_ids, skip_special_tokens=True))


def decode_outputs(
    outputs,
    input_ids,
    tokenizer,
    prompt,
    image_size,
    input_shape=DEFAULT_INPUT_SHAPE,
    box_threshold=0.35,
    text_threshold=0.25,
    transform=None,
    iou_threshold=None,
    offsets=None,
):
    """Decode Grounding DINO logits and boxes into SDK detection results."""
    if len(outputs) < 2:
        raise ValueError("Grounding DINO ONNX output must contain logits and boxes")
    logits = np.asarray(outputs[0])
    boxes = np.asarray(outputs[1])
    if logits.ndim == 2:
        logits = logits[None, ...]
    if boxes.ndim == 2:
        boxes = boxes[None, ...]
    if logits.ndim != 3 or boxes.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        raise ValueError("Unexpected Grounding DINO output shapes")

    scores = 1.0 / (1.0 + np.exp(-logits[0]))
    max_scores = scores.max(axis=1)
    if transform is None:
        image_width, image_height = image_size
        model_height, model_width = map(int, input_shape[2:4])
        scale = min(model_width / image_width, model_height / image_height)
        padding = (0.0, 0.0)
    else:
        scale, padding = transform
    image_width, image_height = map(float, image_size)
    model_height, model_width = map(int, input_shape[2:4])

    candidates = []
    for index, score in enumerate(max_scores):
        if float(score) < float(box_threshold):
            continue
        token_mask = scores[index] > float(text_threshold)
        phrase = _phrase_from_tokens(
            tokenizer,
            input_ids,
            token_mask,
            prompt,
            offsets=offsets,
            token_scores=scores[index],
        )
        if not phrase:
            continue
        center_x, center_y, width, height = boxes[0, index] * np.array(
            [model_width, model_height, model_width, model_height],
            dtype=np.float32,
        )
        pad_x, pad_y = padding
        if isinstance(scale, (tuple, list)):
            scale_x, scale_y = map(float, scale)
        else:
            scale_x = scale_y = float(scale)
        x1 = ((center_x - width / 2) - pad_x) / scale_x
        y1 = ((center_y - height / 2) - pad_y) / scale_y
        x2 = ((center_x + width / 2) - pad_x) / scale_x
        y2 = ((center_y + height / 2) - pad_y) / scale_y
        x1, y1 = max(0, min(image_width, x1)), max(0, min(image_height, y1))
        x2, y2 = max(0, min(image_width, x2)), max(0, min(image_height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "label": phrase,
            "score": float(score),
            "box": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
            "class_id": int(index),
        })

    if iou_threshold is None or len(candidates) < 2:
        return candidates
    return _nms_by_phrase(candidates, float(iou_threshold))


def _nms_by_phrase(results, threshold):
    """Apply simple per-phrase NMS while preserving SDK result dictionaries."""
    kept = []
    for phrase in dict.fromkeys(result["label"] for result in results):
        group = [result for result in results if result["label"] == phrase]
        group.sort(key=lambda result: result["score"], reverse=True)
        while group:
            current = group.pop(0)
            kept.append(current)
            group = [
                result for result in group
                if _box_iou(current["box"], result["box"]) <= threshold
            ]
    return sorted(kept, key=lambda result: result["score"], reverse=True)


def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union if union else 0.0


class GroundingDINOModel(OnnxSessionMixin):
    """Run a Grounding DINO ONNX graph with text prompts."""

    def __init__(
        self,
        model_uri=DEFAULT_MODEL_URI,
        tokenizer_uri=None,
        providers=None,
    ):
        model_uri = os.fspath(model_uri)
        model_path = _resolve_file(model_uri)
        config_uri = f"{os.path.splitext(model_uri)[0]}.json"
        if os.path.isfile(config_uri):
            config_path = config_uri
        else:
            config_path = _resolve_file(config_uri)
        self.config = load_json(config_path)
        self.input_shape = tuple(
            self.config.get("inputShape", DEFAULT_INPUT_SHAPE)
        )
        self.max_text_length = int(
            self.config.get("maxTextLength", DEFAULT_MAX_TEXT_LENGTH)
        )
        self.letterbox = bool(self.config.get("letterbox", True))
        self.force_full_pixel_mask = bool(
            self.config.get("forceFullPixelMask", False)
        )
        self.tokenizer = self._load_tokenizer(
            tokenizer_uri or self.config.get("tokenizer", DEFAULT_TOKENIZER_URI)
        )
        try:
            self._init_onnx_session(
                model_path,
                providers=_grounding_dino_providers(providers),
            )
            self._inputs = {
                item.name: item for item in self.session.get_inputs()
            }
        except Exception:
            # ``get_inputs`` may fail after the native session is created.
            # Release it here because a partially constructed model cannot be
            # closed by the caller.
            self.close()
            raise

    @staticmethod
    def _load_tokenizer(tokenizer_uri):
        try:
            from transformers import BertTokenizerFast
        except ImportError as error:
            raise ImportError(
                "Grounding DINO requires the optional 'transformers' dependency"
            ) from error
        vocab_path = _resolve_file(tokenizer_uri)
        return BertTokenizerFast(vocab_file=vocab_path, do_lower_case=True)

    def _tokenize(self, prompt):
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_offsets_mapping=True,
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        offsets = encoded.get("offset_mapping")
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        values = {
            "input_ids": input_ids,
            "attention_mask": np.asarray(
                encoded["attention_mask"], dtype=np.int64
            )[None, :] if np.asarray(encoded["attention_mask"]).ndim == 1
            else np.asarray(encoded["attention_mask"], dtype=np.int64),
            "token_type_ids": np.asarray(
                encoded.get("token_type_ids", np.zeros_like(input_ids)),
                dtype=np.int64,
            ),
        }
        if values["token_type_ids"].ndim == 1:
            values["token_type_ids"] = values["token_type_ids"][None, :]
        if offsets is not None and np.asarray(offsets).ndim == 3:
            offsets = np.asarray(offsets)[0].tolist()
        elif offsets is not None:
            offsets = np.asarray(offsets).tolist()
        return values, offsets

    def _build_inputs(self, image, token_values, pixel_mask):
        values = {
            "image": image,
            "pixel_mask": pixel_mask,
            **token_values,
        }
        input_data = {}
        for name, input_info in self._inputs.items():
            kind = _input_kind(name)
            if kind == "position_ids":
                length = token_values["input_ids"].shape[1]
                value = np.arange(length, dtype=np.int64)[None, :]
            elif kind == "text_self_attention_masks":
                length = token_values["input_ids"].shape[1]
                value = np.ones((1, length, length), dtype=np.bool_)
            elif kind in values:
                value = values[kind]
            else:
                raise ValueError(f"Unsupported Grounding DINO input: {name}")
            input_data[name] = np.asarray(
                value, dtype=_dtype_for_input(input_info.type)
            )
        return input_data

    def run(
        self,
        img,
        prompt=None,
        text_prompt=None,
        labels=None,
        box_threshold=0.35,
        text_threshold=0.25,
        conf_threshold=None,
        iou_threshold=0.7,
    ):
        """Detect objects described by a prompt or a list of candidate labels."""
        self._ensure_open()
        if prompt is None:
            prompt = text_prompt
        if prompt is None:
            if not labels:
                raise ValueError("Grounding DINO requires 'prompt' or 'labels'")
            prompt = ". ".join(str(label).strip().strip(".") for label in labels) + "."
        elif not isinstance(prompt, str):
            prompt = ". ".join(str(label).strip().strip(".") for label in prompt) + "."
        else:
            # The bundled ONNX graph was exported with a period-separated
            # prompt. Its traced special-token handling expects the period
            # token between the text and [SEP], even for one label.
            prompt = prompt.strip()
            if not prompt.endswith((".", "?")):
                prompt += "."
        if conf_threshold is not None:
            box_threshold = conf_threshold

        pixel_values, pixel_mask, scale, padding = prepare_image(
            img,
            self.input_shape,
            letterbox=self.letterbox,
            force_full_pixel_mask=self.force_full_pixel_mask,
        )
        token_values, offsets = self._tokenize(prompt)
        inputs = self._build_inputs(pixel_values, token_values, pixel_mask)
        outputs = self.session.run(None, inputs)
        return decode_outputs(
            outputs,
            token_values["input_ids"][0],
            self.tokenizer,
            prompt,
            image_size=(img.shape[1], img.shape[0]),
            input_shape=self.input_shape,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            transform=(scale, padding),
            iou_threshold=iou_threshold,
            offsets=offsets,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "DEFAULT_MODEL_URI",
    "GroundingDINOModel",
    "decode_outputs",
    "prepare_image",
]
