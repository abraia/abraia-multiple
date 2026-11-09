"""ONNX inference for CLIP-based RGB image classifiers."""

import numpy as np
from PIL import Image

from abraia.inference.models.classification import postprocess
from abraia.inference.models.detection import validate_model_config
from abraia.inference.session import OnnxSessionMixin, load_onnx_model_config


CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_clip(image, input_shape):
    """Resize and normalize an RGB image using CLIP's 224-pixel transform."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "CLIP classification expects a grayscale, RGB, or RGBA image"
        )
    image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)

    target_height, target_width = map(int, input_shape[2:4])
    height, width = image.shape[:2]
    if width < height:
        resized_width = target_width
        resized_height = int(target_width * height / width)
    else:
        resized_height = target_height
        resized_width = int(target_height * width / height)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), Image.Resampling.BICUBIC
    )
    left = (resized_width - target_width) // 2
    top = (resized_height - target_height) // 2
    cropped = np.asarray(resized)[
        top:top + target_height, left:left + target_width
    ].astype(np.float32) / 255.0
    normalized = (cropped - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


class ClipClassifier(OnnxSessionMixin):
    """Run an exported PyTorch CLIP image classifier."""

    def __init__(self, model_uri, *, providers=None, accelerator=None):
        model_path, self.config = load_onnx_model_config(model_uri)
        self.task, self.input_shape, self.classes = validate_model_config(
            self.config
        )
        if self.task != "classification":
            raise ValueError("CLIP models require the classification task")
        self._init_onnx_session(
            model_path,
            providers=providers,
            accelerator=accelerator,
        )
        self.input_name = self.session.get_inputs()[0].name

    def run(
        self, image, top_k=1, score_threshold=None, labels=None,
        conf_threshold=None, **kwargs,
    ):
        """Classify one image with the exported CLIP model."""
        del kwargs
        self._ensure_open()
        if score_threshold is None:
            score_threshold = conf_threshold
        outputs = self.session.run(
            None,
            {self.input_name: preprocess_clip(image, self.input_shape)},
        )
        return postprocess(
            outputs,
            self.classes,
            top_k=top_k,
            score_threshold=score_threshold,
            labels=labels,
        )


__all__ = ["ClipClassifier", "preprocess_clip"]
