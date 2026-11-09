"""ONNX image embeddings from the DINOv2 ViT-B/14 register backbone."""

import numpy as np
from PIL import Image

from abraia.inference.session import OnnxSessionMixin
from abraia.utils.remote import resolve_model_file


DEFAULT_MODEL_URI = "multiple/models/dinov2/dinov2_vitb14_reg_518.onnx"
DEFAULT_IMAGE_SIZE = 518
EMBEDDING_SIZE = 768


class DinoV2ImageEncoder(OnnxSessionMixin):
    """Compute global DINOv2 image features with an ONNX Runtime session."""

    def __init__(
        self,
        model_uri=DEFAULT_MODEL_URI,
        *,
        batch_size=1,
        providers=None,
        accelerator=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        self.batch_size = int(batch_size)
        self._init_onnx_session(
            resolve_model_file(model_uri),
            providers=providers,
            accelerator=accelerator,
        )
        try:
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            shape = model_input.shape
            if len(shape) != 4 or shape[1] != 3:
                raise ValueError(
                    "DINOv2 ONNX input must be NCHW with three RGB channels"
                )
            fixed_batch_size = shape[0]
            if isinstance(fixed_batch_size, int) and fixed_batch_size > 0:
                self.batch_size = min(self.batch_size, fixed_batch_size)
            self.input_height, self.input_width = map(int, shape[2:])
            if (
                self.input_height != self.input_width
                or self.input_height < 224
                or self.input_height % 14
            ):
                raise ValueError(
                    "DINOv2 ONNX input must be square, at least 224 pixels, "
                    "and divisible by 14"
                )
            model_output = self.session.get_outputs()[0]
            self.output_name = model_output.name
            output_shape = model_output.shape
            self.embedding_size = (
                int(output_shape[-1])
                if len(output_shape) == 2 and isinstance(output_shape[-1], int)
                else EMBEDDING_SIZE
            )
        except Exception:
            self.close()
            raise

    def get_image_embeddings(self, images):
        """Return one CLS-token feature vector per RGB image.

        Images may be Pillow images or uint8 NumPy arrays. Inputs are resized
        to the fixed model resolution and converted to RGB float32 in [0, 1];
        ImageNet normalization is part of the exported ONNX graph.
        """
        self._ensure_open()
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_size), dtype=np.float32)

        embeddings = []
        for start in range(0, len(images), self.batch_size):
            batch = np.stack([
                _prepare_image(image, self.input_width, self.input_height)
                for image in images[start:start + self.batch_size]
            ])
            output = np.asarray(
                self.session.run(
                    [self.output_name], {self.input_name: batch}
                )[0],
                dtype=np.float32,
            )
            expected = (len(batch), self.embedding_size)
            if output.shape != expected:
                raise ValueError(
                    "DINOv2 ONNX returned embeddings with shape "
                    f"{output.shape}; expected {expected}"
                )
            embeddings.append(output)
        return np.concatenate(embeddings, axis=0)


def _prepare_image(image, width, height):
    """Resize a grayscale, RGB, or RGBA uint8 image for DINOv2."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("DINOv2 expects a grayscale, RGB, or RGBA image")
        image = np.asarray(image, dtype=np.uint8)

    resized = Image.fromarray(image).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    return np.ascontiguousarray(
        np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    )


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MODEL_URI",
    "DinoV2ImageEncoder",
    "EMBEDDING_SIZE",
]
