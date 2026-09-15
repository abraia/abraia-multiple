"""Pillow-backed image loading and encoding helpers."""

import base64
from io import BytesIO

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from .filesystem import make_dirs


pillow_heif.register_heif_opener()


def load_image(src, mode="RGB", max_size=2048):
    """Load an image as a NumPy array, optionally constraining its size."""
    with Image.open(src) as opened:
        image = ImageOps.exif_transpose(opened).convert(mode)
    if max_size is not None and (image.width > max_size or image.height > max_size):
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.array(image)


def encode_image(image, format="PNG", mode=None):
    """Encode a NumPy image array and return its encoded bytes."""
    pil_image = Image.fromarray(np.asarray(image))
    if mode is not None:
        pil_image = pil_image.convert(mode)
    with BytesIO() as buffer:
        pil_image.save(buffer, format=format)
        return buffer.getvalue()


def save_image(img, dest):
    make_dirs(dest)
    Image.fromarray(img).save(dest)
    return dest


def show_image(img):
    Image.fromarray(img).show()


def image_base64(img, format="jpeg"):
    image = Image.fromarray(img)
    with BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format};base64,{encoded}"


__all__ = [
    "encode_image",
    "image_base64",
    "load_image",
    "save_image",
    "show_image",
]
