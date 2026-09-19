"""NumPy and mask helpers shared by SDK and Studio code."""

import numpy as np
from PIL import Image

from .image import encode_image


def as_array(value, dtype=None, copy=False):
    array = np.asarray(value, dtype=dtype)
    return array.copy() if copy else array


def array_from_image_buffer(buffer, width, height, stride, channels=3):
    rows = np.frombuffer(buffer, dtype=np.uint8).reshape(int(height), int(stride))
    return rows[:, : int(width) * int(channels)].reshape(
        int(height), int(width), int(channels)
    ).copy()


def array_shape(value):
    return np.asarray(value).shape


def array_ndim(value):
    return np.asarray(value).ndim


def array_size(value):
    return np.asarray(value).size


def array_copy(value):
    return np.asarray(value).copy()


def copy_mask(value):
    """Return an owned boolean mask independent of the input buffer."""
    return np.asarray(value).copy() > 0


def array_squeeze(value):
    return np.squeeze(value)


def array_to_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


def zeros_array(shape, dtype="uint8"):
    return np.zeros(shape, dtype=dtype)


def mask_array(value):
    return np.asarray(value) > 0


def merge_masks(target, mask):
    np.logical_or(target, mask, out=target)
    return target


def encode_mask_overlay(mask, color=(32, 184, 121), alpha=96):
    mask = mask_array(mask)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = np.where(mask, int(alpha), 0).astype(np.uint8)
    return encode_image(rgba, format="PNG")


def compose_mask_layers(layers, shape, alpha=96):
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    combined = np.zeros(shape, dtype=bool)
    for mask, color in layers or []:
        mask = mask_array(mask)
        if mask.shape != tuple(shape):
            continue
        np.logical_or(combined, mask, out=combined)
        rgba[mask, :3] = color
        rgba[mask, 3] = int(alpha)
    return combined, encode_image(rgba, format="PNG")


def combined_mask(layers, shape):
    """Combine shape-compatible masks from colored mask layers."""
    combined = np.zeros(tuple(shape), dtype=bool)
    for mask, _color in layers or []:
        mask = mask_array(mask)
        if mask.shape == combined.shape:
            np.logical_or(combined, mask, out=combined)
    return combined


def mask_matches_image(mask, image):
    """Return whether a mask matches the first two image dimensions."""
    return np.asarray(mask).shape == np.asarray(image).shape[:2]


def resize_mask(mask, size):
    mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    resized = mask_image.resize(
        (int(size[0]), int(size[1])), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


__all__ = [
    "array_copy",
    "array_from_image_buffer",
    "array_ndim",
    "array_shape",
    "array_size",
    "array_squeeze",
    "array_to_list",
    "as_array",
    "combined_mask",
    "compose_mask_layers",
    "copy_mask",
    "encode_mask_overlay",
    "mask_array",
    "mask_matches_image",
    "merge_masks",
    "resize_mask",
    "zeros_array",
]
