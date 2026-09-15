"""High-level image editing operations.

Model-backed editors are imported and instantiated lazily so importing
``abraia.editing`` remains cheap and does not initialize ONNX sessions.
"""

from functools import lru_cache

import cv2
import numpy as np

from ..utils import draw


@lru_cache(maxsize=1)
def _face_detector():
    from ..inference.faces import Retinaface

    return Retinaface()


@lru_cache(maxsize=1)
def _plate_detector():
    from ..inference.plates import PlateDetector

    return PlateDetector()


@lru_cache(maxsize=1)
def _smartcrop():
    from .smartcrop import Smartcrop

    return Smartcrop()


@lru_cache(maxsize=1)
def _background_remover():
    from .removebg import BackgroundRemover

    return BackgroundRemover()


@lru_cache(maxsize=1)
def _upscaler():
    from .upscale import Upscaler

    return Upscaler()


@lru_cache(maxsize=1)
def _lama():
    from .inpaint import LAMA

    return LAMA()


def detect_faces(img):
    """Detect faces in an image."""
    return _face_detector().detect_faces(img)


def detect_plates(img):
    """Detect license plates in an image."""
    return _plate_detector().detect(img)


def detect_smartcrop(img, size):
    """Return the recommended crop rectangle for ``size``."""
    return _smartcrop().detect(img, size)


def build_mask(img, plates, faces):
    """Build a uint8 anonymization mask from plate and face detections."""
    mask = np.zeros(img.shape[:2], np.uint8)
    for result in plates or []:
        polygon = result.get('polygon')
        if polygon is not None and len(polygon):
            draw.draw_filled_polygon(mask, polygon, 255)
        elif result.get('mask') is not None and result.get('box') is not None:
            draw.draw_mask(mask, result['mask'], result['box'], 255)
    for result in faces or []:
        box = result.get('box')
        if box is not None:
            draw.draw_filled_ellipse(mask, box, 255)
    return mask


def anonymize_image(img):
    """Blur detected faces and license plates in ``img``."""
    plates = detect_plates(img)
    faces = detect_faces(img)
    return draw.draw_blurred_mask(img, build_mask(img, plates, faces))


def remove_background(img):
    """Remove the background from an RGB image."""
    return _background_remover().remove(img)


def blur_background(img):
    """Blur the background while retaining the detected foreground."""
    mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    back = draw.draw_blurred_mask(img.copy(), mask)
    return draw.draw_overlay(back, remove_background(img))


def upscale_image(img):
    """Upscale an image, limiting oversized inputs before inference."""
    if max(img.shape) > 1920:
        height, width = img.shape[:2]
        scale = 1920 / max(img.shape)
        size = (round(scale * width), round(scale * height))
        img = cv2.resize(img, size, cv2.INTER_LINEAR)
    return _upscaler().upscale(img)


def smartcrop_image(img, size):
    """Crop an image to preserve salient content at the requested size."""
    return _smartcrop().transform(img, size)


def inpaint_image(img, mask):
    """Fill masked pixels using the LAMA inpainting model."""
    return _lama().inpaint(img, mask)


def clean_image(img):
    """Interactively select and inpaint an image region."""
    from ..inference.sam import InteractiveSAM

    interactive_sam = InteractiveSAM(img)
    return interactive_sam.interactive_mask(
        callback=lambda image, mask: _lama().inpaint(image, mask)
    )


__all__ = [
    'anonymize_image',
    'blur_background',
    'build_mask',
    'clean_image',
    'detect_faces',
    'detect_plates',
    'detect_smartcrop',
    'inpaint_image',
    'remove_background',
    'smartcrop_image',
    'upscale_image',
]
