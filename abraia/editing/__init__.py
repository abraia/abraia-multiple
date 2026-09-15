import cv2
import numpy as np
from functools import lru_cache

from .removebg import BackgroundRemover
from .upscale import Upscaler, SwinIR
from .smartcrop import Smartcrop
from .inpaint import LAMA
from ..inference.sam import SAM

from ..inference import PlateDetector
from ..inference.faces import FaceRecognizer, Retinaface
from ..utils import draw, Sketcher


@lru_cache(maxsize=1)
def _face_detector():
    return Retinaface()


@lru_cache(maxsize=1)
def _plate_detector():
    return PlateDetector()


@lru_cache(maxsize=1)
def _smartcrop():
    return Smartcrop()


@lru_cache(maxsize=1)
def _background_remover():
    return BackgroundRemover()


@lru_cache(maxsize=1)
def _upscaler():
    return Upscaler()


@lru_cache(maxsize=1)
def _lama():
    return LAMA()


def detect_faces(img):
    return _face_detector().detect_faces(img)


def detect_plates(img):
    return _plate_detector().detect(img)


def detect_smartcrop(img, size):
    return _smartcrop().detect(img, size)


def build_mask(img, plates, faces):
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
    plates = detect_plates(img)
    faces = detect_faces(img)
    mask = build_mask(img, plates, faces)
    out = draw.draw_blurred_mask(img, mask)
    return out


def remove_background(img):
    return _background_remover().remove(img)


def blur_background(img):
    mask = (np.ones(img.shape[:2]) * 255).astype(np.uint8)
    back = draw.draw_blurred_mask(img.copy(), mask)
    fore = remove_background(img)
    out = draw.draw_overlay(back, fore)
    return out


def upscale_image(img):
    if max(img.shape) > 1920:
        h, w = img.shape[:2]
        scale = 1920 / max(img.shape)
        size = (round(scale * w), round(scale * h))
        img = cv2.resize(img, size, cv2.INTER_LINEAR)
    return _upscaler().upscale(img)


def smartcrop_image(img, size):
    return _smartcrop().transform(img, size)


def inpaint_image(img, mask):
    return _lama().inpaint(img, mask)


def clean_image(img):
    from ..inference.sam import InteractiveSAM
    interactive_sam = InteractiveSAM(img)
    return interactive_sam.interactive_mask(
        callback=lambda i, m: _lama().inpaint(i, m)
    )
