import cv2
import numpy as np

from .removebg import RemoveBG
from .upscale import ESRGAN, SwinIR
from .smartcrop import Smartcrop


from ..detect import load_model
from ..faces import Recognition
from .. import draw


def detect_faces(img):
    recognition = Recognition()
    return recognition.detect_faces(img)


def detect_plates(img):
    detection = load_model('multiple/models/alpd-seg.onnx')
    return detection.run(img, approx=0.02)


def detect_smartcrop(img, size):
    smartcrop = Smartcrop()
    return smartcrop.detect(img, size)


def build_mask(img, plates, faces):
    mask = np.zeros(img.shape[:2], np.uint8)
    [draw.draw_filled_polygon(mask, result['polygon'], 255) for result in plates]
    [draw.draw_filled_ellipse(mask, result['box'], 255) for result in faces]
    return mask


def anonymize_image(img):
    plates = detect_plates(img)
    faces = detect_faces(img)
    mask = build_mask(img, plates, faces)
    out = draw.draw_blurred_mask(img, mask)
    return out


def remove_background(img):
    removebg = RemoveBG()
    out = removebg.remove(img)
    return out


def upscale_image(img):
    if max(img.shape) > 1920:
        h, w = img.shape[:2]
        scale = 1920 / max(img.shape)
        size = (round(scale * w), round(scale * h))
        img = cv2.resize(img, size, cv2.INTER_LINEAR)
    upscaler = ESRGAN()
    out = upscaler.upscale(img)
    return out


def smartcrop_image(img, size):
    smartcrop = Smartcrop()
    return smartcrop.transform(img, size)
