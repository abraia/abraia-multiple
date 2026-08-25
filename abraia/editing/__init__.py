import cv2
import numpy as np

from .removebg import BackgroundRemover
from .upscale import Upscaler, SwinIR
from .smartcrop import Smartcrop
from .inpaint import LAMA
from ..inference.sam import SAM

from ..inference import PlateDetector
from ..inference.faces import FaceRecognizer
from ..utils import draw, Sketcher


def detect_faces(img):
    recognition = FaceRecognizer()
    return recognition.detect_faces(img)


def detect_plates(img):
    plate = PlateDetector()
    return plate.detect(img)


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
    removebg = BackgroundRemover()
    out = removebg.remove(img)
    return out


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
    upscaler = Upscaler()
    out = upscaler.upscale(img)
    return out


def smartcrop_image(img, size):
    smartcrop = Smartcrop()
    return smartcrop.transform(img, size)


def inpaint_image(img, mask):
    lama = LAMA()
    return lama.inpaint(img, mask)


def clean_image(img):
    from ..inference.sam import InteractiveSAM
    interactive_sam = InteractiveSAM(img)
    lama = LAMA()
    return interactive_sam.interactive_mask(callback=lambda i, m: lama.inpaint(i, m))

