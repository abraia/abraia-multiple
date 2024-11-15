import numpy as np

from PIL import Image

from .rembg import RemoveBG
from .upscale import ESRGAN, SwinIR


def remove_background(im):
    rembg = RemoveBG()
    img = np.array(im)
    out = rembg.remove(img)
    return Image.fromarray(out)


def upscale_image(im):
    if max(im.size) > 512:
        im.thumbnail([512, 512])
    upscaler = ESRGAN()
    img = np.array(im)
    out = upscaler.upscale(img)
    return Image.fromarray(out)
