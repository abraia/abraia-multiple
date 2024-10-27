import numpy as np

from PIL import Image

from .rembg import RemoveBG
from .upscale import Upscaler


def remove_background(im):
    rembg = RemoveBG()
    img = np.array(im)
    out = rembg.remove(img, True)
    return Image.fromarray(out)


def upscale_image(im):
    # if max(im.size) > 1024:
    #     im = im.thumbnail([1024, 1024])
    upscaler = Upscaler()
    img = np.array(im)
    out = upscaler.upscale(img)
    return Image.fromarray(out)
