import numpy as np

from abraia.utils import load_image
from abraia.editing import smartcrop_image


def test_smart_crop():
    img = load_image('images/mick-jagger.jpg')
    resized = smartcrop_image(img, (150, 300))
    assert resized.dtype == np.uint8
    assert resized.shape == (300, 150, 3)


# def test_smart_resize_gray():
#     img = load_image('images/skate_gray.jpg')
#     resized = smartcrop_image(img, (150, 300))
#     assert resized.dtype == np.uint8
#     assert resized.shape == (300, 150)


# def test_smart_resize_transparent():
#     img = load_image('images/logo.png')
#     resized = smartcrop_image(img, (100, 100))
#     assert resized.dtype == np.uint8
#     assert resized.shape == (100, 100, 4)
