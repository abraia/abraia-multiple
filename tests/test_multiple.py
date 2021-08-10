import os
import numpy as np

from multiple import Multiple

multiple = Multiple()


def test_load_image():
    img = multiple.load_image('lion.jpg')
    assert isinstance(img, np.ndarray)


def test_load_metadata():
    meta = multiple.load_metadata('lion.jpg')
    assert meta['MIMEType'] == 'image/jpeg'


def test_save_image():
    img = multiple.load_image('lion.jpg')
    resp = multiple.save_image('lion.png', img)
    assert resp['path'] == 'lion.png'
