import os
import numpy as np
import pytest

from io import BytesIO
from abraia import Abraia

abraia = Abraia()

pytestmark = pytest.mark.skipif(
    os.environ.get("ABRAIA_RUN_LIVE_TESTS") != "1",
    reason="Live Abraia API tests are disabled; set ABRAIA_RUN_LIVE_TESTS=1",
)


def test_list_files():
    files, folders = abraia.list_files()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_file():
    path = abraia.upload_file('images/car.jpg')
    path = abraia.upload_file('images/lion.jpg')
    assert path == 'lion.jpg'
    path = abraia.upload_file('images/birds.jpg', 'birds.jpg')
    assert path == 'birds.jpg'
    path = abraia.upload_file('https://api.abraia.me/files/demo/tests/tiger.jpg', 'tiger.jpg')
    assert path == 'tiger.jpg'


def test_check_file():
    path = 'lion.jpg'
    assert abraia.check_file(path)
    path = '_lion.jpg'
    assert not abraia.check_file(path)


def test_download_file():
    dest = abraia.download_file('tiger.jpg', 'images/tiger.jpg')
    assert dest == 'images/tiger.jpg'


def test_remove_file():
    path = abraia.remove_file('tiger.jpg')
    assert path == 'tiger.jpg'


def test_load_file():
    stream = abraia.load_file('lion.jpg')
    assert isinstance(stream, BytesIO)


def test_save_file():
    path = abraia.save_file('test.txt', 'this is a simple test.')
    assert path == 'test.txt'


def test_load_image():
    img = abraia.load_image('lion.jpg')
    assert isinstance(img, np.ndarray)


def test_save_image():
    img = abraia.load_image('lion.jpg')
    path = abraia.save_image('lion.png', img)
    assert path == 'lion.png'
