import os

from io import BytesIO
from abraia import Abraia
from PIL import Image

abraia = Abraia()


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


def test_optimize_image():
    abraia.transform_image('birds.jpg', 'images/birds_o.jpg')
    assert os.path.isfile('images/birds_o.jpg')


def test_resize_image():
    abraia.transform_image('lion.jpg', 'images/lion_o.jpg', params={'width': 333})
    assert os.path.isfile('images/lion_o.jpg')


def test_smartcrop_image():
    abraia.transform_image('birds.jpg', 'images/birds_375x375.jpg', {'width': 375, 'height': 375})
    assert os.path.isfile('images/birds_375x375.jpg')


def test_convert_image():
    abraia.upload_file('images/bat.svg')
    abraia.transform_image('bat.svg', 'images/bat.png')
    assert os.path.isfile('images/bat.png')


def test_screenshot_webpage():
    abraia.transform_image('screenshot.jpg', 'images/screenshot.jpg', {'url': 'https://www.bbc.com'})
    assert os.path.isfile('images/screenshot.jpg')


def test_load_file():
    stream = abraia.load_file('lion.jpg')
    assert isinstance(stream, BytesIO)


def test_load_metadata():
    meta = abraia.load_metadata('lion.jpg')
    assert meta['MIMEType'] == 'image/jpeg'


def test_save_file():
    path = abraia.save_file('test.txt', 'this is a simple test.')
    assert path == 'test.txt'


def test_load_image():
    im = abraia.load_image('lion.jpg')
    assert isinstance(im, Image.Image)


def test_save_image():
    im = abraia.load_image('lion.jpg')
    path = abraia.save_image('lion.png', im)
    assert path == 'lion.png'
