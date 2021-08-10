import os
import pytest

from io import BytesIO
from abraia import Abraia

abraia = Abraia()


def test_load_user_info():
    user = abraia.load_user()
    assert isinstance(user, dict)


def test_list_files():
    files, folders = abraia.list()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_file():
    resp = abraia.upload('images/lion.jpg')
    assert resp['path'] == 'lion.jpg'
    resp = abraia.upload('images/tiger.jpg', 'tiger.jpg')
    assert resp['path'] == 'tiger.jpg'


def test_upload_remote():
    # url = 'https://upload.wikimedia.org/wikipedia/commons/f/f1/100_m_final_Berlin_2009.JPG'
    resp = abraia.upload('https://api.abraia.me/files/demo/birds.jpg', 'birds.jpg')
    assert resp['path'] == 'birds.jpg'


# def test_move_file():
#     abraia.move_file(os.path.join(userid, 'tiger.jpg'), userid + '/test/tiger.jpg')
#     resp = abraia.move_file(userid + '/test/tiger.jpg', os.path.join(userid, filename))
#     assert resp['name'] == 'tiger.jpg'


def test_download_file():
    resp = abraia.download('tiger.jpg')
    assert isinstance(resp, BytesIO)


def test_remove_file():
    resp = abraia.remove('tiger.jpg')
    assert resp['path'] == 'tiger.jpg'


# TODO: Remove analyze image (depreciated) and add detect image
# def test_analyze_image():
#     resp = abraia.analyze_image(os.path.join(userid, 'tiger.jpg'), {'ar': 1})
#     assert isinstance(resp, dict)


def test_optimize_image():
    abraia.transform('birds.jpg', 'images/birds_o.jpg')
    assert os.path.isfile('images/birds_o.jpg')


def test_resize_image():
    abraia.transform('lion.jpg', 'images/lion_o.jpg', params={'width': 333})
    assert os.path.isfile('images/lion_o.jpg')


def test_smartcrop_image():
    abraia.transform('birds.jpg', 'images/birds_375x375.jpg', {'width': 375, 'height': 375})
    assert os.path.isfile('images/birds_375x375.jpg')


def test_convert_image():
    abraia.upload('images/bat.svg')
    abraia.transform('bat.svg', 'images/bat.png')
    assert os.path.isfile('images/bat.png')


def test_screenshot_webpage():
    abraia.transform('screenshot.jpg', 'images/screenshot.jpg', {'url': 'https://www.bbc.com'})
    assert os.path.isfile('images/screenshot.jpg')


def test_process_branded_image():
    abraia.transform('lion.jpg', 'images/lion_brand.jpg', {'action': 'test.atn', 'height': 333})
    assert os.path.isfile('images/lion_brand.jpg')


def test_load_file():
    resp = abraia.load('lion.jpg')
    assert isinstance(resp, BytesIO)


def test_load_metadata():
    resp = abraia.metadata('lion.jpg')
    assert resp['MIMEType'] == 'image/jpeg'


def test_save_file():
    resp = abraia.save('test.txt', 'this is a simple test.')
    assert isinstance(resp, dict)
    assert resp['path'].endswith('test.txt')
