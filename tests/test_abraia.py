import os
import sys
import pytest

from io import BytesIO
from abraia import Abraia, APIError

abraia = Abraia()
userid = abraia.load_user()['id']
filename = 'tiger.jpg'


def test_load_user_info():
    user = abraia.load_user()
    assert isinstance(user, dict)


def test_list_files():
    files, folders = abraia.list_files(userid+'/')
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_remote():
    url = 'https://api.abraia.me/files/demo/birds.jpg'
    resp = abraia.upload_remote(url, userid+'/')
    assert resp['name'] == 'birds.jpg'


def test_upload_file():
    resp = abraia.upload_file(os.path.join('images', filename), userid+'/')
    assert resp['name'] == 'tiger.jpg'


def test_move_file():
    abraia.move_file(os.path.join(userid, filename), userid + '/test/tiger.jpg')
    resp = abraia.move_file(userid + '/test/tiger.jpg', os.path.join(userid, filename))
    assert resp['name'] == 'tiger.jpg'


def test_download_file():
    resp = abraia.download_file(os.path.join(userid, 'tiger.jpg'))
    assert isinstance(resp, BytesIO)


def test_load_metadata():
    resp = abraia.load_metadata(os.path.join(userid, 'tiger.jpg'))
    assert resp['MIMEType'] == 'image/jpeg'


# def test_analyze_image():
#     resp = abraia.analyze_image(os.path.join(userid, 'tiger.jpg'), {'ar': 1})
#     assert isinstance(resp, dict)


def test_optimize_image():
    resp = abraia.transform_image(os.path.join(userid, 'birds.jpg'), {'q': 'auto'})
    assert isinstance(resp, BytesIO)


def test_resize_image():
    resp = abraia.transform_image(os.path.join(userid, 'tiger.jpg'), {'width': 333})
    assert isinstance(resp, BytesIO)


def test_convert_svg_image():
    resp = abraia.transform_image(os.path.join(userid, 'bat.svg'), {})
    assert isinstance(resp, BytesIO)


def test_screenshot_webpage():
    resp = abraia.transform_image(
        os.path.join(userid, 'screenshot.jpg'), {'url': 'https://www.bbc.com'})
    assert isinstance(resp, BytesIO)


def test_remove_file():
    resp = abraia.remove_file(os.path.join(userid, 'tiger.jpg'))
    assert resp['name'] == 'tiger.jpg'


def test_list():
    """Test an API call to list stored files and folders"""
    files, folders = abraia.list()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_from_file():
    """Tests an API call to upload a local file"""
    resp = abraia.from_file('images/tiger.jpg')
    assert isinstance(resp, Abraia)
    assert resp.path.endswith('tiger.jpg')


def test_upload_from_url():
    resp = abraia.upload(
        'https://upload.wikimedia.org/wikipedia/commons/f/f1/100_m_final_Berlin_2009.JPG')
    assert resp['path'] == '100_m_final_Berlin_2009.JPG'


def test_smartcrop_image_from_file():
    abraia.from_file('images/birds.jpg').resize(width=375, height=375).to_file('images/birds_375x375.jpg')
    assert os.path.isfile('images/birds_375x375.jpg')


def test_process_branded_image():
    abraia.transform('lion.jpg', 'images/lion_brand.jpg', {'action': 'test.atn', 'height': 333})
    assert os.path.isfile('images/lion_brand.jpg')


def test_remove_stored_image():
    resp = abraia.remove('tiger.jpg')
    assert resp['path'] == 'tiger.jpg'


def test_server_error():
    with pytest.raises(APIError) as excinfo:
        abraia.from_file('images/fake.jpg')
    error = excinfo.value
    assert error.code == 501
