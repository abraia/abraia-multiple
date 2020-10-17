import os
import sys
import pytest

from io import BytesIO
from abraia import Abraia, APIError

abraia = Abraia()
userid = abraia.load_user()['id']
filename = 'tiger.jpg'


def test_load_user():
    """Test an API call to load user info"""
    user = abraia.load_user()
    assert isinstance(user, dict)


def test_list_files():
    """Test an API call to list stored files and folders"""
    files, folders = abraia.list_files(userid+'/')
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_remote():
    """Tests an API call to upload a remote file"""
    url = 'https://api.abraia.me/files/demo/birds.jpg'
    resp = abraia.upload_remote(url, userid+'/')
    assert resp['name'] == 'birds.jpg'


def test_upload_file():
    """Tests an API call to upload a local file"""
    resp = abraia.upload_file(os.path.join('images', filename), userid+'/')
    assert resp['name'] == 'tiger.jpg'


def test_move_file():
    """Test an API call to move a stored file"""
    abraia.move_file(os.path.join(userid, filename),
                     userid + '/test/tiger.jpg')
    resp = abraia.move_file(userid + '/test/tiger.jpg',
                            os.path.join(userid, filename))
    assert resp['name'] == 'tiger.jpg'


def test_download_file():
    """Tests an API call to download an stored file"""
    resp = abraia.download_file(os.path.join(userid, 'tiger.jpg'))
    assert isinstance(resp, BytesIO)


def test_load_metadata():
    """Tests an API call to load metadata from an stored file"""
    resp = abraia.load_metadata(os.path.join(userid, 'tiger.jpg'))
    assert resp['MIMEType'] == 'image/jpeg'
    # assert resp['ImageSize'] == '1920x1271'


def test_analyze_image():
    """Tests an API call to analyze an image"""
    resp = abraia.analyze_image(os.path.join(userid, 'tiger.jpg'), {'ar': 1})
    assert isinstance(resp, dict)


def test_transform_image():
    """Test an API call to transform an image"""
    resp = abraia.transform_image(os.path.join(
        userid, 'tiger.jpg'), {'width': 333})
    assert isinstance(resp, BytesIO)


def test_transform_video():
    """Test an API call to transform a video"""
    resp = abraia.transform_video(os.path.join(
        userid, 'videos/bigbuckbunny.mp4'), {'format': 'jpg'})
    assert isinstance(resp, dict)


def test_remove_file():
    """Test an API call to remove an stored file"""
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


def test_from_url():
    """Test an API call to upload a remote file"""
    url = 'https://upload.wikimedia.org/wikipedia/commons/f/f1/100_m_final_Berlin_2009.JPG'
    source = abraia.from_url(url)
    assert isinstance(source, Abraia)
    assert source.path.endswith('100_m_final_Berlin_2009.JPG')


def test_optimize_image_from_url():
    abraia.from_url('https://api.abraia.me/files/demo/birds.jpg').to_file(
        'images/birds_o.jpg')
    assert os.path.isfile('images/birds_o.jpg')


def test_resize_image_from_file():
    abraia.from_file('images/lion.jpg').resize(
        width=500).to_file('images/lion_500.jpg')
    assert os.path.isfile('images/lion_500.jpg')


def test_smartcrop_image_from_file():
    abraia.from_file('images/birds.jpg').resize(
        width=375, height=375).to_file('images/birds_375x375.jpg')
    assert os.path.isfile('images/birds_375x375.jpg')


def test_process_branded_image():
    abraia.from_store('lion.jpg').process(
        {'action': 'test.atn', 'height': 333}).to_file('images/lion_brand.jpg')
    assert os.path.isfile('images/lion_brand.jpg')


def test_restore_stored_image():
    buffer = abraia.from_store('birds.jpg').to_buffer()
    with open('images/birds.jpg', 'rb') as f:
        buff = BytesIO(f.read())
    assert sys.getsizeof(buff) == sys.getsizeof(buffer)


def test_remove_stored_image():
    resp = abraia.from_store('tiger.jpg').remove()
    assert resp['name'] == 'tiger.jpg'


def test_server_error():
    with pytest.raises(APIError) as excinfo:
        abraia.from_file('images/fake.jpg')
    error = excinfo.value
    assert error.code == 501
