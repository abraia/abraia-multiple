import os
import pytest
from abraia import Client, APIError

client = Client()
userid = client.check()
filename = 'tiger.jpg'


def test_list_files():
    """Test an API call to list stored files and folders"""
    files, folders = client.list_files()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_file():
    """Tests an API call to upload a local file"""
    resp = client.upload_file(
        os.path.join('images', filename),
        userid+'/', type='image/jpeg')
    assert isinstance(resp, dict)


def test_download_file():
    """Tests an API call to download an stored file"""
    resp = client.download_file(userid+'/'+filename)
    assert resp.status_code == 200


def test_move_file():
    """Test an API call to move a stored file"""
    client.move_file(os.path.join(userid, filename), userid + '/test/' + filename)
    resp = client.move_file(userid + '/test/' + filename, os.path.join(userid, filename))
    assert isinstance(resp, dict)
    assert resp['file']['source'] == userid + '/' + filename


def test_remote_file():
    """Test an API call to upload a remote file"""
    url = 'https://abraia.me/images/random.jpg'
    resp = client.remote(url, userid+'/')
    assert isinstance(resp, dict)


def test_transform_image():
    """Test an API call to transform an image"""
    resp = client.transform_image(os.path.join(userid, filename))
    assert resp.status_code == 200


# def test_analyze_image():
#     """Test an API call to analyze an image"""
#     json = client.analyze_image(os.path.join(userid, filename))
#     assert isinstance(json, dict)


def test_aesthetics_image():
    """Test an API call to predict image aeshetics"""
    json = client.aesthetics_image(os.path.join(userid, filename))
    assert isinstance(json, dict)


def test_process_video():
    """Test an API call to process a video file"""
    with pytest.raises(APIError):
        client.process_video(os.path.join(userid, filename))


def test_remove_file():
    """Test an API call to remove an stored file"""
    resp = client.remove_file(os.path.join(userid, filename))
    assert isinstance(resp, dict)
