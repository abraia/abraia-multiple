import os
import pytest
from abraia import Client, APIError

client = Client()
userid = client.check()
filename = 'tiger.jpg'


def test_list_files():
    """Test an API call to list stored files and folders"""
    files, folders = client.list()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_file():
    """Tests an API call to upload a local file"""
    resp = client.upload(
        os.path.join('images', filename),
        userid+'/', type='image/jpeg')
    assert isinstance(resp, dict)


def test_download_file():
    """Tests an API call to download an stored file"""
    resp = client.download(userid+'/'+filename)
    assert resp.status_code == 200


def test_remote_file():
    """Test an API call to upload a remote file"""
    url = 'https://abraia.me/images/random.jpg'
    resp = client.remote(url, userid+'/')
    assert isinstance(resp, dict)
    # assert source.path == 'random.jpg'


def test_transform():
    """Test an API call to transform an image"""
    resp = client.transform(os.path.join(userid, filename))
    assert resp.status_code == 200


# def test_analyze():
#     """Test an API call to analyze an image"""
#     json = client.analyze(os.path.join(userid, filename))
#     assert isinstance(json, dict)


def test_aesthetics():
    """Test an API call to predict image aeshetics"""
    json = client.aesthetics(os.path.join(userid, filename))
    assert isinstance(json, dict)


def test_transcode():
    """Test an API call to transcode a video file"""
    with pytest.raises(APIError):
        client.transcode(os.path.join(userid, filename))


def test_remove_file():
    """Test an API call to remove an stored file"""
    resp = client.remove(os.path.join(userid, filename))
    assert isinstance(resp, dict)
