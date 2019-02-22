import os
import pytest
from abraia import Client, APIError

client = Client()
userid = client.load_user()['user']['id']
filename = 'tiger.jpg'


def test_load_user():
    """Test an API call to load user info"""
    resp = client.load_user()
    assert isinstance(resp, dict)


def test_list_files():
    """Test an API call to list stored files and folders"""
    files, folders = client.list_files()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_remote():
    """Tests an API call to upload a remote file"""
    url = 'https://api.abraia.me/files/demo/birds.jpg'
    resp = client.upload_remote(url, userid+'/')
    assert isinstance(resp, dict)
    assert resp['name'] == 'birds.jpg'


def test_upload_file():
    """Tests an API call to upload a local file"""
    resp = client.upload_file(
        os.path.join('images', filename),
        userid+'/', type='image/jpeg')
    assert isinstance(resp, dict)
    assert resp['name'] == 'tiger.jpg'


def test_move_file():
    """Test an API call to move a stored file"""
    client.move_file(os.path.join(userid, filename), userid + '/test/tiger.jpg')
    resp = client.move_file(userid + '/test/tiger.jpg', os.path.join(userid, filename))
    assert isinstance(resp, dict)
    assert resp['name'] == 'tiger.jpg'


def test_download_file():
    """Tests an API call to download an stored file"""
    resp = client.download_file(userid+'/'+filename)
    assert resp.status_code == 200


def test_transform_image():
    """Test an API call to transform an image"""
    resp = client.transform_image(os.path.join(userid, filename))
    assert resp.status_code == 200


def test_process_video():
    """Test an API call to process a video file"""
    with pytest.raises(APIError):
        client.process_video(os.path.join(userid, filename))


def test_remove_file():
    """Test an API call to remove an stored file"""
    resp = client.remove_file(os.path.join(userid, filename))
    assert isinstance(resp, dict)
    assert resp['name'] == 'tiger.jpg'
