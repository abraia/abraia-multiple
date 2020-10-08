import os

from io import BytesIO
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
    files, folders = client.list_files(userid+'/')
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_upload_remote():
    """Tests an API call to upload a remote file"""
    url = 'https://api.abraia.me/files/demo/birds.jpg'
    resp = client.upload_remote(url, userid+'/')
    assert resp['name'] == 'birds.jpg'


def test_upload_file():
    """Tests an API call to upload a local file"""
    resp = client.upload_file(os.path.join('images', filename), userid+'/', type='image/jpeg')
    assert resp['name'] == 'tiger.jpg'


def test_move_file():
    """Test an API call to move a stored file"""
    client.move_file(os.path.join(userid, filename), userid + '/test/tiger.jpg')
    resp = client.move_file(userid + '/test/tiger.jpg', os.path.join(userid, filename))
    assert resp['name'] == 'tiger.jpg'


def test_download_file():
    """Tests an API call to download an stored file"""
    resp = client.download_file(os.path.join(userid, 'tiger.jpg'))
    assert isinstance(resp, BytesIO)


def test_load_metadata():
    """Tests an API call to load metadata from an stored file"""
    resp = client.load_metadata(os.path.join(userid, 'tiger.jpg'))
    assert resp['MIMEType'] == 'image/jpeg'
    # assert resp['ImageSize'] == '1920x1271'


def test_analyze_image():
    """Tests an API call to analyze an image"""
    resp = client.analyze_image(os.path.join(userid, 'tiger.jpg'), {'ar': 1})
    assert isinstance(resp, dict)


def test_transform_image():
    """Test an API call to transform an image"""
    resp = client.transform_image(os.path.join(userid, 'tiger.jpg'), {'width': 333})
    assert isinstance(resp, BytesIO)


def test_transform_video():
    """Test an API call to transform a video"""
    resp = client.transform_video(os.path.join(userid, 'videos/bigbuckbunny.mp4'), {'format': 'jpg'})
    assert isinstance(resp, dict)


def test_remove_file():
    """Test an API call to remove an stored file"""
    resp = client.remove_file(os.path.join(userid, 'tiger.jpg'))
    assert resp['name'] == 'tiger.jpg'
