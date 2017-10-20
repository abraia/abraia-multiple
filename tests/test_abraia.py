import os
from abraia import abraia


def test_from_file():
    """Tests an API call to upload a local file"""
    source = abraia.from_file(os.path.join(
        os.path.dirname(__file__), '../images/lion.jpg'))
    assert isinstance(source, abraia.Client)


def test_from_url():
    """Test an API call to upload a remote file"""
    source = abraia.from_url('https://abraia.me/images/tiger.jpg')
    assert isinstance(source, abraia.Client)


def test_to_file():
    """Test an API call to save to local file"""
    source = abraia.from_url('https://abraia.me/images/tiger.jpg')
    source.to_file(os.path.join(
        os.path.dirname(__file__), '../images/optimized.jpg'))


def test_resize():
    """Test an API call to resize an image"""
    source = abraia.from_url('https://abraia.me/images/lion.jpg')
    resized = source.resize(width=150, height=150)
    resized.to_file(os.path.join(
        os.path.dirname(__file__), '../images/resized.jpg'))
