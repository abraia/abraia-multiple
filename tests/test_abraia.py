import os
import pytest
from abraia import abraia


def test_from_file():
    """Tests an API call to upload a local file"""
    source = abraia.from_file(os.path.join(
        os.path.dirname(__file__), '../images/tiger.jpg'))
    assert isinstance(source, abraia.Client)


def test_from_url():
    """Test an API call to upload a remote file"""
    url = 'https://abraia.me/images/random.jpg'
    source = abraia.from_url(url)
    assert isinstance(source, abraia.Client)
    assert source.params['url'] == url


def test_to_file():
    """Test an API call to save to local file"""
    output = os.path.join(
        os.path.dirname(__file__), '../images/optimized.jpg')
    source = abraia.from_url('https://abraia.me/images/random.jpg')
    source.to_file(output)
    assert os.path.isfile(output)


def test_resize():
    """Test an API call to resize an image"""
    output = os.path.join(
        os.path.dirname(__file__), '../images/resized.jpg')
    source = abraia.from_file(os.path.join(
        os.path.dirname(__file__), '../images/lion.jpg'))
    resized = source.resize(width=500)
    resized.to_file(output)
    assert os.path.isfile(output)


def test_smartcrop():
    """Test an API call to smartcrop an image"""
    output = os.path.join(
        os.path.dirname(__file__), '../images/cropped.jpg')
    source = abraia.from_file(os.path.join(
        os.path.dirname(__file__), '../images/lion.jpg'))
    cropped = source.resize(width=333, height=333)
    cropped.to_file(output)
    assert os.path.isfile(output)


def test_exception():
    """Test an API exception to save no file"""
    source = abraia.from_url('https://abraia.me/images/tiger.jpg')
    with pytest.raises(abraia.APIError):
        source.to_file(os.path.join(
            os.path.dirname(__file__), '../images/error.jpg'))
