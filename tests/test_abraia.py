import os
from abraia import abraia


def test_list():
    """Test an API call to list stored files and folders"""
    files, folders = abraia.list()
    assert isinstance(files, list)
    assert isinstance(folders, list)


def test_from_file():
    """Tests an API call to upload a local file"""
    resp = abraia.from_file('images/tiger.jpg')
    assert isinstance(resp, abraia.Client)
    assert resp.path.endswith('tiger.jpg')


def test_from_url():
    """Test an API call to upload a remote file"""
    url = 'https://upload.wikimedia.org/wikipedia/commons/f/f1/100_m_final_Berlin_2009.JPG'
    source = abraia.from_url(url)
    assert isinstance(source, abraia.Client)
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


def test_restore_stored_image():
    abraia.from_store('birds.jpg').to_file('images/birds.bak.jpg')
    assert os.path.isfile('images/birds.bak.jpg')


def test_remove_stored_image():
    resp = abraia.from_store('tiger.jpg').remove()
    assert resp['name'] == 'tiger.jpg'


# def test_exception():
#     """Test an API exception to save no file"""
#     source = abraia.from_url('https://abraia.me/images/tiger.jpg')
#     with pytest.raises(client.APIError):
#         source.to_file(os.path.join(
#             os.path.dirname(__file__), '../images/error.jpg'))
