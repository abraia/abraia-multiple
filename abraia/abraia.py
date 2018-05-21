import os
import requests
from io import BytesIO

from . import config

session = requests.Session()
session.auth = config.load_auth()


def _remote_file(url):
    imgbuf = None
    try:
        response = requests.get(url)
        content_type = response.headers['content-type']
        if content_type in config.MIME_TYPES.values():
            imgbuf = BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(e)
    if imgbuf is None:
        raise APIError('Resource not found: {}'.format(url))
    return imgbuf


def from_file(filename):
    return Client().from_file(filename)


def from_url(url):
    return Client().from_url(url)


def from_store(path):
    return Client().from_store(path)


def list():
    return Client().list()


def remove(path):
    return Client().delete(path)


class Client:
    def __init__(self):
        self.path = ''
        self.params = {}

    def from_file(self, file):
        url = '{}/images'.format(config.API_URL)
        file = file if isinstance(file, BytesIO) else open(file, 'rb')
        files = dict(file=file)
        resp = session.post(url, files=files)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(url, resp.status_code))
        file = resp.json()['file']
        self.path = file.get('source')
        self.params = {'q': 'auto'}
        return self

    def from_store(self, path):
        self.path = path
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        return self.from_file(_remote_file(url))

    def to_file(self, filename):
        root, ext = os.path.splitext(filename)
        self.params['fmt'] = ext.lower()[1:] if ext != '' else None
        url = '{}/images/{}'.format(config.API_URL, self.path)
        resp = session.get(url, params=self.params, stream=True)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return self

    def resize(self, width=None, height=None):
        if width:
            self.params['w'] = width
        if height:
            self.params['h'] = height
        return self

    def analyze(self):
        url = '{}/analysis/{}'.format(config.API_URL, self.path)
        resp = session.get(url, params=self.params)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp.json()

    def list(self):
        url = '{}/images'.format(config.API_URL)
        resp = session.get(url)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        files = resp.json()['files']
        return files

    def delete(self, path):
        url = '{}/images/{}'.format(config.API_URL, path)
        resp = session.delete(url)
        if resp.status_code != 200:
            raise APIError('DELETE {} {}'.format(url, resp.status_code))
        return resp.json()


class APIError(Exception):
    def __init__(self, message):
        super(APIError, self).__init__(message)
        self.message = message
