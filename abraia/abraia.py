import requests
from datetime import datetime

from . import config

session = requests.Session()
session.auth = config.load_auth()


def from_file(filename):
    return Client().from_file(filename)


def from_url(url):
    return Client().from_url(url)


def list():
    json = Client().files()
    files = [(datetime.fromtimestamp(
        f['date']), f['size'], f['name']) for f in json['files']]
    return '\n'.join(['{}  {:>7}  {}'.format(*f) for f in files])


def remove(path):
    return Client().delete(path)


class Client:
    def __init__(self):
        self.url = ''
        self.params = {}
        self.resp = ''

    def files(self):
        path = '{}/images'.format(config.API_URL)
        resp = session.get(path)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(self.url, resp.status_code))
        return resp.json()

    def from_file(self, filename):
        path = '{}/images'.format(config.API_URL)
        files = dict(file=open(filename, 'rb'))
        resp = session.post(path, files=files)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(path, resp.status_code))
        self.resp = resp.json()
        self.url = '{}/images/{}'.format(config.API_URL, self.resp['filename'])
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        self.url = '{}/images'.format(config.API_URL)
        self.params = {'url': url, 'q': 'auto'}
        return self

    def to_file(self, filename):
        resp = session.get(self.url, params=self.params, stream=True)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(self.url, resp.status_code))
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

    def delete(self, filename):
        url = '{}/images/{}'.format(config.API_URL, filename)
        resp = session.delete(url)
        if resp.status_code != 200:
            raise APIError('DELETE {} {}'.format(self.url, resp.status_code))
        return resp.json()


class APIError(Exception):
    def __init__(self, message):
        super(APIError, self).__init__(message)
        self.message = message
