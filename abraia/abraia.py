import requests
from . import config

session = requests.Session()
session.auth = config.load_auth()


def from_file(filename):
    return Client().from_file(filename)


def from_url(url):
    return Client().from_url(url)


class APIError(Exception):
    def __init__(self, message):
        super(APIError, self).__init__(message)
        self.message = message


class Client:
    def __init__(self):
        self.url = ''
        self.params = {}
        self.resp = ''

    def files(self):
        path = '{}/images'.format(config.api_url)
        resp = session.get(path)
        return resp.json()

    def from_file(self, filename):
        path = '{}/images'.format(config.api_url)
        files = dict(file=open(filename, 'rb'))
        resp = session.post(path, files=files)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(path, resp.status_code))
        self.resp = resp.json()
        self.url = '{}/images/{}'.format(config.api_url, self.resp['filename'])
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        self.url = '{}/images'.format(config.api_url)
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
        url = '{}/images/{}'.format(config.api_url, filename)
        resp = session.delete(url)
        return resp.json()
