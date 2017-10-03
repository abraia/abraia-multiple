import os
import requests

ABRAIA_API_URL = 'https://abraia.me/api'
ABRAIA_API_KEY = os.environ.get('ABRAIA_API_KEY', None)

session = requests.Session()
# session.params = {}
# session.params['api_key'] = ABRAIA_API_KEY


def from_file(filename):
    return Client().from_file(filename)


def from_url(url):
    return Client().from_url(url)


class Client:
    def __init__(self):
        self.url = ''
        self.params = {}
        self.response = ''

    def files(self):
        path = '{}/images'.format(ABRAIA_API_URL)
        response = session.get(path)
        return response.json()

    def from_file(self, filename):
        path = '{}/images'.format(ABRAIA_API_URL)
        files = dict(file=open(filename, 'rb'))
        response = session.post(path, files=files)
        if response.status_code == 201:
            self.response = response.json()
            self.url = '{}/images/{}'.format(
                ABRAIA_API_URL, self.response['filename'])
            self.params = {}
        return self

    def from_url(self, url):
        self.url = '{}/images'.format(ABRAIA_API_URL)
        self.params = {'url': url}
        return self

    def to_file(self, filename):
        response = session.get(self.url, params=self.params)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
        return response.status_code

    def resize(self, width=None, height=None):
        if width:
            self.params['w'] = width
        if height:
            self.params['h'] = height
        return self

    def delete(self, filename):
        url = '{}/images/{}'.format(ABRAIA_API_URL, filename)
        response = session.delete(url)
        return response.json()
