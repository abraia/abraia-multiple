import os

from .client import Client


def list(path=''):
    return Abraia().list_files(path=path)


def from_file(filename):
    return Abraia().from_file(filename)


def from_url(url):
    return Abraia().from_url(url)


def from_store(path):
    return Abraia().from_store(path)


class Abraia(Client):
    def __init__(self):
        super(Abraia, self).__init__()
        self.userid = self.check()
        self.path = ''
        self.params = {}

    def from_file(self, file):
        resp = self.upload_file(file, self.userid + '/')
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        resp = self.upload_remote(url, self.userid + '/')
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def from_store(self, path):
        self.path = self.userid + '/' + path
        self.params = {}
        return self

    def to_file(self, filename):
        root, ext = os.path.splitext(filename)
        if self.params and ext:
            self.params['fmt'] = ext.lower()[1:]
        resp = self.transform_image(self.path, self.params)
        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return self

    def resize(self, width=None, height=None, mode=None):
        if width:
            self.params['w'] = width
        if height:
            self.params['h'] = height
        if mode:
            self.params['m'] = mode
        return self

    def filter(self, filter):
        self.params['f'] = filter
        return self

    def remove(self):
        return self.remove_file(self.path)
