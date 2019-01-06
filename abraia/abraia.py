import os

from .client import Client


class Abraia(Client):
    def __init__(self):
        super(Abraia, self).__init__()
        self.userid = self.check()
        self.path = ''
        self.params = {}

    def from_file(self, file):
        resp = self.upload_file(file, self.userid+'/')
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def from_store(self, path):
        self.path = path
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        resp = self.upload_remote(url, self.userid+'/')
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def to_file(self, filename):
        root, ext = os.path.splitext(filename)
        self.params['fmt'] = ext.lower()[1:] if ext != '' else None
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


def from_file(filename):
    return Abraia().from_file(filename)


def from_url(url):
    return Abraia().from_url(url)


def from_store(path):
    return Abraia().from_store(path)


def list(path=''):
    return Abraia().list_files(path=path)


def remove(path):
    return Abraia().remove_file(path)
