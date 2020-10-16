import os

from .client import Client


class Abraia(Client):
    def __init__(self, folder=''):
        super(Abraia, self).__init__()
        self.userid = self.load_user()['id']
        self.folder = folder
        self.params = {}
        self.path = ''

    def list(self, folder=''):
        length = len(self.userid) + 1
        files, folders = self.list_files(path=self.userid + '/' + folder)
        files = map(lambda f: {'name': f['name'], 'size': f['size'],
                               'date': f['date'], 'path': f['source'][length:]}, files)
        folders = map(lambda f: {'name': f['name'], 'path': f['source'][length:]}, folders) 
        return list(files), list(folders)

    def from_file(self, file):
        resp = self.upload_file(file, self.userid + '/' + self.folder)
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def from_url(self, url):
        resp = self.upload_remote(url, self.userid + '/' + self.folder)
        self.path = resp['source']
        self.params = {'q': 'auto'}
        return self

    def from_store(self, path):
        self.path = self.userid + '/' + path
        self.params = {}
        return self

    def to_buffer(self, format=None):
        if format and self.params:
            self.params['format'] = format.lower()
        buffer = self.transform_image(self.path, self.params)
        return buffer

    def to_file(self, filename):
        root, ext = os.path.splitext(filename)
        if ext and self.params:
            self.params['format'] = ext.lower()[1:]
        buffer = self.transform_image(self.path, self.params)
        with open(filename, 'wb') as f:
            f.write(buffer.getvalue())
        return self

    def resize(self, width=None, height=None, mode=None):
        if width:
            self.params['width'] = width
        if height:
            self.params['height'] = height
        if mode:
            self.params['mode'] = mode
        return self

    def process(self, params={}):
        self.params.update(params)
        return self

    def remove(self):
        return self.remove_file(self.path)
