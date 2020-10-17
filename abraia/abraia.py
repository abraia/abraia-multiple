import os
import base64
import requests
import mimetypes

from fnmatch import fnmatch
from datetime import datetime
from io import BytesIO
from . import config


class APIError(Exception):
    def __init__(self, message, code=0):
        super(APIError, self).__init__(message, code)
        self.message = message
        self.code = code


class Client(object):
    def __init__(self):
        self.auth = config.load_auth()

    def load_user(self):
        if self.auth[0] and self.auth[1]:
            url = '{}/users'.format(config.API_URL)
            resp = requests.get(url, auth=self.auth)
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            return resp.json()['user']
        raise APIError('Unauthorized', 401)

    def list_files(self, path=''):
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        for f in resp['files']:
            f['date'] = datetime.fromtimestamp(f['date'])
        return resp['files'], resp['folders']

    def upload_remote(self, url, path):
        json = {'url': url}
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return resp['file']

    def upload_file(self, file, path):
        source = path + os.path.basename(file) if path.endswith('/') else path
        name = os.path.basename(source)
        type = mimetypes.guess_type(name)[0] or 'binary/octet-stream'
        # json = {'name': name, 'type': type, 'md5': md5}  if md5 else {'name': name, 'type': type}
        json = {'name': name, 'type': type}
        url = '{}/files/{}'.format(config.API_URL, source)
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        url = resp.json()['uploadURL']
        data = file if isinstance(file, BytesIO) else open(file, 'rb')
        resp = requests.put(url, data=data, headers={'Content-Type': type})
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return {'name': name, 'source': source}

    def move_file(self, old_path, new_path):
        url = '{}/files/{}'.format(config.API_URL, new_path)
        resp = requests.post(url, json={'store': old_path}, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return resp['file']

    def download_file(self, path, dest=''):
        # T0D0 Add dest ptina paramater to save file
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.get(url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return BytesIO(resp.content)

    def remove_file(self, path):
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return resp['file']

    def load_metadata(self, path):
        url = '{}/metadata/{}'.format(config.API_URL, path)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def remove_metadata(self, path):
        url = '{}/metadata/{}'.format(config.API_URL, path)
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def analyze_image(self, path, params={}):
        url = '{}/analysis/{}'.format(config.API_URL, path)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        if resp.get('salmap'):
            resp['salmap'] = BytesIO(base64.b64decode(resp['salmap'][23:]))
        return resp

    def detect_labels(self, path, params={}):
        url = '{}/rekognition/{}'.format(config.API_URL, path)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def transform_image(self, path, params={}):
        if params.get('action'):
            params['background'] = '{}/images/{}'.format(config.API_URL, path)
            if params.get('fmt') is None:
                params['fmt'] = params['background'].split('.').pop()
            path = '{}/{}'.format(path.split('/')[0], params['action'])
        url = '{}/images/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return BytesIO(resp.content)

    def transform_video(self, path, params={}):
        url = '{}/videos/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()


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
