import os
import io
import urllib
import hashlib
import requests
import mimetypes

from fnmatch import fnmatch
from datetime import datetime
from io import BytesIO
from . import config


def md5sum(src):
    hash_md5 = hashlib.md5()
    with open(src, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class APIError(Exception):
    def __init__(self, message, code=0):
        super(APIError, self).__init__(message, code)
        self.code = code
        try:
            self.message = message.json()['message']
        except:
            self.message = ''


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
        return {}

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
        md5 = md5sum(file) # TODO: Refactor names to src, dest, path (cloud)
        type = mimetypes.guess_type(name)[0] or 'binary/octet-stream'
        json = {'name': name, 'type': type, 'md5': md5}  if md5 else {'name': name, 'type': type}
        url = '{}/files/{}'.format(config.API_URL, source)
        resp = requests.post(url, json=json, auth=self.auth) # TODO: Refactor to reduce requests code (like with client.js)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        url = resp.get('uploadURL')
        if url:
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

    # def analyze_image(self, path, params={}):
    #     url = '{}/analysis/{}'.format(config.API_URL, path)
    #     resp = requests.get(url, auth=self.auth)
    #     if resp.status_code != 200:
    #         raise APIError(resp.text, resp.status_code)
    #     resp = resp.json()
    #     if resp.get('salmap'):
    #         resp['salmap'] = BytesIO(base64.b64decode(resp['salmap'][23:]))
    #     return resp

    def detect_labels(self, path, params={}):
        url = '{}/rekognition/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, auth=self.auth)
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


class Abraia(Client):
    def __init__(self, folder=''):
        super(Abraia, self).__init__()
        self.userid = self.load_user().get('id')
        self.folder = folder
        self.params = {}
        self.path = ''

    def list(self, path=''):
        length = len(self.userid) + 1
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        # TODO: Change path to manage userid
        files, folders = self.list_files(path=self.userid + '/' + folder)
        files = list(map(lambda f: {'path': f['source'][length:], 'name': f['name'], 'size': f['size'], 'date': f['date']}, files))
        folders = list(map(lambda f: {'path': f['source'][length:], 'name': f['name']}, folders))
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
            folders = list(filter(lambda f: fnmatch(f['path'], path), folders))
        return files, folders

    def upload(self, src, path=''):
        length = len(self.userid) + 1
        if isinstance(src, str) and src.startswith('http'):
            # f = self.upload_remote(src, self.userid + '/' + path)
            # return {'path': f['source'][length:]}
            src = urllib.request.urlretrieve(src)[0]
        f = self.upload_file(src, self.userid + '/' + path)
        return {'path': f['source'][length:]}

    def download(self, path, dest=''):
        buffer = self.download_file(self.userid + '/' + path)
        if dest:
            with open(dest, 'wb') as f:
                f.write(buffer.getbuffer())
            return dest
        return buffer

    def remove(self, path):
        length = len(self.userid) + 1
        f = self.remove_file(self.userid + '/' + path)
        return {'path': f['source'][length:]}

    def transform(self, path, dest, params={'quality': 'auto'}):
        ext = dest.split('.').pop().lower()
        params['format'] = self.params.get('format') or ext
        buffer = self.transform_image(self.userid + '/' + path, params=params)
        with open(dest, 'wb') as f:
            f.write(buffer.getvalue())

    def load(self, path):
        return self.download_file(self.userid + '/' + path)

    def metadata(self, path):
        return self.load_metadata(self.userid + '/' + path)

    def save(self, path, buffer):
         self.upload(io.BytesIO(buffer), path)
    