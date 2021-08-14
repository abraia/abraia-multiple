import os
import io
import hashlib
import requests
import tempfile
import mimetypes
import numpy as np

from PIL import Image
from fnmatch import fnmatch
from datetime import datetime
from . import config


API_URL = 'https://api.abraia.me'
tempdir = tempfile.gettempdir()


def file_path(f, userid):
    return f['source'][len(userid)+1:]


def md5sum(src):
    hash_md5 = hashlib.md5()
    f = io.BytesIO(src.getvalue()) if isinstance(src, io.BytesIO) else open(src, 'rb')
    for chunk in iter(lambda: f.read(4096), b''):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


class APIError(Exception):
    def __init__(self, message, code=0):
        super(APIError, self).__init__(message, code)
        self.code = code
        try:
            self.message = message.json()['message']
        except:
            self.message = ''


class Abraia:
    def __init__(self, folder=''):
        self.auth = config.load_auth()
        self.userid = self.load_user().get('id')
        self.folder = folder

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

    def upload_remote(self, url, path):
        json = {'url': url}
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file'], self.userid)

    def upload_file(self, file, path):
        source = path + os.path.basename(file) if path.endswith('/') else path
        name = os.path.basename(source)
        md5 = md5sum(file) # TODO: Refactor names to src, dest, path (cloud)
        type = mimetypes.guess_type(name)[0] or 'binary/octet-stream'
        json = {'name': name, 'type': type, 'md5': md5}  if md5 else {'name': name, 'type': type}
        url = '{}/files/{}'.format(config.API_URL, source)
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        url = resp.get('uploadURL')
        if url:
            data = file if isinstance(file, io.BytesIO) else open(file, 'rb')
            resp = requests.put(url, data=data, headers={'Content-Type': type})
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
        return {'name': name, 'source': source}

    def upload(self, src, path=''):
        if isinstance(src, str) and src.startswith('http'):
            return self.upload_remote(src, path)
        f = self.upload_file(src, self.userid + '/' + path)
        return file_path(f, self.userid)

    def move_file(self, old_path, new_path):
        url = '{}/files/{}'.format(config.API_URL, new_path)
        resp = requests.post(url, json={'store': old_path}, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return resp['file']

    def download_file(self, path, dest=''):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.get(url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        if dest:
            with open(dest, 'wb') as f:
                f.write(resp.content)
            return dest
        return io.BytesIO(resp.content)

    def remove_file(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file'], self.userid)

    def load_metadata(self, path):
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def remove_metadata(self, path):
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def transform_image(self, path, dest, params={'quality': 'auto'}):
        ext = dest.split('.').pop().lower()
        params['format'] = params.get('format') or ext
        if params.get('action'):
            params['background'] = f"{API_URL}/images/{path}"
            if params.get('fmt') is None:
                params['fmt'] = params['background'].split('.').pop()
            path = '{}/{}'.format(path.split('/')[0], params['action'])
        url = f"{API_URL}/images/{self.userid}/{path}"
        resp = requests.get(url, params=params, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        with open(dest, 'wb') as f:
            f.write(resp.content)

    def load_file(self, path):
        stream = self.download_file(path)
        try:
            return stream.getvalue().decode('utf-8')
        except:
            return stream

    def load_image(self, path):
        stream = self.download_file(path)
        return np.asarray(Image.open(stream))

    def save_file(self, path, stream):
        # TODO: Rename save as save_file
        stream =  io.BytesIO(bytes(stream, 'utf-8')) if isinstance(stream, str) else stream
        f = self.upload_file(stream, self.userid + '/' + path)
        return file_path(f, self.userid)

    def save_image(self, path, img):
        # stream = io.BytesIO()
        # mime = mimetypes.guess_type(path)[0]
        # format = mime.split('/')[1]
        # Image.fromarray(img).save(stream, format)
        # print(mime, format)
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        Image.fromarray(img).save(dest)
        return self.upload(dest, path)

    def capture_text(self, path):
        url = f"{API_URL}/rekognition/{self.userid}/{path}"
        resp = requests.get(url, params={'mode': 'text'}, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        text = list(filter(lambda t: t.get('ParentId') is None, resp.json().get('Text')));
        return [t.get('DetectedText') for t in text]

    def detect_labels(self, path):
        url = f"{API_URL}/rekognition/{self.userid}/{path}"
        resp = requests.get(url, params={'mode': 'labels'}, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        labels = resp.json().get('Labels')
        return [l.get('Name') for l in labels]
    