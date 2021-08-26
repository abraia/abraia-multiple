import os
import hashlib
import requests
import tempfile
import mimetypes
import numpy as np

from PIL import Image
from io import BytesIO
from fnmatch import fnmatch
from datetime import datetime

from . import config


API_URL = 'https://api.abraia.me'
tempdir = tempfile.gettempdir()


def file_path(f, userid):
    return f['source'][len(userid)+1:]


def md5sum(src):
    hash_md5 = hashlib.md5()
    f = BytesIO(src.getvalue()) if isinstance(src, BytesIO) else open(src, 'rb')
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
            url = f"{API_URL}/users"
            resp = requests.get(url, auth=self.auth)
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            return resp.json()['user']
        return {}

    def list_files(self, path=''):
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        url = f"{API_URL}/files/{self.userid}/{folder}"
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        for f in resp['files']:
            f['date'] = datetime.fromtimestamp(f['date'])
        files, folders = resp['files'], resp['folders']
        files = list(map(lambda f: {'path': file_path(f, self.userid), 'name': f['name'], 'size': f['size'], 'date': f['date']}, files))
        folders = list(map(lambda f: {'path': file_path(f, self.userid), 'name': f['name']}, folders))
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
            folders = list(filter(lambda f: fnmatch(f['path'], path), folders))
        return files, folders

    def upload_file(self, src, path=''):
        if path == '' or path.endswith('/'):
            path = path + os.path.basename(src)
        json = {}
        name = os.path.basename(path)
        type = mimetypes.guess_type(name)[0] or 'binary/octet-stream'
        if isinstance(src, str) and src.startswith('http'):
            json = {'url': src}
        else:
            json = {'name': name, 'type': type}
            md5 = md5sum(src)
            if md5:
                json['md5'] = md5
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        url = resp.get('uploadURL')
        if url:
            data = src if isinstance(src, BytesIO) else open(src, 'rb')
            resp = requests.put(url, data=data, headers={'Content-Type': type})
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            return file_path({'name': name, 'source': f"{self.userid}/{path}"}, self.userid)
        return file_path(resp['file'], self.userid)

    def move_file(self, old_path, new_path):
        json = {'store': f"{self.userid}/{old_path}"}
        url = f"{API_URL}/files/{self.userid}/{new_path}"
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file'], self.userid)

    def download_file(self, path, dest=''):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.get(url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        if not dest:
            return BytesIO(resp.content)
        with open(dest, 'wb') as f:
            f.write(resp.content)

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
            params['background'] = f"{API_URL}/images/{self.userid}/{path}"
            if params.get('fmt') is None:
                params['fmt'] = params['background'].split('.').pop()
            path = f"{self.userid}/{params['action']}"
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
        stream =  BytesIO(bytes(stream, 'utf-8')) if isinstance(stream, str) else stream
        return self.upload_file(stream, path)

    def save_image(self, path, img):
        # stream = BytesIO()
        # mime = mimetypes.guess_type(path)[0]
        # format = mime.split('/')[1]
        # Image.fromarray(img).save(stream, format)
        # print(mime, format)
        basename = os.path.basename(path)
        src = os.path.join(tempdir, basename)
        Image.fromarray(img).save(src)
        return self.upload_file(src, path)

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
    