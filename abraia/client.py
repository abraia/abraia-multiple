import os
import json
import hashlib
import requests

from PIL import Image
from io import BytesIO
from fnmatch import fnmatch
from datetime import datetime

from . import config
from .utils import API_URL, get_type, temporal_src, save_data


def file_path(source, userid):
    return source[len(userid)+1:]


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
    def __init__(self):
        abraia_id, abraia_key = config.load()
        self.auth = config.load_auth(abraia_key)
        self.userid = abraia_id

    def get_api(self, url, params):
        resp = requests.get(url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def list_files(self, path=''):
        dirname, basename = os.path.dirname(path), os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        url = f"{API_URL}/files/{self.userid}/{folder}"
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        files = list(map(lambda f: {'path': file_path(f['source'], self.userid), 'name': f['name'], 'type': get_type(f['name']), 'size': f['size'], 'date': datetime.fromtimestamp(f['date'])}, resp['files']))
        folders = list(map(lambda f: {'path': file_path(f['source'], self.userid), 'name': f['name']}, resp['folders']))
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
            folders = list(filter(lambda f: fnmatch(f['path'], path), folders))
        return files, folders

    def upload_file(self, src, path=''):
        if path == '' or path.endswith('/'):
            path = path + os.path.basename(src)
        name, type = os.path.basename(path), get_type(path)
        json = {'url': src} if isinstance(src, str) and src.startswith('http') else {'name': name, 'type': type, 'md5': md5sum(src)}
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
            return file_path(f"{self.userid}/{path}", self.userid)
        return file_path(resp['file']['source'], self.userid)

    def check_file(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.head(url, auth=self.auth)
        if resp.status_code == 404:
            return False
        if resp.status_code in [307, 400, 403]:
            return True
        raise APIError(resp.text, resp.status_code)

    def move_file(self, old_path, new_path):
        json = {'store': f"{self.userid}/{old_path}"}
        url = f"{API_URL}/files/{self.userid}/{new_path}"
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

    def download_file(self, path, dest='', cache=False):
        # TODO: Remove BytesIO download option
        url = f"{API_URL}/files/{self.userid}/{path}"
        if cache and dest == '':
            dest = temporal_src(path)
            if os.path.exists(dest):
                return dest
        resp = requests.get(url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        if dest:
            save_data(dest, resp.content)
            return dest
        return BytesIO(resp.content)
    
    def remove_file(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

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
        save_data(dest, resp.content)

    def load_file(self, path):
        stream = self.download_file(path)
        try:
            return stream.getvalue().decode('utf-8')
        except:
            return stream

    def save_file(self, path, stream):
        stream =  BytesIO(bytes(stream, 'utf-8')) if isinstance(stream, str) else stream
        return self.upload_file(stream, path)

    def load_json(self, path):
        return json.loads(self.load_file(path))

    def save_json(self, path, values):
        return self.save_file(path, json.dumps(values))

    def load_image(self, path):
        return Image.open(self.download_file(path))

    def save_image(self, path, im):
        src = temporal_src(path)
        im.save(src)
        return self.upload_file(src, path)
