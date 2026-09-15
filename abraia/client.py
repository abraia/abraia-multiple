import os
import json
import requests
import time
from requests.adapters import HTTPAdapter

from PIL import Image
from io import BytesIO
from fnmatch import fnmatch
from datetime import datetime

from . import config
from .utils import API_URL, HEADERS, md5sum, get_type, temporal_src, save_data, load_image, save_image


def file_path(source, userid):
    return source[len(userid)+1:]


class APIError(Exception):
    def __init__(self, message, code=0):
        super(APIError, self).__init__(message, code)
        self.code = code
        try:
            self.message = message.json()['message']
        except:
            self.message = ''


class Abraia:
    request_timeout = (10, 120)
    request_retries = 3

    def __init__(self):
        abraia_id, abraia_key = config.load()
        self.auth = config.load_auth(abraia_key)
        self.userid = abraia_id
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def _request(self, method, url, **kwargs):
        """Perform a resilient API request, including retryable resets."""
        kwargs.setdefault('timeout', self.request_timeout)
        data = kwargs.get('data')
        position = data.tell() if hasattr(data, 'tell') else None
        last_error = None
        for attempt in range(self.request_retries):
            if position is not None and hasattr(data, 'seek'):
                data.seek(position)
            try:
                return self.session.request(method, url, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = error
                if attempt + 1 == self.request_retries:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        raise last_error

    def get_api(self, url, params):
        resp = self._request('GET', url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def list_files(self, path=''):
        dirname, basename = os.path.dirname(path), os.path.basename(path)
        folder = dirname + '/' if dirname else dirname
        url = f"{API_URL}/files/{self.userid}/{folder}"
        resp = self._request('GET', url, auth=self.auth)
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
        resp = self._request('POST', url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        url = resp.get('uploadURL')
        if url:
            data = src if isinstance(src, BytesIO) else open(src, 'rb')
            try:
                resp = self._request('PUT', url, data=data, headers={'Content-Type': type})
            finally:
                if data is not src:
                    data.close()
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            return file_path(f"{self.userid}/{path}", self.userid)
        return file_path(resp['file']['source'], self.userid)

    def check_file(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('HEAD', url, auth=self.auth)
        if resp.status_code == 404:
            return False
        if resp.status_code in [307, 400, 403]:
            return True
        raise APIError(resp.text, resp.status_code)

    def move_file(self, old_path, new_path):
        json = {'store': f"{self.userid}/{old_path}"}
        url = f"{API_URL}/files/{self.userid}/{new_path}"
        resp = self._request('POST', url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

    def download_file(self, path, dest, cache=False):
        url = f"{API_URL}/files/{self.userid}/{path}"
        if cache and os.path.exists(dest):
            return dest
        resp = self._request('GET', url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        save_data(dest, resp.content)
        return dest
    
    def remove_file(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('DELETE', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

    def load_metadata(self, path):
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = self._request('GET', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def remove_metadata(self, path):
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = self._request('DELETE', url, auth=self.auth)
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
        resp = self._request('GET', url, params=params, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        save_data(dest, resp.content)

    def load_file(self, path, cache=False):
        dest = temporal_src(path)
        self.download_file(path, dest, cache=cache)
        try:
            with open(dest, 'r') as f:
                return f.read()
        except:
            with open(dest, 'rb') as f:
                return BytesIO(f.read())

    def save_file(self, path, stream):
        stream =  BytesIO(bytes(stream, 'utf-8')) if isinstance(stream, str) else stream
        return self.upload_file(stream, path)

    def load_json(self, path):
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('GET', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def save_json(self, path, values):
        return self.save_file(path, json.dumps(values))

    def load_image(self, path):
        dest = temporal_src(path)
        self.download_file(path, dest, cache=True)
        return load_image(dest)

    def save_image(self, path, im):
        src = temporal_src(path)
        save_image(im, src)
        return self.upload_file(src, path)
