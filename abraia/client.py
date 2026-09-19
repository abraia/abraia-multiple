import os
import json

from io import BytesIO
from fnmatch import fnmatch
from datetime import datetime

from . import config
from .utils.remote import (
    API_URL,
    HEADERS,
    create_session,
    request_with_retries,
    temporal_src,
)


_NO_DELEGATE = object()


def md5sum(*args, **kwargs):
    from .utils.filesystem import md5sum as _md5sum

    return _md5sum(*args, **kwargs)


def get_type(*args, **kwargs):
    from .utils.filesystem import get_type as _get_type

    return _get_type(*args, **kwargs)


def save_data(*args, **kwargs):
    from .utils.filesystem import save_data as _save_data

    return _save_data(*args, **kwargs)


def load_image(*args, **kwargs):
    from .utils.image import load_image as _load_image

    return _load_image(*args, **kwargs)


def save_image(*args, **kwargs):
    from .utils.image import save_image as _save_image

    return _save_image(*args, **kwargs)


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

    def __init__(self, abraia_key=None, client=None):
        """Create an API client, optionally backed by another client.

        The injected-client form keeps wrappers and tests on the same public
        method surface without copying session internals into subclasses.
        """
        if client is not None:
            self._client = client
            self.client = client
            self.auth = getattr(client, 'auth', None)
            self.userid = client.userid
            self.api_key = getattr(client, 'api_key', None)
            self.session = getattr(client, 'session', None)
            return

        self._client = None
        if abraia_key is None:
            abraia_id, abraia_key = config.load()
        else:
            abraia_id, _api_secret = config.load_auth(abraia_key)
        self.auth = config.load_auth(abraia_key)
        self.userid = abraia_id
        self.api_key = abraia_key
        self.session = create_session(HEADERS)

    def _delegate(self, method, *args, **kwargs):
        client = getattr(self, '_client', None)
        if client is None:
            return _NO_DELEGATE
        return getattr(client, method)(*args, **kwargs)

    def _request(self, method, url, **kwargs):
        """Perform a resilient API request, including retryable resets."""
        kwargs.setdefault('timeout', self.request_timeout)
        return request_with_retries(
            self.session,
            method,
            url,
            retries=self.request_retries,
            **kwargs,
        )

    def get_api(self, url, params):
        delegated = self._delegate('get_api', url, params)
        if delegated is not _NO_DELEGATE:
            return delegated
        resp = self._request('GET', url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def list_files(self, path=''):
        delegated = self._delegate('list_files', path)
        if delegated is not _NO_DELEGATE:
            return delegated
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
        delegated = self._delegate('upload_file', src, path)
        if delegated is not _NO_DELEGATE:
            return delegated
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
        delegated = self._delegate('check_file', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('HEAD', url, auth=self.auth)
        if resp.status_code == 404:
            return False
        if resp.status_code in [307, 400, 403]:
            return True
        raise APIError(resp.text, resp.status_code)

    def move_file(self, old_path, new_path):
        delegated = self._delegate('move_file', old_path, new_path)
        if delegated is not _NO_DELEGATE:
            return delegated
        json = {'store': f"{self.userid}/{old_path}"}
        url = f"{API_URL}/files/{self.userid}/{new_path}"
        resp = self._request('POST', url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

    def download_file(self, path, dest, cache=False):
        delegated = self._delegate('download_file', path, dest, cache=cache)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        if cache and os.path.exists(dest):
            return dest
        resp = self._request('GET', url, stream=True, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        save_data(dest, resp.content)
        return dest

    def _download_cached(self, path):
        """Download a remote file into the shared process cache."""
        destination = self._cached_destination(path)
        return self.download_file(path, destination, cache=True)

    @staticmethod
    def _cached_destination(path):
        """Return the shared process-cache destination for a remote path."""
        return temporal_src(str(path))
    
    def remove_file(self, path):
        delegated = self._delegate('remove_file', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('DELETE', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        return file_path(resp['file']['source'], self.userid)

    def load_metadata(self, path):
        delegated = self._delegate('load_metadata', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = self._request('GET', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def remove_metadata(self, path):
        delegated = self._delegate('remove_metadata', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/metadata/{self.userid}/{path}"
        resp = self._request('DELETE', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def transform_image(self, path, dest, params={'quality': 'auto'}):
        delegated = self._delegate('transform_image', path, dest, params=params)
        if delegated is not _NO_DELEGATE:
            return delegated
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
        delegated = self._delegate('load_file', path, cache=cache)
        if delegated is not _NO_DELEGATE:
            return delegated
        dest = temporal_src(path)
        self.download_file(path, dest, cache=cache)
        try:
            with open(dest, 'r') as f:
                return f.read()
        except:
            with open(dest, 'rb') as f:
                return BytesIO(f.read())

    def save_file(self, path, stream):
        delegated = self._delegate('save_file', path, stream)
        if delegated is not _NO_DELEGATE:
            return delegated
        stream =  BytesIO(bytes(stream, 'utf-8')) if isinstance(stream, str) else stream
        return self.upload_file(stream, path)

    def load_json(self, path):
        delegated = self._delegate('load_json', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('GET', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        return resp.json()

    def save_json(self, path, values):
        delegated = self._delegate('save_json', path, values)
        if delegated is not _NO_DELEGATE:
            return delegated
        return self.save_file(path, json.dumps(values))

    def load_image(self, path):
        delegated = self._delegate('load_image', path)
        if delegated is not _NO_DELEGATE:
            return delegated
        dest = self._download_cached(path)
        return load_image(dest)

    def load_image_details(self, path):
        """Load a standard remote image and its metadata."""
        image = self.load_image(path)
        return image, self.load_metadata(path)

    def load_image_preview(self, path, size=144, bands=(0, 1, 2)):
        """Return a generic preview through the shared source contract.

        Standard Abraia images do not have spectral band selection, so the
        size and bands hints are accepted for API compatibility and the normal
        image loader supplies the preview-sized representation.
        """
        del size, bands
        return self.load_image_details(path)

    def save_image(self, path, im):
        delegated = self._delegate('save_image', path, im)
        if delegated is not _NO_DELEGATE:
            return delegated
        src = temporal_src(path)
        save_image(im, src)
        return self.upload_file(src, path)
