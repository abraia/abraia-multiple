import os
import json
import shutil
import posixpath

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
from .utils.concurrency import get_default_remote_request_scheduler
from .utils.cache import RemoteFileCache


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
        self.code = code
        detail = None
        if isinstance(message, str):
            detail = message
        elif isinstance(message, dict):
            detail = message.get('message') or message.get('error')
        else:
            try:
                payload = message.json()
            except (AttributeError, TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                detail = payload.get('message') or payload.get('error')
            if not detail:
                detail = getattr(message, 'text', None)
        self.message = str(detail or message)
        super(APIError, self).__init__(self.message)


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
        scheduler_override = kwargs.pop('_scheduler', None)
        kwargs.setdefault('timeout', self.request_timeout)
        return request_with_retries(
            self.session,
            method,
            url,
            retries=self.request_retries,
            scheduler=(
                scheduler_override
                if scheduler_override is not None
                else getattr(self, "_request_scheduler", None)
            ),
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
        dirname, basename = os.path.dirname(path), os.path.basename(path)
        if delegated is not _NO_DELEGATE:
            files, folders = delegated
        else:
            folder = dirname + '/' if dirname else dirname
            url = f"{API_URL}/files/{self.userid}/{folder}"
            resp = self._request('GET', url, auth=self.auth)
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            resp = resp.json()
            files = list(map(lambda f: {'path': file_path(f['source'], self.userid), 'name': f['name'], 'type': get_type(f['name']), 'size': f['size'], 'date': datetime.fromtimestamp(f['date'])}, resp['files']))
            folders = list(map(lambda f: {'path': file_path(f['source'], self.userid), 'name': f['name']}, resp['folders']))
        if not basename:
            RemoteFileCache(self.userid).reconcile_listing(dirname, files)
        if basename:
            files = list(filter(lambda f: fnmatch(f['path'], path), files))
            folders = list(filter(lambda f: fnmatch(f['path'], path), folders))
        return files, folders

    def upload_file(self, src, path='', remote_info=None):
        delegated = self._delegate('upload_file', src, path)
        if delegated is not _NO_DELEGATE:
            if not path or path.endswith('/'):
                remote_path = delegated
                if not remote_path and isinstance(src, str):
                    remote_path = posixpath.join(path, os.path.basename(src))
            else:
                remote_path = path
            self._store_uploaded_file(remote_path, src, remote_info)
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
        if remote_info is None:
            remote_info = resp.get('file')
        if remote_info and isinstance(remote_info.get('date'), (int, float)):
            remote_info = dict(remote_info)
            remote_info['date'] = datetime.fromtimestamp(remote_info['date'])
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
            result = file_path(f"{self.userid}/{path}", self.userid)
        else:
            result = file_path(resp['file']['source'], self.userid)
        self._store_uploaded_file(path, src, remote_info)
        return result

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
            self.invalidate_cached(old_path)
            self.invalidate_cached(new_path)
            return delegated
        json = {'store': f"{self.userid}/{old_path}"}
        url = f"{API_URL}/files/{self.userid}/{new_path}"
        resp = self._request('POST', url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        self.invalidate_cached(old_path)
        self.invalidate_cached(new_path)
        return file_path(resp['file']['source'], self.userid)

    def download_file(self, path, dest=None, remote_info=None):
        """Download through the shared cache, optionally copying to ``dest``."""
        cache = RemoteFileCache(self.userid)
        cached = cache.get_or_create(
            path,
            lambda destination: self._download_file_uncached(path, destination),
            remote_info=remote_info,
        )
        if dest is None or os.path.abspath(cached) == os.path.abspath(dest):
            return cached
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.copy2(cached, dest)
        return dest

    def _download_file_uncached(self, path, dest):
        """Fetch a remote file directly into a cache staging path."""
        client = getattr(self, '_client', None)
        uncached = getattr(client, '_download_file_uncached', None)
        if uncached is not None:
            return uncached(path, dest)
        delegated = self._delegate('download_file', path, dest)
        if delegated is not _NO_DELEGATE:
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        def transfer():
            resp = self._request(
                'GET', url, stream=True, auth=self.auth, _scheduler=False
            )
            if resp.status_code != 200:
                raise APIError(resp.text, resp.status_code)
            save_data(dest, resp.content)
            return dest

        scheduler = (
            getattr(self, "_request_scheduler", None)
            or get_default_remote_request_scheduler()
        )
        return scheduler.run(transfer)

    def invalidate_cached(self, path):
        """Remove a cached file and its dependent thumbnail."""
        return RemoteFileCache(self.userid).invalidate_source(path)

    def _store_uploaded_file(self, path, source, remote_info=None):
        """Cache uploaded local bytes without querying remote metadata."""
        if not path:
            return
        cache = RemoteFileCache(self.userid)
        if isinstance(source, str) and source.startswith('http'):
            self.invalidate_cached(path)
            return
        cache.invalidate_thumbnail(path)
        try:
            seeded = cache.store(path, source, remote_info)
            if seeded is None:
                seeded = cache.store(path, source)
            if seeded is None:
                self.invalidate_cached(path)
        except (OSError, TypeError, ValueError):
            self.invalidate_cached(path)

    def clear_cache(self):
        """Clear all remote files cached for this account."""
        return RemoteFileCache(self.userid).clear()

    def remove_file(self, path):
        delegated = self._delegate('remove_file', path)
        if delegated is not _NO_DELEGATE:
            self.invalidate_cached(path)
            return delegated
        url = f"{API_URL}/files/{self.userid}/{path}"
        resp = self._request('DELETE', url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError(resp.text, resp.status_code)
        resp = resp.json()
        self.invalidate_cached(path)
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

    def transform_image(self, path, dest, params=None):
        params = {'quality': 'auto'} if params is None else dict(params)
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

    def load_file(self, path):
        dest = self.download_file(path)
        try:
            with open(dest, 'r') as f:
                return f.read()
        except (OSError, UnicodeError):
            with open(dest, 'rb') as f:
                return BytesIO(f.read())

    def save_file(self, path, stream):
        stream = BytesIO(stream.encode('utf-8')) if isinstance(stream, str) else stream
        return self.upload_file(stream, path)

    def load_json(self, path):
        dest = self.download_file(path)
        with open(dest, 'r', encoding='utf-8') as source:
            return json.load(source)

    def save_json(self, path, values):
        return self.save_file(path, json.dumps(values))

    def load_image(self, path):
        dest = self.download_file(path)
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
        src = temporal_src(path)
        save_image(im, src)
        return self.upload_file(src, path)
