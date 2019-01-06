import os
import requests
from io import BytesIO

from . import config


class APIError(Exception):
    def __init__(self, message):
        super(APIError, self).__init__(message)
        self.message = message


class Client(object):
    def __init__(self):
        self.auth = config.load_auth()

    def check(self):
        files, folders = self.list_files()
        return folders[0]['name']

    def load_user(self):
        url = '{}/users'.format(config.API_URL)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp.json()

    def list_files(self, path=''):
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.get(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        resp = resp.json()
        return resp['files'], resp['folders']

    def upload_remote(self, url, path):
        json = {'url': url}
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(url, resp.status_code))
        resp = resp.json()
        return resp['file']

    def upload_file(self, file, path, type=''):
        source = path + os.path.basename(file) if path.endswith('/') else path
        name = os.path.basename(source)
        url = '{}/files/{}'.format(config.API_URL, source)
        json = {'name': name, 'type': type}
        resp = requests.post(url, json=json, auth=self.auth)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(url, resp.status_code))
        url = resp.json()['uploadURL']
        data = file if isinstance(file, BytesIO) else open(file, 'rb')
        resp = requests.put(url, data=data, headers={'Content-Type': type})
        if resp.status_code != 200:
            raise APIError('PUT {} {}'.format(url, resp.status_code))
        return {'name': name, 'source': source}

    def move_file(self, old_path, new_path):
        url = '{}/files/{}'.format(config.API_URL, new_path)
        resp = requests.post(url, json={'store': old_path}, auth=self.auth)
        if resp.status_code != 201:
            raise APIError('POST {} {}'.format(url, resp.status_code))
        resp = resp.json()
        return resp['file']

    def download_file(self, path):
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.get(url, stream=True)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp

    def remove_file(self, path):
        url = '{}/files/{}'.format(config.API_URL, path)
        resp = requests.delete(url, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('DELETE {} {}'.format(url, resp.status_code))
        resp = resp.json()
        return resp['file']

    def transform_image(self, path, params=''):
        url = '{}/images/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, stream=True)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp

    def analyze_image(self, path, params=''):
        url = '{}/analysis/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp.json()

    def aesthetics_image(self, path, params=''):
        url = '{}/aesthetics/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp.json()

    def process_video(self, path, params=''):
        url = '{}/videos/{}'.format(config.API_URL, path)
        resp = requests.get(url, params=params, auth=self.auth)
        if resp.status_code != 200:
            raise APIError('GET {} {}'.format(url, resp.status_code))
        return resp.json()
