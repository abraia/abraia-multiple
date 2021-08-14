import os
import sys
import base64

CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.abraia')

MIME_TYPES = {'jpg': 'image/jpeg',
              'jpeg': 'image/jpeg',
              'png': 'image/png',
              'gif': 'image/gif',
              'svg': 'image/svg+xml',
              'webp': 'image/webp',
              'bmp': 'image/bmp',
              'pdf': 'application/pdf',
              'psd': 'image/vnd.adobe.photoshop'}


def base64encode(str):
    str = str.encode('utf-8') if sys.version_info[0] == 3 else str
    str = base64.b64encode(str)
    return str.decode('ascii') if isinstance(str, bytes) else str


def base64decode(str):
    str = base64.b64decode(str)
    return str.decode('ascii') if isinstance(str, bytes) else str


def load_auth():
    abraia_key = os.environ.get('ABRAIA_KEY')
    if abraia_key:
        api_key, api_secret = base64decode(abraia_key).split(':')
        return api_key, api_secret
    elif os.path.isfile(CONFIG_FILE):
        config = {}
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = list(map(lambda v: v.strip(), line.split(':')))
                config[key] = value
        return config['abraia_api_key'], config['abraia_api_secret']
    return '', ''


def save_auth(api_key, api_secret):
    content = ('abraia_api_key: {}\n'
               'abraia_api_secret: {}\n').format(api_key, api_secret)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def load_key():
    api_key, api_secret = load_auth()
    abraia_key = base64encode('{}:{}'.format(api_key, api_secret)) if api_key and api_secret else ''
    return abraia_key


def save_key(abraia_key):
    api_key, api_secret = base64decode(abraia_key).split(':')
    save_auth(api_key, api_secret)
