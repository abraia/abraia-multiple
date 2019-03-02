import os
import sys
import base64

API_URL = 'https://api.abraia.me'
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.abraia')

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.psd']
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
    if os.path.isfile(CONFIG_FILE) and (abraia_key is None):
        config = {}
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = list(map(lambda v: v.strip(), line.split(':')))
                config[key] = value
        return config['abraia_api_key'], config['abraia_api_secret']
    elif abraia_key:
        api_key, api_secret = base64decode(abraia_key).split(':')
        return api_key, api_secret
    return '', ''


def save_auth(api_key, api_secret):
    content = ('abraia_api_key: {}\n'
               'abraia_api_secret: {}\n').format(api_key, api_secret)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
