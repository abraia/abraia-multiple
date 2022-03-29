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


def load():
    abraia_id = os.environ.get('ABRAIA_ID')
    abraia_key = os.environ.get('ABRAIA_KEY')
    if abraia_id and abraia_key:
        return abraia_id, abraia_key
    elif os.path.isfile(CONFIG_FILE):
        config = {}
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = list(map(lambda v: v.strip(), line.split(':')))
                config[key] = value
        return config.get('abraia_id', ''), config.get('abraia_key', '')
    return '', ''


def load_auth(abraia_key):
    if abraia_key:
        api_key, api_secret = base64decode(abraia_key).split(':')
        return api_key, api_secret
    return '', ''


def save(abraia_id, abraia_key):
    content = ('abraia_id: {}\nabraia_key: {}\n').format(abraia_id, abraia_key)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
