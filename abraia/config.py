import os

API_URL = 'https://api.abraia.me'
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.abraia')

MIME_TYPES = {'jpg': 'image/jpeg',
              'jpeg': 'image/jpeg',
              'png': 'image/png',
              'gif': 'image/gif',
              'webp': 'image/webp',
              'bmp': 'image/bmp',
              'pdf': 'application/pdf'}


def load_auth():
    api_key = os.environ.get('ABRAIA_API_KEY')
    api_secret = os.environ.get('ABRAIA_API_SECRET')
    config = {'abraia_api_key': api_key, 'abraia_api_secret': api_secret}
    if os.path.isfile(CONFIG_FILE) and (api_key is None or api_secret is None):
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = list(map(lambda v: v.strip(), line.split(':')))
                config[key] = value
    return config['abraia_api_key'], config['abraia_api_secret']


def save_auth(api_key, api_secret):
    content = ('abraia_api_key: {}\n'
               'abraia_api_secret: {}\n').format(api_key, api_secret)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
