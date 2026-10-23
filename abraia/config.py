import os
import base64

CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.abraia')

def base64encode(value):
    return base64.b64encode(value.encode('utf-8')).decode('ascii')


def base64decode(value):
    return base64.b64decode(value).decode('ascii')


def load():
    abraia_id = os.environ.get('ABRAIA_ID')
    abraia_key = os.environ.get('ABRAIA_KEY')
    if abraia_key:
        return _resolve_credentials(abraia_id, abraia_key)
    elif os.path.isfile(CONFIG_FILE):
        config = {}
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()
        return _resolve_credentials(
            config.get('abraia_id', ''),
            config.get('abraia_key', ''),
        )
    return '', ''


def _resolve_credentials(abraia_id, abraia_key):
    """Resolve the user ID from an encoded key when it is not supplied."""
    if abraia_id or not abraia_key:
        return abraia_id or '', abraia_key or ''
    api_key, _api_secret = load_auth(abraia_key)
    return api_key, abraia_key


def load_auth(abraia_key):
    if abraia_key:
        api_key, api_secret = base64decode(abraia_key).split(':')
        return api_key, api_secret
    return '', ''


def save(abraia_key):
    """Save the API key used to authenticate the client.

    The user ID is encoded in the API key and is resolved by :func:`load`, so
    it does not need to be configured separately.
    """
    content = 'abraia_key: {}\n'.format(abraia_key)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)
