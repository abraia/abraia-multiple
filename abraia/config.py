import os

api_url = 'https://abraia.me/api'
config_file = os.path.join(os.path.expanduser('~'), '.abraia')


def load_auth():
    api_key = os.environ.get('ABRAIA_API_KEY')
    api_secret = os.environ.get('ABRAIA_API_SECRET')
    config = {'abraia_api_key': api_key, 'abraia_api_secret': api_secret}
    if os.path.isfile(config_file) and (api_key is None or api_secret is None):
        with open(config_file, 'r') as f:
            for line in f:
                key, value = list(map(lambda v: v.strip(), line.split(':')))
                config[key] = value
    return config['abraia_api_key'], config['abraia_api_secret']


def save_auth(api_key, api_secret):
    content = ('abraia_api_key: {}\n'
               'abraia_api_secret: {}\n').format(api_key, api_secret)
    with open(config_file, 'w') as f:
        f.write(content)
