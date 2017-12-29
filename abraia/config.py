import os
import yaml

api_url = 'https://abraia.me/api'
config_file = os.path.join(os.path.expanduser('~'), '.abraia')


def load_auth():
    api_key = os.environ.get('ABRAIA_API_KEY')
    api_secret = os.environ.get('ABRAIA_API_SECRET')
    config = {'abraia_api_key': api_key, 'abraia_api_secret': api_secret}
    if os.path.isfile(config_file) and api_key is None or api_secret is None:
        with open(config_file, 'r') as f:
            config = yaml.load(f)
    return config['abraia_api_key'], config['abraia_api_secret']


def save_auth(api_key, api_secret):
    config = {'abraia_api_key': api_key, 'abraia_api_secret': api_secret}
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
