import os
import json
import tempfile
import requests

from tqdm import tqdm
from PIL import Image

tempdir = tempfile.gettempdir()


API_URL = 'https://api.abraia.me'


def download(url, dest, chunk_size=1024):
    filename = os.path.basename(dest)
    resp = requests.get(url, stream=True, allow_redirects=True)
    total = int(resp.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(desc=filename, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def url_path(path):
    return f"{API_URL}/files/{path}"


def temporal_src(path):
    dest = os.path.join(tempdir, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def download_file(path):
    url = url_path(path)
    dest = temporal_src(os.path.basename(url))
    if not os.path.exists(dest):
        # r = requests.get(url, allow_redirects=True)
        # open(dest, 'wb').write(r.content)
        download(url, dest)
    return dest


def load_json(src):
    with open(src, 'r') as file:
        return json.load(file)


def load_image(src):
    return Image.open(src).convert('RGB')


def get_color(idx):
    colors = ['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321',
    '#417505', '#BD10E0', '#9013FE', '#4A90E2', '#50E3C2', '#B8E986',
    '#000000', '#545454', '#737373', '#A6A6A6', '#D9D9D9', '#FFFFFF']
    return colors[idx % (len(colors) - 1)]


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
