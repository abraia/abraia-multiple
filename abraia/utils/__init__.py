import os
import json
import base64
import tempfile
import requests
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps

from .video import Video
from .sketcher import Sketcher
from .draw import render_results

tempdir = tempfile.gettempdir()


API_URL = 'https://api.abraia.me'


def is_url(url):
    return url.startswith('http://') or url.startswith('https://')


def url_path(path):
    return f"{API_URL}/files/{path}"


def temporal_src(path):
    dest = os.path.join(tempdir, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def download_url(url: str, dest: str, chunk_size: int = 8192):
    filename = os.path.basename(dest)
    temp_dest = Path(dest).with_name(filename + '.part')
    temp_dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(temp_dest, 'wb') as f, tqdm(desc=filename, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size): 
                size = f.write(chunk)
                bar.update(size)
            f.flush()
    temp_dest.rename(dest)


# def download_url(url, dest, chunk_size=4096):
#     filename = os.path.basename(dest)
#     resp = requests.get(url, stream=True, allow_redirects=True)
#     total = int(resp.headers.get('content-length', 0))
#     with open(dest, 'wb') as f, tqdm(desc=filename, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
#         for chunk in resp.iter_content(chunk_size=chunk_size):
#             size = f.write(chunk)
#             bar.update(size)
#         f.flush()


def download_file(path):
    dest = temporal_src(path)
    if not os.path.exists(dest):
        # r = requests.get(url, allow_redirects=True)
        # open(dest, 'wb').write(r.content)
        download_url(url_path(path), dest)
    return dest


def load_url(url):
    return requests.get(url, stream=True).raw


def load_json(src):
    with open(src, 'r') as file:
        return json.load(file)


def load_image(src, mode='RGB'):
    # Fix image orientation based on its EXIF data
    im = ImageOps.exif_transpose(Image.open(src)) 
    return np.array(im.convert(mode))


def save_image(img, dest):
    dirname = os.path.dirname(dest)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    Image.fromarray(img).save(dest)


def show_image(img):
    Image.fromarray(img).show()


def image_base64(img, format='jpeg'):
    im = Image.fromarray(img)
    with BytesIO() as buffer:
        im.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/{format};base64,{encoded}'


def get_color(idx):
    colors = ['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321',
              '#417505', '#BD10E0', '#9013FE', '#4A90E2', '#50E3C2', '#B8E986',
              '#000000', '#545454', '#737373', '#A6A6A6', '#D9D9D9', '#FFFFFF']
    return colors[idx % (len(colors) - 1)]


def get_providers():
    available_providers = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in available_providers if provider in providers]
