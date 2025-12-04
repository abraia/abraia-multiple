import os
import gzip
import json
import base64
import hashlib
import tempfile
import requests
import mimetypes
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor

from .video import Video
from .sketcher import Sketcher
from .draw import get_color, render_results

tempdir = tempfile.gettempdir()

API_URL = 'https://api.abraia.me'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

mimetypes.add_type('image/webp', '.webp')


def get_type(path):
    return mimetypes.guess_type(path)[0] or 'binary/octet-stream'


def md5sum(src):
    hash_md5 = hashlib.md5()
    f = BytesIO(src.getvalue()) if isinstance(src, BytesIO) else open(src, 'rb')
    for chunk in iter(lambda: f.read(4096), b''):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def is_url(url):
    return url.startswith('http://') or url.startswith('https://')


def url_path(path):
    return f"{API_URL}/files/{path}"


def list_dir(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def make_dirs(dest):
    """Create directory if it doesn't exist."""
    dirname = os.path.dirname(dest)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def temporal_src(path):
    dest = os.path.join(tempdir, path)
    make_dirs(dest)
    return dest


def download_url(url: str, dest: str, chunk_size: int = 8192):
    filename = os.path.basename(dest)
    temp_dest = Path(dest).with_name(filename + '.part')
    temp_dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=HEADERS, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(temp_dest, 'wb') as f, tqdm(desc=filename, total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size): 
                size = f.write(chunk)
                bar.update(size)
            f.flush()
    temp_dest.rename(dest)


def download_file(path):
    dest = temporal_src(path)
    if not os.path.exists(dest):
        download_url(url_path(path), dest)
    return dest


def load_url(url):
    r = requests.get(url, headers=HEADERS, stream=True, allow_redirects=True)
    return r.raw if r.status_code == 200 else None


def load_json(src, gz=False):
    with gzip.open(src, 'rt') if gz else open(src, 'r') as f:
        return json.load(f)
    

def save_json(data, dest, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wt') if gz else open(dest, 'w') as f:    
        f.write(json.dumps(data, cls=NumpyEncoder))
    return dest


def load_data(src, gz=False):
    with gzip.open(src, 'rb') if gz else open(src, 'rb') as f:
        return f.read()


def save_data(dest, data, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wb') if gz else open(dest, 'wb') as f:
        f.write(data)
    return dest


def load_image(src, mode='RGB'):
    # Fix image orientation based on its EXIF data
    im = ImageOps.exif_transpose(Image.open(src)) 
    return np.array(im.convert(mode))


def save_image(img, dest):
    make_dirs(dest)
    Image.fromarray(img).save(dest)
    return dest


def show_image(img):
    Image.fromarray(img).show()


def image_base64(img, format='jpeg'):
    im = Image.fromarray(img)
    with BytesIO() as buffer:
        im.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f'data:image/{format};base64,{encoded}'


def get_providers():
    available_providers = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in available_providers if provider in providers]


def process_map(task, *values, desc='', max_workers=3):
    with ProcessPoolExecutor(max_workers) as exe:
        with tqdm(total=len(values[0]), desc=desc) as pbar:
            for result in exe.map(task, *values):
                pbar.set_postfix_str(result)
                pbar.update(1)
