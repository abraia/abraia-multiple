import os
import gzip
import json
import base64
import hashlib
import tempfile
import requests
import mimetypes
import numpy as np
import pillow_heif
import time
from requests.adapters import HTTPAdapter

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor

from .video import Video
from .stream import VideoInput, VideoDisplay
from .display import Sketcher, Window
from .draw import get_color, render_results

pillow_heif.register_heif_opener()

tempdir = tempfile.gettempdir()

API_URL = 'https://api.abraia.me'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

_url_session = requests.Session()
_url_session.headers.update(HEADERS)
_url_adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
_url_session.mount('https://', _url_adapter)
_url_session.mount('http://', _url_adapter)

mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/heic', '.heic')
mimetypes.add_type('image/heif', '.heif')


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


def get_remote_file_size(url: str, timeout: int = 30):
    """Get the size of a remote file via HEAD request."""
    try:
        r = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        length = r.headers.get('Content-Length')
        return int(length) if length else None
    except Exception:
        return None


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


def load_url(url, timeout=(10, 120)):
    for attempt in range(3):
        try:
            r = _url_session.get(url, stream=True, allow_redirects=True, timeout=timeout)
            if r.status_code == 200:
                return r.raw
            r.close()
            return None
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                return None
            time.sleep(0.25 * (attempt + 1))
    return None


def load_url_bytes(url, timeout=(10, 120)):
    """Load a remote resource and close its connection after reading it."""
    for attempt in range(3):
        response = load_url(url, timeout=timeout)
        if response is None:
            return None
        try:
            return response.read()
        except (OSError, requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                return None
            time.sleep(0.25 * (attempt + 1))
        finally:
            response.close()


def load_json(src, gz=False):
    with gzip.open(src, 'rt') if gz else open(src, 'r') as f:
        return json.load(f)
    

def save_json(data, dest, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wt') if gz else open(dest, 'w') as f:    
        f.write(json.dumps(data, cls=NumpyEncoder))
    return dest


def load_text(src, gz=False):
    with gzip.open(src, 'rt') if gz else open(src, 'r') as f:
        return f.read()


def save_text(dest, text, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wt') if gz else open(dest, 'w') as f:
        f.write(text)
    return dest


def load_data(src, gz=False):
    with gzip.open(src, 'rb') if gz else open(src, 'rb') as f:
        return f.read()


def save_data(dest, data, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wb') if gz else open(dest, 'wb') as f:
        f.write(data)
    return dest


def load_image(src, mode='RGB', max_size=2048):
    im = ImageOps.exif_transpose(Image.open(src).convert(mode)) 
    if im.width > max_size or im.height > max_size:
        im.thumbnail((max_size, max_size), Image.LANCZOS)
    return np.array(im)


def encode_image(image, format='PNG', mode=None):
    """Encode a NumPy image array and return its encoded bytes."""
    pil_image = Image.fromarray(np.asarray(image))
    if mode is not None:
        pil_image = pil_image.convert(mode)
    with BytesIO() as buffer:
        pil_image.save(buffer, format=format)
        return buffer.getvalue()


def as_array(value, dtype=None, copy=False):
    """Convert an array-like value using the SDK's NumPy dependency."""
    array = np.asarray(value, dtype=dtype)
    return array.copy() if copy else array


def array_from_image_buffer(buffer, width, height, stride, channels=3):
    """Decode a packed uint8 image buffer into an owned HWC array."""
    rows = np.frombuffer(buffer, dtype=np.uint8).reshape(int(height), int(stride))
    return rows[:, :int(width) * int(channels)].reshape(
        int(height), int(width), int(channels)
    ).copy()


def array_shape(value):
    """Return the shape of an array-like value."""
    return np.asarray(value).shape


def array_ndim(value):
    """Return the number of dimensions of an array-like value."""
    return np.asarray(value).ndim


def array_size(value):
    """Return the number of elements in an array-like value."""
    return np.asarray(value).size


def array_copy(value):
    """Return an owned NumPy copy of an array-like value."""
    return np.asarray(value).copy()


def array_squeeze(value):
    """Remove singleton dimensions from an array-like value."""
    return np.squeeze(value)


def array_to_list(value):
    """Convert a NumPy value to a regular Python list when applicable."""
    return value.tolist() if hasattr(value, 'tolist') else value


def zeros_array(shape, dtype='uint8'):
    """Return a zero-filled NumPy array."""
    return np.zeros(shape, dtype=dtype)


def mask_array(value):
    """Return an array-like value as a boolean mask."""
    return np.asarray(value) > 0


def merge_masks(target, mask):
    """Merge a boolean mask into an existing mask in place."""
    np.logical_or(target, mask, out=target)
    return target


def encode_mask_overlay(mask, color=(32, 184, 121), alpha=96):
    """Encode a colored RGBA mask overlay as PNG bytes."""
    mask = mask_array(mask)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = np.where(mask, int(alpha), 0).astype(np.uint8)
    return encode_image(rgba, format='PNG')


def compose_mask_layers(layers, shape, alpha=96):
    """Compose colored mask layers, returning ``(mask, PNG overlay bytes)``."""
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    combined = np.zeros(shape, dtype=bool)
    for mask, color in layers or []:
        mask = mask_array(mask)
        if mask.shape != tuple(shape):
            continue
        np.logical_or(combined, mask, out=combined)
        rgba[mask, :3] = color
        rgba[mask, 3] = int(alpha)
    return combined, encode_image(rgba, format='PNG')


def resize_mask(mask, size):
    """Resize a binary mask to ``(width, height)`` with nearest-neighbor sampling."""
    mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    resized = mask_image.resize(
        (int(size[0]), int(size[1])), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


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
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in available_providers if provider in providers]


def process_map(task, *values, desc='', max_workers=3):
    with ProcessPoolExecutor(max_workers) as exe:
        with tqdm(total=len(values[0]), desc=desc) as pbar:
            for result in exe.map(task, *values):
                pbar.set_postfix_str(result)
                pbar.update(1)


def process_media(src, callback):
    if get_type(str(src)).startswith('image'):
        img = load_image(src)
        out = callback(img)
        show_image(out)
    else:
        video = Video(src)
        for frame in video:
            out = callback(frame)
            video.show(out)
