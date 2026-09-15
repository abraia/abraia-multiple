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
import threading
from urllib.parse import quote
from requests.adapters import HTTPAdapter

from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ProcessPoolExecutor

from .video import Video, make_dirs
from .stream import VideoInput, VideoDisplay
from .display import Sketcher, Window
from .draw import get_color, render_results
from .pipeline import (
    FrameContext,
    LineCounterStage,
    Pipeline,
    RegionFilterStage,
    RegionTimerStage,
    TrackerStage,
)

pillow_heif.register_heif_opener()

tempdir = tempfile.gettempdir()

API_URL = 'https://api.abraia.me'

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

_url_sessions = threading.local()


def _get_url_session():
    """Return a connection-pooled session local to the current thread."""
    session = getattr(_url_sessions, 'session', None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        _url_sessions.session = session
    return session

mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/heic', '.heic')
mimetypes.add_type('image/heif', '.heif')


def get_type(path):
    return mimetypes.guess_type(path)[0] or 'binary/octet-stream'


def md5sum(src):
    hash_md5 = hashlib.md5()
    if isinstance(src, BytesIO):
        f = BytesIO(src.getvalue())
    else:
        f = open(src, 'rb')
    with f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def is_url(url):
    return url.startswith('http://') or url.startswith('https://')


def url_path(path):
    path = os.fspath(path).replace('\\', '/')
    return f"{API_URL}/files/{quote(path.lstrip('/'), safe='/')}"


def list_dir(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def get_remote_file_size(url: str, timeout: int = 30):
    """Get the size of a remote file via HEAD request."""
    try:
        with _get_url_session().head(
            url, timeout=timeout, allow_redirects=True
        ) as response:
            if not response.ok:
                return None
            length = response.headers.get('Content-Length')
            return int(length) if length else None
    except Exception:
        return None


def temporal_src(path):
    relative = Path(os.fspath(path))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Temporary paths must remain relative to the cache directory')
    dest = Path(tempdir) / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    return str(dest)


def download_url(
    url: str,
    dest: str,
    chunk_size: int = 8192,
    timeout=(10, 120),
):
    """Download a URL atomically, removing partial output on failure."""
    destination = Path(dest)
    filename = destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f'.{filename}.', suffix='.part', dir=destination.parent
    )
    os.close(temp_fd)
    temp_dest = Path(temp_name)
    try:
        with _get_url_session().get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with temp_dest.open('wb') as fileobj, tqdm(
                desc=filename,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    size = fileobj.write(chunk)
                    bar.update(size)
                fileobj.flush()
                os.fsync(fileobj.fileno())
        os.replace(temp_dest, destination)
    except Exception:
        temp_dest.unlink(missing_ok=True)
        raise


def download_file(path):
    dest = temporal_src(path)
    if not os.path.exists(dest):
        download_url(url_path(path), dest)
    return dest


def load_url(url, timeout=(10, 120)):
    """Return a readable response body for a URL.

    The returned raw stream owns the underlying connection; callers must
    close it after consuming the response. Prefer :func:`load_url_bytes` when
    a stream is not required.
    """
    for attempt in range(3):
        try:
            r = _get_url_session().get(
                url, stream=True, allow_redirects=True, timeout=timeout
            )
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
    with gzip.open(src, 'rt', encoding='utf-8') if gz else open(
        src, 'r', encoding='utf-8'
    ) as f:
        return json.load(f)
    

_MISSING = object()


def save_json(dest, data=_MISSING, gz=False):
    """Save JSON using the same destination-first order as other save APIs.

    The historical ``save_json(data, dest)`` order remains accepted for
    compatibility with older SDK callers.
    """
    if (
        not isinstance(dest, (str, os.PathLike))
        and isinstance(data, (str, os.PathLike))
    ):
        dest, data = data, dest
    if data is _MISSING:
        raise TypeError('save_json() missing required argument: data')
    make_dirs(dest)
    with gzip.open(dest, 'wt', encoding='utf-8') if gz else open(
        dest, 'w', encoding='utf-8'
    ) as f:
        json.dump(data, f, cls=NumpyEncoder)
    return dest


def load_text(src, gz=False):
    with gzip.open(src, 'rt', encoding='utf-8') if gz else open(
        src, 'r', encoding='utf-8'
    ) as f:
        return f.read()


def save_text(dest, text, gz=False):
    make_dirs(dest)
    with gzip.open(dest, 'wt', encoding='utf-8') if gz else open(
        dest, 'w', encoding='utf-8'
    ) as f:
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
    with Image.open(src) as opened:
        im = ImageOps.exif_transpose(opened).convert(mode)
    if max_size is not None and (im.width > max_size or im.height > max_size):
        im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
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
    return [provider for provider in providers if provider in available_providers]


def process_map(task, *values, desc='', max_workers=3):
    """Apply a picklable task in parallel and return results with progress."""
    if not values:
        return []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = []
        with tqdm(total=len(values[0]), desc=desc) as pbar:
            for result in exe.map(task, *values):
                results.append(result)
                pbar.set_postfix_str(str(result))
                pbar.update(1)
        return results


def process_media(src, callback):
    if get_type(str(src)).startswith('image'):
        img = load_image(src)
        out = callback(img)
        show_image(out)
    else:
        with Video(src) as video:
            for frame in video:
                out = callback(frame)
                video.show(out)


__all__ = [
    'API_URL',
    'HEADERS',
    'FrameContext',
    'LineCounterStage',
    'NumpyEncoder',
    'Pipeline',
    'RegionFilterStage',
    'RegionTimerStage',
    'Sketcher',
    'TrackerStage',
    'Video',
    'VideoDisplay',
    'VideoInput',
    'Window',
    'array_copy',
    'array_from_image_buffer',
    'array_ndim',
    'array_shape',
    'array_size',
    'array_squeeze',
    'array_to_list',
    'as_array',
    'compose_mask_layers',
    'download_file',
    'download_url',
    'encode_image',
    'encode_mask_overlay',
    'get_color',
    'get_providers',
    'get_remote_file_size',
    'get_type',
    'image_base64',
    'is_url',
    'list_dir',
    'load_data',
    'load_image',
    'load_json',
    'load_text',
    'load_url',
    'load_url_bytes',
    'make_dirs',
    'mask_array',
    'md5sum',
    'merge_masks',
    'process_map',
    'process_media',
    'render_results',
    'resize_mask',
    'save_data',
    'save_image',
    'save_json',
    'save_text',
    'show_image',
    'temporal_src',
    'url_path',
    'zeros_array',
]
