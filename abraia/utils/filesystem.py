"""Small filesystem and serialization helpers used by the SDK."""

import gzip
import hashlib
import json
import mimetypes
import os

import numpy as np


mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/heic", ".heic")
mimetypes.add_type("image/heif", ".heif")


def get_type(path):
    """Return the MIME type inferred from a path."""
    return mimetypes.guess_type(os.fspath(path))[0] or "binary/octet-stream"


def make_dirs(dest):
    """Create the parent directory for ``dest`` when necessary."""
    dirname = os.path.dirname(os.fspath(dest))
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def list_dir(folder):
    """Return regular files in ``folder`` in deterministic order."""
    folder = os.fspath(folder)
    return [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if os.path.isfile(os.path.join(folder, name))
    ]


def md5sum(src):
    """Return the MD5 digest of a path or binary file-like object.

    File-like objects are read from their beginning and restored to their
    original position, which keeps upload callers reusable after hashing.
    """
    digest = hashlib.md5()
    if hasattr(src, "read"):
        position = src.tell() if hasattr(src, "tell") else None
        if hasattr(src, "seek"):
            src.seek(0)
        try:
            for chunk in iter(lambda: src.read(4096), b""):
                digest.update(chunk)
        finally:
            if position is not None and hasattr(src, "seek"):
                src.seek(position)
        return digest.hexdigest()

    with open(src, "rb") as fileobj:
        for chunk in iter(lambda: fileobj.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for common NumPy scalar and array values."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_json(src, gz=False):
    with gzip.open(src, "rt", encoding="utf-8") if gz else open(
        src, "r", encoding="utf-8"
    ) as fileobj:
        return json.load(fileobj)


def save_json(dest, data, gz=False):
    """Save JSON using destination-first arguments."""
    make_dirs(dest)
    with gzip.open(dest, "wt", encoding="utf-8") if gz else open(
        dest, "w", encoding="utf-8"
    ) as fileobj:
        json.dump(data, fileobj, cls=NumpyEncoder)
    return dest


def load_text(src, gz=False):
    with gzip.open(src, "rt", encoding="utf-8") if gz else open(
        src, "r", encoding="utf-8"
    ) as fileobj:
        return fileobj.read()


def save_text(dest, text, gz=False):
    make_dirs(dest)
    with gzip.open(dest, "wt", encoding="utf-8") if gz else open(
        dest, "w", encoding="utf-8"
    ) as fileobj:
        fileobj.write(text)
    return dest


def load_data(src, gz=False):
    with gzip.open(src, "rb") if gz else open(src, "rb") as fileobj:
        return fileobj.read()


def save_data(dest, data, gz=False):
    make_dirs(dest)
    with gzip.open(dest, "wb") if gz else open(dest, "wb") as fileobj:
        fileobj.write(data)
    return dest


__all__ = [
    "NumpyEncoder",
    "get_type",
    "list_dir",
    "load_data",
    "load_json",
    "load_text",
    "make_dirs",
    "md5sum",
    "save_data",
    "save_json",
    "save_text",
]
