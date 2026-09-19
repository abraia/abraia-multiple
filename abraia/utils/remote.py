"""HTTP and SDK cache helpers."""

import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


API_URL = "https://api.abraia.me"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    )
}
tempdir = tempfile.gettempdir()

_url_sessions = threading.local()


def create_session(headers=None):
    """Create a connection-pooled HTTP session for SDK requests."""
    session = requests.Session()
    session.headers.update(headers or HEADERS)
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_url_session():
    """Return a connection-pooled session local to the current thread."""
    session = getattr(_url_sessions, "session", None)
    if session is None:
        session = create_session()
        _url_sessions.session = session
    return session


def request_with_retries(
    session,
    method,
    url,
    retries=3,
    backoff=0.5,
    **kwargs,
):
    """Perform a request with retry-safe handling for rewindable bodies."""
    data = kwargs.get("data")
    position = data.tell() if hasattr(data, "tell") else None
    last_error = None
    for attempt in range(retries):
        if position is not None and hasattr(data, "seek"):
            data.seek(position)
        try:
            return session.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as error:
            last_error = error
            if attempt + 1 == retries:
                raise
            time.sleep(backoff * (2 ** attempt))
    raise last_error


def is_url(url):
    return str(url).startswith(("http://", "https://"))


def url_path(path):
    path = os.fspath(path).replace("\\", "/")
    return f"{API_URL}/files/{quote(path.lstrip('/'), safe='/')}"


def get_remote_file_size(url: str, timeout: int = 30):
    """Get the size of a remote file via a best-effort HEAD request."""
    try:
        with _get_url_session().head(
            url, timeout=timeout, allow_redirects=True
        ) as response:
            if not response.ok:
                return None
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (OSError, ValueError, requests.RequestException):
        return None


def temporal_src(path):
    """Return a path below the process cache directory.

    Relative path components containing ``..`` are rejected so remote names
    cannot escape the cache through ordinary path traversal.
    """
    relative = Path(os.fspath(path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Temporary paths must remain relative to the cache directory")
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
        prefix=f".{filename}.", suffix=".part", dir=destination.parent
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
            total = int(response.headers.get("content-length", 0))
            with temp_dest.open("wb") as fileobj, tqdm(
                desc=filename,
                total=total,
                unit="iB",
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

    Callers own the returned raw stream and must close it after consuming it.
    """
    for attempt in range(3):
        try:
            response = request_with_retries(
                _get_url_session(),
                "GET",
                url,
                retries=3,
                backoff=0.25,
                stream=True,
                allow_redirects=True,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.raw
            response.close()
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


__all__ = [
    "API_URL",
    "HEADERS",
    "create_session",
    "download_file",
    "download_url",
    "get_remote_file_size",
    "is_url",
    "load_url",
    "load_url_bytes",
    "request_with_retries",
    "temporal_src",
    "url_path",
]
