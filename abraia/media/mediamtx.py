"""Install and run MediaMTX for the Raspberry Pi camera demo."""

from contextlib import contextmanager
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


RELEASE_API = "https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
RTSP_PORT = 8554
STARTUP_TIMEOUT = 20


def _cache_root(cache_dir):
    return (
        Path(cache_dir)
        if cache_dir
        else Path.home() / ".cache" / "abraia" / "mediamtx"
    )


def _linux_architecture():
    if platform.system() != "Linux":
        raise RuntimeError(
            "The automatic pi-camera setup requires Linux on a Raspberry Pi."
        )
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("armv6l", "armv6"):
        return "armv6"
    if machine in ("armv7l", "armv8l"):
        return "armv7"
    raise RuntimeError(
        f"The automatic pi-camera setup does not support Linux architecture "
        f"'{machine}'. Use Raspberry Pi OS on a 32-bit or 64-bit ARM device."
    )


def _release_asset(release, name):
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            url = asset.get("browser_download_url", "")
            parsed_url = urlparse(url)
            if parsed_url.scheme != "https" or parsed_url.hostname != "github.com":
                raise RuntimeError(
                    "MediaMTX release assets must come from GitHub over HTTPS."
                )
            return url
    raise RuntimeError(f"MediaMTX release is missing the '{name}' asset.")


def _download(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _extract_binary(archive_data, archive_name):
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as archive:
            member = next(
                item for item in archive.getmembers()
                if item.isfile() and Path(item.name).name == "mediamtx"
            )
            binary = archive.extractfile(member)
            if binary is None:
                raise RuntimeError("MediaMTX archive does not contain its executable.")
            return binary.read()
    except (tarfile.TarError, StopIteration) as error:
        raise RuntimeError(
            f"Unable to read the MediaMTX archive '{archive_name}'."
        ) from error


def install_mediamtx(cache_dir=None):
    """Return a cached MediaMTX executable, installing a verified release once."""
    architecture = _linux_architecture()
    existing = shutil.which("mediamtx")
    if existing:
        return Path(existing)

    root = _cache_root(cache_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    binary_path = root / "mediamtx"
    if binary_path.is_file() and os.access(str(binary_path), os.X_OK):
        return binary_path

    response = requests.get(RELEASE_API, timeout=30)
    response.raise_for_status()
    release = response.json()
    tag = release.get("tag_name", "")
    if not re.fullmatch(r"v?[0-9A-Za-z._-]+", tag):
        raise RuntimeError("MediaMTX returned an invalid release version.")

    archive_name = f"mediamtx_{tag}_linux_{architecture}.tar.gz"
    archive_url = _release_asset(release, archive_name)
    checksums_url = _release_asset(release, "checksums.sha256")
    archive_data = _download(archive_url)
    checksums = _download(checksums_url).decode("utf-8", errors="replace")
    expected = None
    for line in checksums.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and Path(parts[1].lstrip("* ")).name == archive_name:
            expected = parts[0].lower()
            break
    if expected is None:
        raise RuntimeError(f"MediaMTX checksums do not include '{archive_name}'.")
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise RuntimeError("MediaMTX published an invalid SHA-256 checksum.")
    actual = hashlib.sha256(archive_data).hexdigest()
    if not secrets.compare_digest(actual, expected):
        raise RuntimeError("The downloaded MediaMTX archive failed its SHA-256 check.")

    binary_data = _extract_binary(archive_data, archive_name)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".mediamtx-", dir=str(root))
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(binary_data)
        os.chmod(temporary_path, 0o700)
        os.replace(temporary_path, str(binary_path))
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return binary_path


def _write_camera_config(root, camera_id):
    config_path = root / "mediamtx.yml"
    state_path = root / "pi-camera.json"
    if config_path.is_file() and state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("camera_id") == camera_id:
                return state["password"]
        except (OSError, ValueError, KeyError):
            pass

    password = secrets.token_hex(16)
    config = (
        "authMethod: internal\n"
        "authInternalUsers:\n"
        "  - user: abraia-reader\n"
        f"    pass: {password}\n"
        "    ips: []\n"
        "    permissions:\n"
        "      - action: read\n"
        "        path: cam\n"
        "rtspTransports: [tcp]\n"
        "paths:\n"
        "  cam:\n"
        "    source: rpiCamera\n"
        f"    rpiCameraCamID: {camera_id}\n"
        "    rpiCameraWidth: 1280\n"
        "    rpiCameraHeight: 720\n"
        "    rpiCameraFPS: 30\n"
    )
    state = json.dumps({"camera_id": camera_id, "password": password})
    for path, content in ((config_path, config), (state_path, state)):
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}-", dir=str(root)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
                temporary.write(content)
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, str(path))
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    return password


def _port_is_open():
    try:
        with socket.create_connection(("127.0.0.1", RTSP_PORT), timeout=0.2):
            return True
    except OSError:
        return False


def _lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 80))
            return connection.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@contextmanager
def managed_pi_camera_server(camera_id=0, cache_dir=None):
    """Install MediaMTX if needed and run its native Raspberry Pi camera source."""
    try:
        camera_id = int(camera_id)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The Raspberry Pi camera ID must be a non-negative integer."
        ) from error
    if camera_id < 0:
        raise ValueError("The Raspberry Pi camera ID must be a non-negative integer.")

    root = _cache_root(cache_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    binary = install_mediamtx(cache_dir=root)
    password = _write_camera_config(root, camera_id)
    if _port_is_open():
        raise RuntimeError(
            f"Port {RTSP_PORT} is already in use. Stop the existing RTSP server "
            "or set ABRAIA_RTSP_URL to publish through it."
        )

    process = subprocess.Popen([str(binary)], cwd=str(root))
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    "MediaMTX stopped during startup. Check that this Raspberry "
                    "Pi OS image supports its native camera source and that a "
                    "camera is connected."
                )
            if _port_is_open():
                stream_url = (
                    f"rtsp://abraia-reader:{password}@{_lan_address()}:"
                    f"{RTSP_PORT}/cam"
                )
                yield stream_url
                return
            time.sleep(0.1)
        raise RuntimeError("MediaMTX did not open its RTSP port in time.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


__all__ = ["install_mediamtx", "managed_pi_camera_server"]
