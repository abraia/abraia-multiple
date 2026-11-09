"""Tests for the first-run MediaMTX setup used by the Pi camera demo."""

from contextlib import contextmanager
from hashlib import sha256
import io
import tarfile
from unittest.mock import patch

import pytest

from abraia.media import mediamtx
from abraia import demo as demo_module


@pytest.mark.parametrize(
    "machine, expected",
    [("aarch64", "arm64"), ("armv7l", "armv7"), ("armv6l", "armv6")],
)
def test_mediamtx_release_architecture_for_raspberry_pi(monkeypatch, machine, expected):
    monkeypatch.setattr(mediamtx.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mediamtx.platform, "machine", lambda: machine)

    assert mediamtx._linux_architecture() == expected


def _release_archive(binary=b"mediamtx executable"):
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:gz") as archive:
        member = tarfile.TarInfo("mediamtx")
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    return archive_data.getvalue()


def _mock_arm_release(monkeypatch, *, valid_checksum=True):
    archive_name = "mediamtx_v1.2.3_linux_arm64.tar.gz"
    archive_data = _release_archive()
    digest = sha256(archive_data).hexdigest() if valid_checksum else "0" * 64
    checksum_data = f"{digest}  {archive_name}\n".encode()
    release_base = "https://github.com/bluenviron/mediamtx/releases/download/v1.2.3"
    archive_url = f"{release_base}/mediamtx.tar.gz"
    checksums_url = f"{release_base}/checksums.sha256"
    release = {
        "tag_name": "v1.2.3",
        "assets": [
            {
                "name": archive_name,
                "browser_download_url": archive_url,
            },
            {
                "name": "checksums.sha256",
                "browser_download_url": checksums_url,
            },
        ],
    }
    contents = {archive_url: archive_data, checksums_url: checksum_data}

    class Response:
        def __init__(self, url):
            self.content = contents.get(url)

        def raise_for_status(self):
            return None

        def json(self):
            return release

    def get(url, **kwargs):
        assert kwargs.get("timeout", 0) > 0
        return Response(url)

    monkeypatch.setattr(mediamtx.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mediamtx, "_linux_architecture", lambda: "arm64")
    monkeypatch.setattr(mediamtx.requests, "get", get)


def test_install_mediamtx_downloads_checksum_verified_arm_release(
    tmp_path, monkeypatch
):
    _mock_arm_release(monkeypatch)

    binary_path = mediamtx.install_mediamtx(cache_dir=tmp_path)

    assert binary_path.read_bytes() == b"mediamtx executable"
    assert binary_path.stat().st_mode & 0o111


def test_install_mediamtx_rejects_a_bad_release_checksum(tmp_path, monkeypatch):
    _mock_arm_release(monkeypatch, valid_checksum=False)

    with pytest.raises(RuntimeError, match="failed its SHA-256 check"):
        mediamtx.install_mediamtx(cache_dir=tmp_path)


def test_pi_camera_server_writes_config_and_stops_child_process(tmp_path):
    processes = []

    class Process:
        def __init__(self, *_args, **_kwargs):
            self.stopped = False
            self.waited = False
            processes.append(self)

        def poll(self):
            return -15 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            self.waited = True
            return 0

    with patch.object(
         mediamtx, "install_mediamtx", return_value=tmp_path / "mediamtx"
    ), \
         patch.object(mediamtx, "_port_is_open", side_effect=[False, True]), \
         patch.object(mediamtx, "_lan_address", return_value="192.0.2.10"), \
         patch.object(mediamtx.subprocess, "Popen", Process):
        with mediamtx.managed_pi_camera_server(
            camera_id="1", cache_dir=tmp_path
        ) as url:
            assert url.startswith("rtsp://abraia-reader:")
            assert url.endswith("@192.0.2.10:8554/cam")
            config = (tmp_path / "mediamtx.yml").read_text(encoding="utf-8")
            assert "source: rpiCamera" in config
            assert "rpiCameraCamID: 1" in config

    assert len(processes) == 1
    assert processes[0].stopped
    assert processes[0].waited


def test_pi_camera_demo_starts_the_managed_server_by_default(monkeypatch):
    camera_ids = []

    @contextmanager
    def managed_server(camera_id):
        camera_ids.append(camera_id)
        yield "rtsp://reader@example.test:8554/cam"

    monkeypatch.delenv("ABRAIA_RTSP_URL", raising=False)
    with patch.object(
        mediamtx, "managed_pi_camera_server", managed_server
    ), patch.object(demo_module.time, "sleep", side_effect=KeyboardInterrupt):
        demo_module.run_camera_pipeline()

    assert camera_ids == [0]


@pytest.mark.parametrize("camera_id", [-1, "camera"])
def test_pi_camera_server_rejects_invalid_camera_id(tmp_path, camera_id):
    with pytest.raises(ValueError, match="non-negative integer"):
        with mediamtx.managed_pi_camera_server(camera_id, cache_dir=tmp_path):
            pass
