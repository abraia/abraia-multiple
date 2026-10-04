from pathlib import Path

from abraia import Abraia
from abraia import cli as cli_module
from abraia.utils.cache import RemoteFileCache


def test_invalidate_cached_removes_source_and_thumbnail(tmp_path, monkeypatch):
    client = object.__new__(Abraia)
    client.userid = "account"
    cache = RemoteFileCache("account", root=tmp_path)
    cached_file = Path(cache.path_for("remote/image.jpg"))
    cached_thumbnail = Path(cache.path_for("remote/tb_image.jpg"))
    cached_file.write_bytes(b"cached")
    cached_thumbnail.write_bytes(b"thumbnail")
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)

    assert client.invalidate_cached("remote/image.jpg") is True
    assert not cached_file.exists()
    assert not cached_thumbnail.exists()
    assert client.invalidate_cached("remote/image.jpg") is False


def test_cli_download_uses_client_cache(tmp_path, monkeypatch):
    output = tmp_path / "downloads"
    calls = []

    def download(path, destination, remote_info=None):
        calls.append((path, destination, remote_info))
        Path(destination).write_bytes(b"fresh remote contents")
        return destination

    monkeypatch.setattr(cli_module.abraia, "download_file", download)

    result = cli_module.download_file("remote/image.jpg", str(output))

    assert result == str(output / "image.jpg")
    assert Path(result).read_bytes() == b"fresh remote contents"
    assert calls == [("remote/image.jpg", str(output / "image.jpg"), None)]


def test_cli_download_passes_listing_metadata_into_client(tmp_path, monkeypatch):
    output = tmp_path / "downloads"
    calls = []
    record = {"path": "remote/image.jpg", "size": 23, "date": "version-1"}

    def download(path, destination, remote_info=None):
        calls.append((path, destination, remote_info))
        Path(destination).write_bytes(b"cached remote contents")
        return destination

    monkeypatch.setattr(cli_module.abraia, "download_file", download)

    result = cli_module.download_file(record, str(output))

    assert result == str(output / "image.jpg")
    assert Path(result).read_bytes() == b"cached remote contents"
    assert calls == [("remote/image.jpg", str(output / "image.jpg"), record)]


def test_download_file_uses_cache_and_materializes_destination(tmp_path, monkeypatch):
    client = object.__new__(Abraia)
    client.userid = "account"
    cache = RemoteFileCache("account", root=tmp_path)
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)
    record = {"path": "remote/image.jpg", "size": 6, "date": "version-1"}
    calls = []

    def fetch(path, destination):
        calls.append(path)
        Path(destination).write_bytes(b"pixels")

    monkeypatch.setattr(client, "_download_file_uncached", fetch)
    destination = tmp_path / "downloads" / "image.jpg"

    assert client.download_file("remote/image.jpg", str(destination)) == str(destination)
    assert destination.read_bytes() == b"pixels"
    client.download_file("remote/image.jpg", str(destination))
    assert calls == ["remote/image.jpg"]


def test_load_json_reads_through_shared_file_cache(tmp_path, monkeypatch):
    client = object.__new__(Abraia)
    client._client = None
    document = tmp_path / "cached.json"
    document.write_text('{"cached": true}')
    monkeypatch.setattr(client, "download_file", lambda _path: str(document))

    assert client.load_json("remote/data.json") == {"cached": True}


def test_upload_seeds_shared_cache_with_uploaded_bytes(tmp_path, monkeypatch):
    class Response:
        status_code = 201

        def json(self):
            return {
                "file": {
                    "source": "account/remote/image.jpg",
                    "size": 6,
                    "date": "version-1",
                }
            }

    client = object.__new__(Abraia)
    client._client = None
    client.userid = "account"
    client.auth = object()
    cache = RemoteFileCache("account", root=tmp_path)
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: Response())
    source = tmp_path / "image.jpg"
    source.write_bytes(b"pixels")

    assert client.upload_file(str(source), "remote/image.jpg") == "remote/image.jpg"
    cached = Path(cache.path_for("remote/image.jpg"))
    assert cached.read_bytes() == b"pixels"
    assert cache._read_signature("remote/image.jpg") == {
        "exists": True,
        "size": 6,
        "date": "version-1",
    }


def test_remote_file_cache_store_seeds_matching_version(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path)
    record = {"size": 6, "date": "version-1"}
    seeded = cache.store("remote/image.jpg", b"pixels", record)
    fetched = []

    result = cache.get_or_create(
        "remote/image.jpg",
        lambda destination: fetched.append(destination),
        remote_info=record,
    )

    assert result == seeded
    assert Path(result).read_bytes() == b"pixels"
    assert fetched == []



def test_download_file_keeps_an_existing_cached_file(tmp_path, monkeypatch):
    client = object.__new__(Abraia)
    client.userid = "account"
    cache = RemoteFileCache("account", root=tmp_path)
    destination = Path(cache.path_for("remote/image.jpg"))
    destination.write_bytes(b"cached contents")
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)
    downloads = []
    monkeypatch.setattr(
        client,
        "_download_file_uncached",
        lambda path, _temporary: downloads.append(path),
    )

    result = client.download_file("remote/image.jpg")

    assert result == str(destination)
    assert destination.read_bytes() == b"cached contents"
    assert downloads == []


def test_load_file_refreshes_through_the_shared_cache(tmp_path, monkeypatch):
    client = object.__new__(Abraia)
    destination = tmp_path / "document.txt"
    destination.write_text("cached text")
    calls = []
    monkeypatch.setattr(
        client,
        "download_file",
        lambda path: calls.append(("cached", path)) or str(destination),
    )
    client._client = None

    assert client.load_file("remote/document.txt") == "cached text"
    assert calls == [("cached", "remote/document.txt")]



def test_injected_client_reads_still_use_shared_file_cache(tmp_path, monkeypatch):
    class Transport:
        userid = "account"

        def __init__(self):
            self.downloads = []

        def list_files(self, _path):
            return ([{"path": "remote/data.json", "size": 16, "date": "v1"}], [])

        def download_file(self, path, destination):
            self.downloads.append(path)
            Path(destination).write_text('{"cached": true}')
            return destination

        def load_json(self, _path):
            raise AssertionError("high-level read bypassed the cache")

    transport = Transport()
    client = Abraia(client=transport)
    cache = RemoteFileCache("account", root=tmp_path)
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)

    assert client.load_json("remote/data.json") == {"cached": True}
    assert transport.downloads == ["remote/data.json"]


def test_injected_client_writes_still_seed_shared_file_cache(tmp_path, monkeypatch):
    class Transport:
        userid = "account"

        def list_files(self, _path):
            return ([{"path": "remote/data.json", "size": 16, "date": "v1"}], [])

        def upload_file(self, source, path):
            assert source.getvalue() == b'{"cached": true}'
            return path

        def save_json(self, *_args):
            raise AssertionError("high-level write bypassed upload cache policy")

    client = Abraia(client=Transport())
    cache = RemoteFileCache("account", root=tmp_path)
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)

    assert client.save_json("remote/data.json", {"cached": True}) == "remote/data.json"
    assert Path(cache.path_for("remote/data.json")).read_bytes() == b'{"cached": true}'


def test_successful_sdk_file_mutations_invalidate_cached_paths(monkeypatch):
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    client = object.__new__(Abraia)
    client._client = None
    client.userid = "account"
    client.auth = object()
    invalidated = []
    responses = [
        Response(201, {"file": {"source": "account/demo/image.jpg"}}),
        Response(201, {"file": {"source": "account/demo/new.jpg"}}),
        Response(200, {"file": {"source": "account/demo/new.jpg"}}),
    ]
    monkeypatch.setattr(client, "invalidate_cached", invalidated.append)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    assert client.upload_file("https://source/image.jpg", "demo/image.jpg") == "demo/image.jpg"
    assert client.move_file("demo/image.jpg", "demo/new.jpg") == "demo/new.jpg"
    assert client.remove_file("demo/new.jpg") == "demo/new.jpg"
    assert invalidated == [
        "demo/image.jpg",
        "demo/image.jpg",
        "demo/new.jpg",
        "demo/new.jpg",
    ]



def test_folder_listing_invalidates_before_the_next_download(tmp_path, monkeypatch):
    current = {"path": "remote/image.jpg", "size": 3, "date": "v1"}
    calls = []

    class Transport:
        userid = "account"

        def list_files(self, _path):
            return ([dict(current)], [])

        def download_file(self, path, destination):
            calls.append(path)
            Path(destination).write_bytes(
                b"one" if current["date"] == "v1" else b"two!!"
            )
            return destination

    client = Abraia(client=Transport())
    cache = RemoteFileCache("account", root=tmp_path)
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)

    first = client.download_file("remote/image.jpg", remote_info=dict(current))
    second = client.download_file("remote/image.jpg", remote_info=dict(current))
    assert first == second
    assert calls == ["remote/image.jpg"]

    current.update(size=5, date="v2")
    client.list_files("remote/")
    third = client.download_file("remote/image.jpg", remote_info=dict(current))

    assert first == second == third
    assert calls == ["remote/image.jpg", "remote/image.jpg"]
    assert Path(third).read_bytes() == b"two!!"


