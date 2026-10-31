import os
from pathlib import Path

import pytest

from abraia import Abraia
from abraia import cli as cli_module
from abraia.utils.cache import RemoteFileCache


def test_load_metadata_reads_the_cached_image(tmp_path, monkeypatch):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    image_path = tmp_path / "embedded.png"
    info = PngInfo()
    info.add_text("Comment", "cached metadata")
    Image.new("RGB", (2, 2)).save(image_path, pnginfo=info)
    client = object.__new__(Abraia)
    downloads = []

    def download(path):
        downloads.append(path)
        return str(image_path)

    monkeypatch.setattr(client, "download_file", download)

    assert client.load_metadata("project/image.png") == {
        "Comment": "cached metadata",
    }
    assert downloads == ["project/image.png"]


def test_remote_file_cache_is_account_scoped(tmp_path):
    first = RemoteFileCache("account-one", root=tmp_path)
    second = RemoteFileCache("account-two", root=tmp_path)
    first_path = Path(first.path_for("project/image.jpg"))
    second_path = Path(second.path_for("project/image.jpg"))

    first_path.write_bytes(b"private image")

    assert first_path != second_path
    assert first_path.exists()
    assert not second_path.exists()


def test_remote_file_cache_prunes_least_recently_used_entry(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path, max_bytes=5)
    old_path = Path(cache.path_for("old.bin"))
    active_path = Path(cache.path_for("active.bin"))
    old_path.write_bytes(b"1234")
    active_path.write_bytes(b"5678")
    os.utime(old_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(active_path, ns=(2_000_000_000, 2_000_000_000))

    removed = cache.prune(protected=str(active_path))

    assert removed == 1
    assert not old_path.exists()
    assert active_path.read_bytes() == b"5678"


def test_remote_file_cache_keeps_protected_oversize_entry(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path, max_bytes=2)
    path = Path(cache.path_for("large.bin"))
    path.write_bytes(b"larger than budget")

    assert cache.prune(protected=str(path)) == 0
    assert path.exists()


def test_remote_file_cache_can_invalidate_and_clear_account_entries(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path, max_bytes=100)
    image = Path(cache.path_for("project/image.jpg"))
    metadata = Path(cache.path_for("project/metadata.json"))
    image.write_bytes(b"image")
    metadata.write_bytes(b"metadata")

    assert cache.invalidate("project/image.jpg") is True
    assert cache.invalidate("project/image.jpg") is False
    assert cache.clear() == 1
    assert not metadata.exists()


def test_remote_file_cache_rejects_absolute_and_parent_paths(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path)

    with pytest.raises(ValueError):
        cache.path_for("../outside")
    with pytest.raises(ValueError):
        cache.path_for("/outside")



def test_remote_file_cache_coalesces_concurrent_misses(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    cache = RemoteFileCache("account", root=tmp_path)
    calls = []
    guard = threading.Lock()

    def fetch(destination):
        with guard:
            calls.append(destination)
        time.sleep(0.02)
        Path(destination).write_bytes(b"complete contents")

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(
            executor.map(
                lambda _index: cache.get_or_create("shared.bin", fetch),
                range(8),
            )
        )

    assert len(set(paths)) == 1
    assert len(calls) == 1
    assert Path(paths[0]).read_bytes() == b"complete contents"


def test_clear_waits_for_active_cache_fill(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    cache = RemoteFileCache("account", root=tmp_path)
    started = threading.Event()
    release = threading.Event()
    clear_started = threading.Event()

    def fetch(destination):
        started.set()
        assert release.wait(timeout=2)
        Path(destination).write_bytes(b"complete")

    def clear_cache():
        clear_started.set()
        return cache.clear()

    with ThreadPoolExecutor(max_workers=2) as executor:
        fill = executor.submit(cache.get_or_create, "active.bin", fetch)
        assert started.wait(timeout=2)
        clear = executor.submit(clear_cache)
        assert clear_started.wait(timeout=2)
        assert not clear.done()
        release.set()
        cached_path = fill.result(timeout=2)
        assert clear.result(timeout=2) == 1

    assert not Path(cached_path).exists()


def test_failed_cache_fill_does_not_publish_partial_file(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path)

    def fail_after_partial_write(destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("interrupted download")

    with pytest.raises(OSError, match="interrupted download"):
        cache.get_or_create("scene.bin", fail_after_partial_write)

    final_path = Path(cache.path_for("scene.bin"))
    assert not final_path.exists()
    assert not list((cache.root / ".staging").glob("download-*"))

    result = cache.get_or_create(
        "scene.bin", lambda destination: Path(destination).write_bytes(b"complete")
    )
    assert Path(result).read_bytes() == b"complete"


def test_pruning_does_not_remove_in_progress_staging_file(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path, max_bytes=1)
    staging = cache.root / ".staging" / "download-active.part"
    staging.parent.mkdir(parents=True)
    staging.write_bytes(b"active transfer")
    cached = Path(cache.path_for("old.bin"))
    cached.write_bytes(b"old contents")

    assert cache.prune() == 1
    assert staging.read_bytes() == b"active transfer"
    assert not cached.exists()



def test_listing_reconciliation_refreshes_changed_cached_file(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path)
    calls = []

    def fetch(content):
        def write(destination):
            calls.append(content)
            Path(destination).write_bytes(content)
        return write

    first_info = {"size": 3, "date": "v1"}
    changed_info = {"size": 5, "date": "v2"}
    path = cache.get_or_create("scene.bin", fetch(b"one"), remote_info=first_info)
    reused = cache.get_or_create(
        "scene.bin", fetch(b"two"), remote_info=changed_info
    )

    assert path == reused
    assert calls == [b"one"]
    cache.reconcile_listing("", [{"path": "scene.bin", **changed_info}])
    updated = cache.get_or_create(
        "scene.bin", fetch(b"newer"), remote_info=changed_info
    )

    assert updated == path
    assert calls == [b"one", b"newer"]
    assert Path(updated).read_bytes() == b"newer"


def test_cache_hit_does_not_request_remote_metadata(tmp_path):
    cache = RemoteFileCache("account", root=tmp_path)
    calls = []
    path = cache.get_or_create(
        "scene.bin",
        lambda destination: Path(destination).write_bytes(b"cached"),
        remote_info={"size": 6, "date": "listed"},
    )

    result = cache.get_or_create(
        "scene.bin", lambda destination: calls.append(destination)
    )

    assert result == path
    assert Path(result).read_bytes() == b"cached"
    assert calls == []


def test_model_asset_download_uses_the_common_cache(tmp_path, monkeypatch):
    from abraia.utils import remote

    cache = RemoteFileCache("shared-artifacts", root=tmp_path)
    calls = []
    monkeypatch.setattr(
        remote, "RemoteFileCache", lambda _userid: cache
    )

    def download(url, destination):
        calls.append(url)
        Path(destination).write_bytes(b"model")

    monkeypatch.setattr(remote, "download_url", download)

    first = remote.download_file("multiple/models/model.onnx")
    second = remote.download_file("multiple/models/model.onnx")

    assert first == second
    assert Path(first).read_bytes() == b"model"
    assert calls == ["https://api.abraia.me/files/multiple/models/model.onnx"]


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


@pytest.mark.parametrize("folder", ["", "nested/downloads"])
def test_cli_download_uses_client_cache(tmp_path, monkeypatch, folder):
    monkeypatch.chdir(tmp_path)
    client = object.__new__(Abraia)
    client.userid = "account"
    cache = RemoteFileCache("account", root=tmp_path / "cache")
    monkeypatch.setattr("abraia.client.RemoteFileCache", lambda _userid: cache)
    calls = []

    def download(path, destination):
        calls.append(path)
        Path(destination).write_bytes(b"fresh remote contents")

    monkeypatch.setattr(client, "_download_file_uncached", download)
    monkeypatch.setattr(cli_module, "abraia", client)

    result = cli_module.download_file("remote/image.jpg", folder)

    assert result == str(Path(folder) / "image.jpg")
    assert Path(result).read_bytes() == b"fresh remote contents"
    cli_module.download_file("remote/image.jpg", folder)
    assert calls == ["remote/image.jpg"]


def test_cli_download_passes_listing_metadata_into_client(tmp_path, monkeypatch):
    output = tmp_path / "downloads"
    output.mkdir()
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


def cache_factory(monkeypatch, root):
    import abraia.client as client_module

    cache = RemoteFileCache("user", root=root)
    monkeypatch.setattr(
        client_module, "RemoteFileCache", lambda _userid: cache
    )
    return cache


def test_download_uses_cache_without_requesting_remote_metadata(
    tmp_path, monkeypatch
):
    captured = {}

    class Cache:
        def get_or_create(self, path, fetch, **options):
            captured.update(options)
            destination = tmp_path / Path(path).name
            fetch(str(destination))
            return str(destination)

    import abraia.client as client_module

    monkeypatch.setattr(client_module, "RemoteFileCache", lambda _userid: Cache())

    class RemoteClient:
        userid = "user"

        def download_file(self, path, destination):
            Path(destination).write_bytes(path.encode())
            return destination

    cached = Abraia(client=RemoteClient()).download_file("folder/tb_image.jpg")

    assert Path(cached).read_bytes() == b"folder/tb_image.jpg"
    assert "remote_info_provider" not in captured


def test_download_returns_cached_file_until_list_files_invalidates_it(
    tmp_path, monkeypatch
):
    cache = cache_factory(monkeypatch, tmp_path)
    downloads = []

    class RemoteClient:
        userid = "user"

        def download_file(self, path, destination):
            downloads.append(path)
            Path(destination).write_bytes(b"remote content")
            return destination

    client = Abraia(client=RemoteClient())
    original = {"size": 14, "date": "v1"}
    changed = {"size": 15, "date": "v2"}
    first = client.download_file("folder/image.jpg", remote_info=original)
    second = client.download_file("folder/image.jpg", remote_info=changed)

    assert first == second
    assert Path(second).read_bytes() == b"remote content"
    assert downloads == ["folder/image.jpg"]
    assert cache._read_signature("folder/image.jpg") == {
        "exists": True,
        "size": 14,
        "date": "v1",
    }


def test_upload_caches_local_bytes_without_remote_listing(tmp_path, monkeypatch):
    cache = cache_factory(monkeypatch, tmp_path)
    listing_calls = []
    cache.store("folder/tb_image.jpg", b"old thumbnail")

    class RemoteClient:
        userid = "user"

        def upload_file(self, _source, path):
            return path

        def list_files(self, path):
            listing_calls.append(path)
            return [], []

    source = tmp_path / "image.jpg"
    source.write_bytes(b"uploaded bytes")
    result = Abraia(client=RemoteClient()).upload_file(
        str(source), "folder/image.jpg"
    )

    cached = Path(cache.path_for("folder/image.jpg"))
    assert result == "folder/image.jpg"
    assert cached.read_bytes() == b"uploaded bytes"
    assert not Path(cache.path_for("folder/tb_image.jpg")).exists()
    assert listing_calls == []


def test_thumbnail_cache_entries_do_not_store_source_metadata(tmp_path):
    cache = RemoteFileCache("user", root=tmp_path)
    source_info = {"size": 128, "date": "source-v1"}

    cache.store(
        "folder/tb_image.jpg",
        b"thumbnail",
        source_info,
    )
    assert cache._read_signature("folder/tb_image.jpg") is None

    cache.invalidate("folder/tb_image.jpg")
    cache.get_or_create(
        "folder/tb_image.jpg",
        lambda destination: Path(destination).write_bytes(b"thumbnail"),
        remote_info=source_info,
    )
    assert cache._read_signature("folder/tb_image.jpg") is None


def test_upload_still_caches_bytes_when_response_metadata_is_stale(
    tmp_path, monkeypatch
):
    cache = cache_factory(monkeypatch, tmp_path)

    class RemoteClient:
        userid = "user"

        def upload_file(self, _source, path):
            return path

    source = tmp_path / "image.jpg"
    source.write_bytes(b"uploaded bytes")
    Abraia(client=RemoteClient()).upload_file(
        str(source),
        "folder/image.jpg",
        remote_info={"size": 1, "date": "pre-upload"},
    )

    cached = Path(cache.path_for("folder/image.jpg"))
    assert cached.read_bytes() == b"uploaded bytes"
    assert cache._read_signature("folder/image.jpg") is None


def test_folder_listing_invalidates_changed_and_deleted_sources_and_thumbnails(
    tmp_path, monkeypatch
):
    cache = cache_factory(monkeypatch, tmp_path)
    old_date = "old"
    current_date = "new"
    for path in (
        "folder/changed.jpg",
        "folder/tb_changed.jpg",
        "folder/deleted.jpg",
        "folder/tb_deleted.jpg",
        "folder/stable.jpg",
        "folder/tb_stable.jpg",
    ):
        data = path.encode()
        remote_info = (
            None
            if Path(path).name.startswith("tb_")
            else {"size": len(path.encode()), "date": old_date}
        )
        cache.store(path, data, remote_info)

    checked_paths = []
    read_signature = cache._read_signature
    monkeypatch.setattr(
        cache,
        "_read_signature",
        lambda path: checked_paths.append(path) or read_signature(path),
    )

    class RemoteClient:
        userid = "user"

        def list_files(self, _path):
            return [
                {
                    "path": "folder/changed.jpg",
                    "size": len(b"folder/changed.jpg"),
                    "date": current_date,
                },
                {
                    "path": "folder/stable.jpg",
                    "size": len(b"folder/stable.jpg"),
                    "date": old_date,
                },
                # Remote thumbnail metadata does not affect thumbnail cache life.
                {"path": "folder/tb_stable.jpg", "size": 999, "date": "new"},
            ], []

    Abraia(client=RemoteClient()).list_files("folder/")

    assert not Path(cache.path_for("folder/changed.jpg")).exists()
    assert not Path(cache.path_for("folder/tb_changed.jpg")).exists()
    assert not Path(cache.path_for("folder/deleted.jpg")).exists()
    assert not Path(cache.path_for("folder/tb_deleted.jpg")).exists()
    assert Path(cache.path_for("folder/stable.jpg")).read_bytes() == b"folder/stable.jpg"
    assert Path(cache.path_for("folder/tb_stable.jpg")).read_bytes() == b"folder/tb_stable.jpg"
    assert set(checked_paths) == {
        "folder/changed.jpg",
        "folder/stable.jpg",
    }
    assert all(not Path(path).name.startswith("tb_") for path in checked_paths)


def test_delete_and_move_invalidate_source_and_dependent_thumbnail(
    tmp_path, monkeypatch
):
    cache = cache_factory(monkeypatch, tmp_path)
    for path in (
        "folder/image.jpg",
        "folder/tb_image.jpg",
        "folder/moved.jpg",
        "folder/tb_moved.jpg",
    ):
        cache.store(path, b"cached", {"size": 6, "date": "v1"})

    class RemoteClient:
        userid = "user"

        def remove_file(self, path):
            return path

        def move_file(self, old_path, new_path):
            return new_path

    client = Abraia(client=RemoteClient())
    client.remove_file("folder/image.jpg")
    client.move_file("folder/moved.jpg", "folder/renamed.jpg")

    for path in (
        "folder/image.jpg",
        "folder/tb_image.jpg",
        "folder/moved.jpg",
        "folder/tb_moved.jpg",
        "folder/renamed.jpg",
        "folder/tb_renamed.jpg",
    ):
        assert not Path(cache.path_for(path)).exists()
