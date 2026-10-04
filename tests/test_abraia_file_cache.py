from pathlib import Path

from abraia.client import Abraia
from abraia.utils.cache import RemoteFileCache


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
