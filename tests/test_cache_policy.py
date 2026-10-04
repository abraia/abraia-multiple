import os
from pathlib import Path

import pytest

from abraia.utils.cache import RemoteFileCache


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
