"""Account-scoped local cache for remote Abraia files."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import posixpath
from pathlib import Path
import shutil
import tempfile
import threading
import time


DEFAULT_CACHE_MAX_BYTES = 10 * 1024 ** 3
CACHE_MAX_BYTES_ENV = "ABRAIA_CACHE_MAX_BYTES"
_PATH_LOCKS = {}
_PATH_LOCKS_GUARD = threading.Lock()
_ACCOUNT_LOCKS = {}
_ACCOUNT_LOCKS_GUARD = threading.Lock()


class _AccountLock:
    """Allow parallel operations but keep account-wide clears exclusive."""

    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def shared(self):
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if not self._readers:
                    self._condition.notify_all()

    @contextmanager
    def exclusive(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


def _account_lock(path):
    key = os.path.normcase(os.path.abspath(path))
    with _ACCOUNT_LOCKS_GUARD:
        lock = _ACCOUNT_LOCKS.get(key)
        if lock is None:
            lock = _AccountLock()
            _ACCOUNT_LOCKS[key] = lock
        return lock


@contextmanager
def _locked_path(path):
    """Serialize cache fills for one path across cache instances."""
    key = os.path.normcase(os.path.abspath(path))
    with _PATH_LOCKS_GUARD:
        entry = _PATH_LOCKS.get(key)
        if entry is None:
            entry = [threading.Lock(), 0]
            _PATH_LOCKS[key] = entry
        entry[1] += 1
    lock = entry[0]
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _PATH_LOCKS_GUARD:
            entry[1] -= 1
            if entry[1] == 0 and _PATH_LOCKS.get(key) is entry:
                del _PATH_LOCKS[key]


class RemoteFileCache:
    """Manage account-scoped remote files under a bounded local cache root.

    Cache fills for the same path are serialized within a process and written
    to a staging file before atomic publication. The byte limit is soft when
    the protected file alone exceeds the budget; that file remains available.
    """

    def __init__(self, userid, root=None, max_bytes=None):
        identity = str(userid or "anonymous").encode("utf-8")
        account_key = hashlib.sha256(identity).hexdigest()[:24]
        cache_root = (
            Path(root)
            if root is not None
            else Path(tempfile.gettempdir()) / "abraia-cache"
        )
        self.root = cache_root / account_key
        if max_bytes is None:
            configured = os.environ.get(CACHE_MAX_BYTES_ENV)
            max_bytes = (
                int(configured)
                if configured is not None
                else DEFAULT_CACHE_MAX_BYTES
            )
        self.max_bytes = max(0, int(max_bytes))

    def path_for(self, path):
        """Return the local path for a remote path, rejecting traversal."""
        relative = Path(os.fspath(path))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] in {".staging", ".metadata"}
        ):
            raise ValueError("Cached paths must remain relative to the cache directory")
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        return str(destination)

    def get_or_create(self, path, fetch, remote_info=None):
        """Return a cached file or atomically publish one fetched copy."""
        with _account_lock(self.root).shared():
            return self._get_or_create(
                path,
                fetch,
                remote_info=remote_info,
            )

    def _get_or_create(self, path, fetch, remote_info=None):
        destination = self.path_for(path)
        with _locked_path(destination):
            if os.path.isfile(destination):
                self.touch(destination)
                return destination

            staging = self.root / ".staging"
            staging.mkdir(parents=True, exist_ok=True)
            temporary_directory = tempfile.mkdtemp(
                prefix="download-", dir=str(staging)
            )
            temporary = os.path.join(temporary_directory, Path(path).name)
            try:
                fetch(temporary)
                if not os.path.isfile(temporary):
                    raise OSError("Cache fetch did not create a file")
                os.replace(temporary, destination)
            finally:
                shutil.rmtree(temporary_directory, ignore_errors=True)

            signature = _cache_signature(path, remote_info)
            if signature is not None and signature.get("exists") is not False:
                self._write_signature(path, signature)
            else:
                self._remove_signature(path)
            self.touch(destination)
            self._prune(protected=destination)
            return destination

    def store(self, path, data, remote_info=None):
        """Atomically seed the cache with uploaded or versioned bytes."""
        signature = _cache_signature(path, remote_info)
        if signature is not None and signature.get("exists") is False:
            return None

        destination = self.path_for(path)
        with _account_lock(self.root).shared():
            with _locked_path(destination):
                staging = self.root / ".staging"
                staging.mkdir(parents=True, exist_ok=True)
                temporary_directory = tempfile.mkdtemp(
                    prefix="store-", dir=str(staging)
                )
                temporary = os.path.join(
                    temporary_directory, Path(path).name
                )
                try:
                    with open(temporary, "wb") as target:
                        if isinstance(data, (bytes, bytearray, memoryview)):
                            target.write(data)
                        elif hasattr(data, "getvalue"):
                            target.write(data.getvalue())
                        elif isinstance(data, (str, os.PathLike)):
                            with open(data, "rb") as source:
                                shutil.copyfileobj(source, target)
                        else:
                            seekable = getattr(data, "seekable", lambda: False)()
                            position = data.tell() if seekable else None
                            if position is not None:
                                data.seek(0)
                            shutil.copyfileobj(data, target)
                            if position is not None:
                                data.seek(position)
                    expected_size = (
                        signature.get("size") if signature is not None else None
                    )
                    if (
                        expected_size is not None
                        and os.path.getsize(temporary) != int(expected_size)
                    ):
                        return None
                    os.replace(temporary, destination)
                    if signature is None:
                        self._remove_signature(path)
                    else:
                        self._write_signature(path, signature)
                finally:
                    shutil.rmtree(temporary_directory, ignore_errors=True)

                self.touch(destination)
                self._prune(protected=destination)
                return str(destination)

    def reconcile_listing(self, folder, files):
        """Invalidate cached files against a successful source-folder listing."""
        folder = Path(os.fspath(folder or "")).as_posix().strip("/")
        if folder == ".":
            folder = ""
        remote_files = {}
        for record in files:
            if not isinstance(record, dict):
                continue
            path = record.get("path")
            if not path:
                continue
            path = Path(os.fspath(path)).as_posix().strip("/")
            if Path(path).name.startswith("tb_"):
                continue
            remote_files[path] = record

        folder_path = Path(folder)
        if (
            folder_path.is_absolute()
            or ".." in folder_path.parts
            or (
                folder_path.parts
                and folder_path.parts[0] in {".metadata", ".staging"}
            )
        ):
            return
        cache_folder = self.root / folder_path
        cache_root = self.root.resolve()
        resolved_folder = cache_folder.resolve()
        if (
            resolved_folder != cache_root
            and cache_root not in resolved_folder.parents
        ):
            return
        if not cache_folder.is_dir():
            return

        cached_paths = []
        for entry in cache_folder.iterdir():
            if not entry.is_file():
                continue
            cached_paths.append(entry.relative_to(self.root).as_posix())

        for path in cached_paths:
            if Path(path).name.startswith("tb_"):
                continue

            source_info = remote_files.get(path)
            if source_info is None:
                self.invalidate_source(path)
                continue
            self._invalidate_if_changed(path, source_info)

    def _invalidate_if_changed(self, path, remote_info):
        expected = _remote_signature(remote_info)
        if expected is None:
            return False
        if self._read_signature(path) == expected:
            return False
        return self.invalidate_source(path)

    def invalidate_source(self, path):
        """Remove a source file and its dependent cached thumbnail."""
        removed = self.invalidate(path)
        return self.invalidate_thumbnail(path) or removed

    def invalidate_thumbnail(self, source_path):
        """Remove the cached thumbnail derived from a source file."""
        name = posixpath.basename(source_path)
        if not source_path or name.startswith("tb_"):
            return False
        thumbnail = posixpath.join(
            posixpath.dirname(source_path), f"tb_{name}"
        )
        return self.invalidate(thumbnail)

    def invalidate(self, path):
        """Remove one cached remote file if it exists."""
        with _account_lock(self.root).shared():
            destination = Path(self.path_for(path))
            with _locked_path(destination):
                try:
                    destination.unlink()
                except FileNotFoundError:
                    removed = False
                else:
                    removed = True
                self._remove_signature(path)
                return removed

    def clear(self):
        """Remove all cached files for this account and return the file count."""
        with _account_lock(self.root).exclusive():
            if not self.root.exists():
                return 0
            count = sum(1 for _entry in self._data_files())
            shutil.rmtree(self.root)
            return count

    def _data_files(self):
        """Yield cached files, excluding metadata and incomplete downloads."""
        for entry in self.root.rglob("*"):
            try:
                parts = entry.relative_to(self.root).parts
                if (
                    entry.is_file()
                    and not {".metadata", ".staging"}.intersection(parts)
                ):
                    yield entry
            except OSError:
                continue

    def _signature_path(self, path):
        relative = Path(os.fspath(path)).as_posix()
        digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        directory = self.root / ".metadata"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}.json"

    def _read_signature(self, path):
        signature_path = self._signature_path(path)
        try:
            with signature_path.open("r", encoding="utf-8") as source:
                return json.load(source)
        except (OSError, ValueError):
            return None

    def _write_signature(self, path, signature):
        signature_path = self._signature_path(path)
        descriptor, temporary = tempfile.mkstemp(
            prefix="signature-", suffix=".tmp", dir=str(signature_path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                json.dump(signature, target, sort_keys=True)
            os.replace(temporary, signature_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _remove_signature(self, path):
        try:
            self._signature_path(path).unlink()
        except FileNotFoundError:
            pass

    def touch(self, path):
        """Mark a cached file as recently used for least-recently-used pruning."""
        try:
            stat = os.stat(path)
            os.utime(path, (time.time(), stat.st_mtime))
        except OSError:
            return False
        return True

    def prune(self, protected=None):
        """Evict least-recently-used files until the account cache fits."""
        with _account_lock(self.root).shared():
            return self._prune(protected=protected)

    def _prune(self, protected=None):
        if not self.root.exists():
            return 0
        protected_path = Path(protected).resolve() if protected else None
        files = []
        total = 0
        for entry in self._data_files():
            try:
                stat = entry.stat()
            except OSError:
                continue
            total += stat.st_size
            files.append((stat.st_atime_ns, entry, stat.st_size))
        removed = 0
        for _accessed, entry, size in sorted(files):
            if total <= self.max_bytes:
                break
            try:
                if protected_path is not None and entry.resolve() == protected_path:
                    continue
                entry.unlink()
            except OSError:
                continue
            relative = entry.relative_to(self.root).as_posix()
            if not relative.startswith(".metadata/"):
                self._remove_signature(relative)
            total -= size
            removed += 1
        return removed


def _cache_signature(path, remote_info):
    if Path(os.fspath(path)).name.startswith("tb_"):
        return None
    return _remote_signature(remote_info)


def _remote_signature(remote_info):
    """Normalize size/date listing fields into a stable cache fingerprint."""
    if remote_info is None:
        return None
    if remote_info.get("_missing"):
        return {"exists": False}
    size = remote_info.get("size")
    date = remote_info.get("date")
    if hasattr(date, "isoformat"):
        date = date.isoformat()
    if size is None and date is None:
        return None
    return {"exists": True, "size": size, "date": date}
