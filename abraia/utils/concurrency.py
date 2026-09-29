"""Shared limits for remote requests and bounded worker submissions."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
import threading
import time

DEFAULT_MAX_REMOTE_REQUESTS = 4
DEFAULT_MAX_REMOTE_WORKERS = 4


class RemoteRequestScheduler:
    """Limit simultaneous remote request work across a client session."""

    def __init__(self, max_concurrent=DEFAULT_MAX_REMOTE_REQUESTS, min_interval=0.0):
        self.max_concurrent = int(max_concurrent)
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least one")
        self.min_interval = float(min_interval)
        if self.min_interval < 0:
            raise ValueError("min_interval cannot be negative")
        self._slots = threading.BoundedSemaphore(self.max_concurrent)
        self._start_lock = threading.Lock()
        self._next_start = 0.0

    def run(self, function, *args, **kwargs):
        """Run one remote operation while holding a shared capacity slot."""
        with self._slots:
            with self._start_lock:
                now = time.monotonic()
                delay = max(0.0, self._next_start - now)
                self._next_start = max(now, self._next_start) + self.min_interval
            if delay:
                time.sleep(delay)
            return function(*args, **kwargs)


_DEFAULT_REMOTE_REQUEST_SCHEDULER = RemoteRequestScheduler()


def get_default_remote_request_scheduler():
    """Return the process-wide limiter used by remote transports."""
    return _DEFAULT_REMOTE_REQUEST_SCHEDULER


def bounded_map(executor, function, iterable, *, max_pending=None):
    """Yield mapped results in order while bounding queued futures."""
    if max_pending is None:
        max_pending = DEFAULT_MAX_REMOTE_WORKERS * 2
    max_pending = max(1, int(max_pending))
    iterator = iter(iterable)
    pending = deque()
    for _ in range(max_pending):
        try:
            value = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(function, value))
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            value = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(function, value))


def bounded_as_completed(executor, function, iterable, *, max_pending=None):
    """Yield ``(input, result)`` pairs as bounded tasks finish.

    Unlike :func:`bounded_map`, this preserves completion order so callers can
    report progress promptly even when an earlier task is slow.
    """
    if max_pending is None:
        max_pending = DEFAULT_MAX_REMOTE_WORKERS * 2
    max_pending = max(1, int(max_pending))
    iterator = iter(iterable)
    pending = {}

    def submit_next():
        try:
            value = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(function, value)] = value
        return True

    for _ in range(max_pending):
        if not submit_next():
            break
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            value = pending.pop(future)
            result = future.result()
            submit_next()
            yield value, result
__all__ = [
    "DEFAULT_MAX_REMOTE_REQUESTS",
    "DEFAULT_MAX_REMOTE_WORKERS",
    "RemoteRequestScheduler",
    "bounded_as_completed",
    "bounded_map",
    "get_default_remote_request_scheduler",
]
