from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace

from abraia.utils.concurrency import RemoteRequestScheduler, bounded_map
from abraia.utils.remote import load_url, request_with_retries


def test_request_scheduler_caps_concurrent_work():
    scheduler = RemoteRequestScheduler(max_concurrent=2)
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak = 0

    def work():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _value: scheduler.run(work), range(4)))

    assert results == [True] * 4
    assert peak == 2


def test_bounded_map_preserves_order_and_caps_submissions():
    class CompletedFuture:
        def __init__(self, executor, value):
            self.executor = executor
            self.value = value

        def result(self):
            self.executor.pending -= 1
            return self.value

    class RecordingExecutor:
        def __init__(self):
            self.pending = 0
            self.peak = 0

        def submit(self, function, value):
            self.pending += 1
            self.peak = max(self.peak, self.pending)
            return CompletedFuture(self, function(value))

    executor = RecordingExecutor()
    values = list(bounded_map(executor, lambda value: value * 2, range(12), max_pending=3))

    assert values == [value * 2 for value in range(12)]
    assert executor.peak == 3


def test_request_with_retries_uses_the_supplied_shared_scheduler():
    class Scheduler:
        def __init__(self):
            self.calls = 0

        def run(self, function):
            self.calls += 1
            return function()

    class Session:
        def request(self, method, url, **kwargs):
            return method, url, kwargs

    scheduler = Scheduler()
    result = request_with_retries(
        Session(), "GET", "/resource", retries=1, scheduler=scheduler
    )

    assert result[:2] == ("GET", "/resource")
    assert scheduler.calls == 1



def test_load_url_holds_the_limiter_through_the_response_body():
    from unittest.mock import patch

    class Scheduler:
        active = False

        def run(self, function):
            self.active = True
            try:
                return function()
            finally:
                self.active = False

    class Response:
        status_code = 200

        def __init__(self, scheduler):
            self.scheduler = scheduler

        @property
        def content(self):
            assert self.scheduler.active
            return b"image bytes"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    scheduler = Scheduler()
    session = type("Session", (), {})()
    session.get = lambda *_args, **_kwargs: Response(scheduler)
    with patch("abraia.utils.remote._get_url_session", return_value=session):
        stream = load_url("https://example.invalid/image.jpg", scheduler=scheduler)

    assert stream.read() == b"image bytes"
    stream.close()


def test_request_scheduler_spaces_request_starts():
    from time import monotonic

    scheduler = RemoteRequestScheduler(max_concurrent=2, min_interval=0.005)
    starts = []
    lock = Lock()

    def work():
        with lock:
            starts.append(monotonic())

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _value: scheduler.run(work), range(6)))

    assert all(
        later - earlier >= 0.004
        for earlier, later in zip(starts, starts[1:])
    )


def test_request_scheduler_preserves_spacing_after_delayed_wakeup(monkeypatch):
    now = 0.0

    def delayed_sleep(delay):
        nonlocal now
        now += delay + 0.01

    monkeypatch.setattr(
        "abraia.utils.concurrency.time",
        SimpleNamespace(monotonic=lambda: now, sleep=delayed_sleep),
    )
    scheduler = RemoteRequestScheduler(max_concurrent=2, min_interval=0.005)
    starts = []

    for _ in range(3):
        scheduler.run(lambda: starts.append(now))

    assert all(
        later - earlier >= scheduler.min_interval
        for earlier, later in zip(starts, starts[1:])
    )
