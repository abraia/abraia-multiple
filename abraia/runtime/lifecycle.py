"""Shared cancellation and close semantics for runtime components."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable, Optional


logger = logging.getLogger(__name__)


def stop_resource(resource) -> None:
    """Request a prompt stop from a source using its public lifecycle API."""
    if resource is None:
        return
    stop = getattr(resource, "stop", None)
    if callable(stop):
        stop()
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def close_resources(resources: Iterable) -> None:
    """Close unique resources without masking the active pipeline error."""
    seen = set()
    for resource in resources:
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception:
            logger.warning("Failed to close runtime resource", exc_info=True)


class CancellationWatcher:
    """Run one cancellation callback while a blocking operation is active."""

    def __init__(
        self,
        is_cancelled: Callable[[], bool],
        on_cancel: Callable[[], None],
        stop_event: Optional[threading.Event] = None,
        interval: float = 0.05,
    ):
        self.is_cancelled = is_cancelled
        self.on_cancel = on_cancel
        self.stop_event = stop_event or threading.Event()
        self.interval = interval
        self._thread = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self.stop_event.wait(self.interval):
            if self.is_cancelled():
                self.on_cancel()
                return

    def close(self, timeout: float = 0.2) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


__all__ = ["CancellationWatcher", "close_resources", "stop_resource"]
