"""Shared ONNX Runtime session helpers."""

import logging
import os

import onnxruntime as ort

from ..utils import get_providers

logger = logging.getLogger(__name__)


def create_onnx_session(path, providers=None):
    """Create an ONNX Runtime session with the SDK provider policy."""
    providers = providers or get_providers()
    try:
        return ort.InferenceSession(os.fspath(path), providers=providers)
    except Exception:
        cpu_provider = "CPUExecutionProvider"
        if cpu_provider not in providers or providers == [cpu_provider]:
            raise
        logger.warning(
            "ONNX providers %s failed for %s; retrying on CPU",
            providers,
            path,
            exc_info=True,
        )
        return ort.InferenceSession(os.fspath(path), providers=[cpu_provider])


def close_session(session):
    """Close an ONNX Runtime session when the backend exposes ``close``."""
    close = getattr(session, "close", None)
    if callable(close):
        close()


def close_resource(resource):
    """Close a model component or a raw backend session."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()
    else:
        close_session(resource)


__all__ = ["close_resource", "close_session", "create_onnx_session"]
