"""Shared ONNX Runtime session helpers."""

import logging
import os

from ..utils import get_providers

logger = logging.getLogger(__name__)


def accelerator_from_providers(providers):
    """Return a user-facing accelerator label for ONNX providers.

    ONNX Runtime may expose several providers for one session, with CPU as a
    fallback.  The first recognized non-CPU provider is the accelerator used
    for the session.  CoreML can select the Apple GPU or Neural Engine at
    runtime, so it is reported as both hardware-accelerated options.
    """
    for provider in providers or ():
        name = str(provider).upper()
        if "HAILO" in name:
            return "HAILO"
        if any(marker in name for marker in (
            "QNN", "NNAPI", "VITISAI", "ARMNN", "NEURO", "NPU",
        )):
            return "NPU"
        if "COREML" in name:
            return "GPU/NPU"
        if any(marker in name for marker in (
            "CUDA", "TENSORRT", "ROCM", "DIRECTML", "MIGRAPHX",
        )):
            return "GPU"
    return "CPU"


def get_model_accelerator(model):
    """Return the accelerator label exposed by an inference model."""
    accelerator = getattr(model, "accelerator", None)
    if accelerator:
        return str(accelerator)
    providers = getattr(model, "execution_providers", None)
    if providers is None:
        session = getattr(model, "session", None)
        get_providers = getattr(session, "get_providers", None)
        providers = get_providers() if callable(get_providers) else ()
    return accelerator_from_providers(providers)


def create_onnx_session(path, providers=None):
    """Create an ONNX Runtime session with the SDK provider policy."""
    import onnxruntime as ort

    providers = list(providers or get_providers())
    try:
        return ort.InferenceSession(os.fspath(path), providers=providers)
    except Exception as exc:
        cpu_provider = "CPUExecutionProvider"
        if cpu_provider not in providers or providers == [cpu_provider]:
            raise
        message = str(exc).lower()
        provider_failure = any(
            marker in message
            for marker in (
                "execution provider",
                "provider",
                "cuda",
                "coreml",
                "tensorrt",
                "error compiling model",
                "working directory",
            )
        )
        if not provider_failure:
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


class OnnxSessionBundle:
    """Own multiple ONNX Runtime sessions with shared provider policy."""

    def __init__(self, paths, providers=None):
        self.sessions = []
        try:
            self.sessions = [
                create_onnx_session(path, providers=providers)
                for path in paths
            ]
        except Exception:
            for session in self.sessions:
                close_session(session)
            raise
        provider_names = []
        for session in self.sessions:
            for provider in session.get_providers():
                if provider not in provider_names:
                    provider_names.append(provider)
        self.execution_providers = tuple(provider_names)
        self.accelerator = accelerator_from_providers(self.execution_providers)
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        sessions, self.sessions = self.sessions, []
        for session in sessions:
            close_session(session)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OnnxSessionMixin:
    """Common lifecycle implementation for inference classes with one session."""

    def _init_onnx_session(self, path, providers=None):
        self.session = create_onnx_session(path, providers=providers)
        self.execution_providers = tuple(self.session.get_providers())
        self.accelerator = accelerator_from_providers(self.execution_providers)
        self._closed = False

    def _ensure_open(self):
        if getattr(self, "_closed", False):
            raise RuntimeError("Inference model has already been closed")

    def close(self):
        if getattr(self, "_closed", False):
            return
        session, self.session = getattr(self, "session", None), None
        self._closed = True
        close_session(session)


__all__ = [
    "OnnxSessionBundle",
    "OnnxSessionMixin",
    "accelerator_from_providers",
    "close_resource",
    "close_session",
    "create_onnx_session",
    "get_model_accelerator",
]
