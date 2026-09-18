"""Shared ONNX Runtime session helpers."""

import logging
import os

from ..utils import get_providers
from .accelerators import (
    available_accelerators,
    onnx_providers,
)

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


def create_onnx_session(path, providers=None, accelerator=None):
    """Create an ONNX Runtime session with the SDK provider policy."""
    import onnxruntime as ort

    if providers is None:
        providers = onnx_providers(accelerator)
    if providers is None:
        providers = get_providers()
    providers = list(providers or ["CPUExecutionProvider"])
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


class ResourceGroup:
    """Own a set of closeable resources and release them in reverse order.

    Composite models use this helper while they are being constructed so a
    failure in a later component cannot leak sessions created earlier. The
    group is intentionally backend-agnostic; resources only need a ``close``
    method or to be raw ONNX Runtime sessions.
    """

    def __init__(self, resources=()):
        self._resources = []
        self._closed = False
        for resource in resources:
            self.add(resource)

    def add(self, resource):
        """Register and return ``resource`` for convenient construction."""
        if resource is None:
            return None
        if self._closed:
            close_resource(resource)
            raise RuntimeError("Resource group has already been closed")
        if not any(existing is resource for existing in self._resources):
            self._resources.append(resource)
        return resource

    def close(self, suppress_errors=False):
        """Close all resources once, optionally suppressing cleanup errors."""
        if self._closed:
            return
        self._closed = True
        resources, self._resources = self._resources, []
        first_error = None
        for resource in reversed(resources):
            try:
                close_resource(resource)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(suppress_errors=exc_type is not None)


class OnnxSessionBundle:
    """Own multiple ONNX Runtime sessions with shared provider policy."""

    def __init__(self, paths, providers=None, accelerator=None):
        self._resources = ResourceGroup()
        self.sessions = []
        try:
            self.sessions = [
                self._resources.add(
                    create_onnx_session(
                        path,
                        providers=providers,
                        accelerator=accelerator,
                    )
                )
                for path in paths
            ]
        except Exception:
            self._resources.close(suppress_errors=True)
            raise
        self._session_map = {}
        if hasattr(paths, "keys"):
            self._session_map = dict(zip(paths.keys(), self.sessions))
        provider_names = []
        for session in self.sessions:
            for provider in session.get_providers():
                if provider not in provider_names:
                    provider_names.append(provider)
        self.execution_providers = tuple(provider_names)
        self.accelerator = accelerator_from_providers(self.execution_providers)
        self._closed = False

    def __getitem__(self, key):
        """Return a session by index or by name when named paths were used."""
        if isinstance(key, int):
            return self.sessions[key]
        return self._session_map[key]

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._resources.close()
        finally:
            self.sessions = []
            self._session_map = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OnnxSessionMixin:
    """Common lifecycle implementation for inference classes with one session."""

    def _init_onnx_session(self, path, providers=None, accelerator=None):
        resources = ResourceGroup()
        try:
            self.session = resources.add(
                create_onnx_session(
                    path,
                    providers=providers,
                    accelerator=accelerator,
                )
            )
            self.execution_providers = tuple(self.session.get_providers())
            self.accelerator = accelerator_from_providers(
                self.execution_providers
            )
        except Exception:
            resources.close(suppress_errors=True)
            raise
        self._resources = resources
        self._closed = False

    def _ensure_open(self):
        if getattr(self, "_closed", False):
            raise RuntimeError("Inference model has already been closed")

    def close(self):
        if getattr(self, "_closed", False):
            return
        resources = getattr(self, "_resources", None)
        self._resources = None
        self.session = None
        self._closed = True
        if resources is not None:
            resources.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "OnnxSessionBundle",
    "OnnxSessionMixin",
    "ResourceGroup",
    "available_accelerators",
    "accelerator_from_providers",
    "close_resource",
    "close_session",
    "create_onnx_session",
    "get_model_accelerator",
]
