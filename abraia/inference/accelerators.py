"""Shared accelerator capability and provider selection helpers."""

from __future__ import annotations

import os
from pathlib import Path

from ..utils import get_providers
from ..utils.remote import ARTIFACT_RESOLVER, is_managed_model_path


CPU_PROVIDER = "CPUExecutionProvider"
GPU_PROVIDER_MARKERS = (
    "CUDA",
    "TENSORRT",
    "ROCM",
    "DIRECTML",
    "MIGRAPHX",
    "COREML",
)

def normalize_accelerator(value):
    """Normalize a requested accelerator name without selecting hardware."""
    value = str(value or "auto").strip().lower()
    aliases = {"default": "auto"}
    value = aliases.get(value, value)
    if value not in ("auto", "onnx", "cpu", "gpu", "hailo"):
        raise ValueError(
            "accelerator must be 'auto', 'onnx', 'cpu', 'gpu', or 'hailo'"
        )
    return value


def is_gpu_provider(provider):
    """Return whether an ONNX Runtime provider uses GPU hardware."""
    name = str(provider).upper()
    return any(marker in name for marker in GPU_PROVIDER_MARKERS)


def onnx_providers(accelerator=None):
    """Return ONNX providers for a request, always retaining CPU fallback.

    ``None`` preserves the existing SDK provider policy. Explicit ``cpu``
    forces CPU. Explicit ``gpu`` selects an available GPU provider and falls
    back to CPU when no GPU provider is installed.
    """
    if accelerator is None:
        return None

    accelerator = normalize_accelerator(accelerator)
    if accelerator in ("cpu", "hailo"):
        return [CPU_PROVIDER]

    try:
        available = list(get_providers())
    except Exception:
        available = []

    if accelerator == "gpu":
        providers = [provider for provider in available if is_gpu_provider(provider)]
        if not providers:
            return [CPU_PROVIDER]
        if CPU_PROVIDER not in providers:
            providers.append(CPU_PROVIDER)
        return providers

    if not available:
        return [CPU_PROVIDER]
    if CPU_PROVIDER not in available:
        available.append(CPU_PROVIDER)
    return available


def hailo_device_arch():
    """Return the connected Hailo architecture, or ``None``."""
    from .hailo.device import detect_hailo_arch
    from .hailo.toolbox import HAILO_AVAILABLE

    if not HAILO_AVAILABLE:
        return None
    return detect_hailo_arch()


def hailo_model_available(config, architecture):
    """Return whether an explicit Hailo model artifact is available."""
    model = config.get("model", {})
    uri = model.get("uri")
    if not uri:
        return False

    path = os.fspath(uri)
    if (os.path.isfile(path) or os.path.isdir(path)) and not is_managed_model_path(path):
        return True
    if path.lower().startswith(("http://", "https://")):
        return False
    if os.path.dirname(path):
        return ARTIFACT_RESOLVER.probe(path).available

    if not architecture or not path.lower().endswith(".hef"):
        return False
    return ARTIFACT_RESOLVER.probe(path).available


def paired_hailo_uri(onnx_uri, task, architecture=None):
    """Return the Hailo URI paired with an ONNX model URI.

    Abraia-managed Hailo models are addressed by their remote path and
    resolved to a cached HEF at runtime. When a sibling HEF exists beside a
    local ONNX model, preserve that explicit local path instead.
    """
    if not onnx_uri:
        return None
    value = os.fspath(onnx_uri)
    if value.lower().startswith(("http://", "https://")):
        return None

    path = Path(value)
    stem = path.stem
    if str(task).strip().lower() == "segmentation":
        stem = stem.replace("-seg", "_seg")

    siblings = []
    if architecture:
        siblings.append(path.with_name(f"{stem}_{architecture}.hef"))
    siblings.append(path.with_name(f"{stem}.hef"))
    for sibling in siblings:
        if sibling.is_file():
            return str(sibling)
    normalized_value = value.replace("\\", "/")
    if architecture and normalized_value.startswith("multiple/"):
        return f"{path.parent.as_posix()}/{stem}_{architecture}.hef"
    return None


def available_accelerators():
    """Return the accelerators currently visible to the SDK."""
    try:
        providers = get_providers()
    except Exception:
        providers = []
    return {
        "cpu": True,
        "gpu": any(is_gpu_provider(provider) for provider in providers),
        "hailo": hailo_device_arch() is not None,
    }


__all__ = [
    "CPU_PROVIDER",
    "available_accelerators",
    "hailo_device_arch",
    "hailo_model_available",
    "is_gpu_provider",
    "normalize_accelerator",
    "onnx_providers",
    "paired_hailo_uri",
]
