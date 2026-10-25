"""Shared accelerator capability and provider selection helpers."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..utils.remote import is_managed_model_path, is_url, probe_artifact


logger = logging.getLogger(__name__)

HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
HAILO10H_ARCH = "hailo10h"

HAILO_ARCHS = {
    "HAILO8L": HAILO8L_ARCH,
    "HAILO8": HAILO8_ARCH,
    "HAILO10H": HAILO10H_ARCH,
    "HAILO15H": HAILO10H_ARCH,
}

HAILO_CLI_IDENTIFY_COMMAND = ("hailortcli", "fw-control", "identify")


def detect_hailo_arch() -> Optional[str]:
    """Return the architecture reported by the connected Hailo device.

    ``None`` means that the CLI is unavailable, the command failed, or its
    output did not contain a known architecture.  Resource resolution decides
    whether that should become an exception for the calling operation.
    """
    try:
        result = subprocess.run(
            HAILO_CLI_IDENTIFY_COMMAND,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Error detecting Hailo architecture: %s", exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Hailo CLI architecture detection failed with exit code %s: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return None

    # hailortcli versions format this as either ``HAILO8`` or ``HAILO-8``.
    # Compare a compact representation so both forms select the same target.
    output = re.sub(r"[^A-Z0-9]", "", result.stdout.upper())
    for marker, architecture in HAILO_ARCHS.items():
        if marker in output:
            return architecture
    return None


CPU_PROVIDER = "CPUExecutionProvider"
GPU_PROVIDER_MARKERS = (
    "CUDA",
    "TENSORRT",
    "ROCM",
    "DIRECTML",
    "MIGRAPHX",
    "COREML",
)


def get_providers():
    """Return ONNX Runtime providers in preferred order."""
    import onnxruntime as ort

    preferred = (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    available = set(ort.get_available_providers())
    return [provider for provider in preferred if provider in available]


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
    from .hailo.toolbox import HAILO_AVAILABLE

    if not HAILO_AVAILABLE:
        return None
    return detect_hailo_arch()


def hailo_model_available(config, architecture):
    """Return whether an explicit Hailo model artifact is available."""
    model = next(
        (
            step.get("model", {})
            for step in config.get("steps", [])
            if step.get("type") == "model"
            and "input" not in step
        ),
        {},
    )
    uri = model.get("uri")
    if not uri:
        return False

    path = os.fspath(uri)
    if (os.path.isfile(path) or os.path.isdir(path)) and not is_managed_model_path(path):
        return True
    if is_url(path):
        return False
    if os.path.dirname(path):
        return probe_artifact(path).available

    if not architecture or not path.lower().endswith(".hef"):
        return False
    return probe_artifact(path).available


def paired_hailo_uri(onnx_uri, architecture=None):
    """Return the exact-stem Hailo URI paired with an ONNX model URI.

    Abraia-managed Hailo models are addressed by their remote path and
    resolved to a cached HEF at runtime. When a sibling HEF exists beside a
    local ONNX model, preserve that explicit local path instead.
    """
    if not onnx_uri:
        return None
    value = os.fspath(onnx_uri)
    if is_url(value):
        return None

    path = Path(value)
    siblings = []
    if architecture:
        siblings.append(
            path.with_name(f"{path.stem}_{architecture}.hef")
        )
    siblings.append(path.with_name(f"{path.stem}.hef"))
    for sibling in siblings:
        if sibling.is_file():
            return str(sibling)

    normalized_value = value.replace("\\", "/")
    if architecture and normalized_value.startswith("multiple/"):
        return f"{path.parent.as_posix()}/{path.stem}_{architecture}.hef"
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
    "HAILO8_ARCH",
    "HAILO8L_ARCH",
    "HAILO10H_ARCH",
    "HAILO_ARCHS",
    "HAILO_CLI_IDENTIFY_COMMAND",
    "detect_hailo_arch",
    "CPU_PROVIDER",
    "available_accelerators",
    "hailo_device_arch",
    "hailo_model_available",
    "is_gpu_provider",
    "normalize_accelerator",
    "onnx_providers",
    "paired_hailo_uri",
]
