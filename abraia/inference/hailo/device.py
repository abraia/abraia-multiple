"""Hailo device discovery helpers.

This module is intentionally independent from Hailo model loading and
postprocessing.  The Hailo CLI is only needed when a caller asks for
architecture-specific resource resolution.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Optional


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


__all__ = [
    "HAILO8_ARCH",
    "HAILO8L_ARCH",
    "HAILO10H_ARCH",
    "HAILO_ARCHS",
    "HAILO_CLI_IDENTIFY_COMMAND",
    "detect_hailo_arch",
]
