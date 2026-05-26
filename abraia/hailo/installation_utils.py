from __future__ import annotations
"""Installation-related utilities.

This module provides utilities for detecting system configuration, package versions,
and managing installation-related tasks for Hailo applications.

Function Organization:
    - Core Detection: Architecture and package detection (used by multiple modules)
    - Version Detection: Functions to detect installed package versions
"""

import platform
import shlex
import subprocess

from .defines import (
    ARM_NAME_I,
    ARM_POSSIBLE_NAME,
    HAILO8_ARCH,
    HAILO8_ARCH_CAPS,
    HAILO8L_ARCH,
    HAILO8L_ARCH_CAPS,
    HAILO10H_ARCH,
    HAILO10H_ARCH_CAPS,
    HAILO15H_ARCH_CAPS,
    HAILO_FW_CONTROL_CMD,
    HAILORT_PACKAGE_NAME,
    HAILORT_PACKAGE_NAME_RPI,
    LINUX_SYSTEM_NAME_I,
    RPI_NAME_I,
    RPI_POSSIBLE_NAME,
    UNKNOWN_NAME_I,
    X86_NAME_I,
    X86_POSSIBLE_NAME,
)
from .hailo_logger import get_logger

hailo_logger = get_logger(__name__)


# =============================================================================
# Core Detection Functions
# =============================================================================

def is_raspberry_pi():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return  RPI_POSSIBLE_NAME in model
    except:
        return False

def detect_host_arch() -> str:
    """Detect the host system architecture.

    Returns:
        str: One of 'x86', 'rpi', 'arm', or 'unknown'
    """
    hailo_logger.debug("Detecting host architecture.")
    machine_name = platform.machine().lower()
    system_name = platform.system().lower()
    hailo_logger.debug(f"Machine: {machine_name}, System: {system_name}")

    if machine_name in X86_POSSIBLE_NAME:
        hailo_logger.info("Detected host architecture: x86")
        return X86_NAME_I
    if machine_name in ARM_POSSIBLE_NAME:
        if system_name == LINUX_SYSTEM_NAME_I and is_raspberry_pi():
            hailo_logger.info("Detected host architecture: Raspberry Pi")
            return RPI_NAME_I
        hailo_logger.info("Detected host architecture: ARM")
        return ARM_NAME_I
    hailo_logger.warning("Unknown host architecture.")
    return UNKNOWN_NAME_I


def detect_hailo_arch() -> str | None:
    """Detect the connected Hailo device architecture.

    Returns:
        str | None: One of 'hailo8', 'hailo8l', 'hailo10h', or None if detection fails
    """
    hailo_logger.debug("Detecting Hailo architecture using hailortcli.")
    try:
        args = shlex.split(HAILO_FW_CONTROL_CMD)
        res = subprocess.run(args, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            hailo_logger.error(f"hailortcli failed with code {res.returncode}")
            return None
        for line in res.stdout.splitlines():
            if HAILO8L_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO8L")
                return HAILO8L_ARCH
            if HAILO8_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO8")
                return HAILO8_ARCH
            if HAILO10H_ARCH_CAPS in line or HAILO15H_ARCH_CAPS in line:
                hailo_logger.debug("Detected Hailo architecture: HAILO10H")
                return HAILO10H_ARCH
    except Exception as e:
        hailo_logger.exception(f"Error detecting Hailo architecture: {e}")
        assert False, "Error detecting Hailo architecture. Is Hailo Installed?"
    hailo_logger.warning("Could not determine Hailo architecture.")
    assert False, "Could not determine Hailo architecture. Is Hailo connected?"


def detect_system_pkg_version(pkg_name: str) -> str:
    """Detect the version of a system package (dpkg).

    Args:
        pkg_name: Name of the system package

    Returns:
        str: Version string or empty string if not found
    """
    hailo_logger.debug(f"Detecting system package version for: {pkg_name}")
    try:
        version = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${Version}", pkg_name], stderr=subprocess.DEVNULL, text=True
        )
        version = version.strip()
        hailo_logger.debug(f"Found version {version} for system package {pkg_name}")
        return version
    except subprocess.CalledProcessError:
        hailo_logger.warning(f"System package {pkg_name} is not installed.")
        return ""


def detect_pkg_installed(pkg_name: str) -> bool:
    """Check if a system package is installed.

    Args:
        pkg_name: Name of the system package

    Returns:
        bool: True if installed, False otherwise
    """
    hailo_logger.debug(f"Checking if system package is installed: {pkg_name}")
    try:
        subprocess.check_output(["dpkg", "-s", pkg_name])
        hailo_logger.debug(f"Package {pkg_name} is installed.")
        return True
    except subprocess.CalledProcessError:
        hailo_logger.debug(f"Package {pkg_name} is not installed.")
        return False


def get_hailort_package_name() -> str:
    """Get the appropriate HailoRT package name based on host architecture.

    Returns:
        str: Package name ('hailort' or 'h10-hailort' for RPI)
    """
    host_arch = detect_host_arch()

    if host_arch == RPI_NAME_I:
        if detect_hailo_arch() == HAILO10H_ARCH:
            # Old hailort version used h10-hailort
            if detect_system_pkg_version(HAILORT_PACKAGE_NAME_RPI):
                hailo_logger.debug(
                    f"Using RPI-specific HailoRT package: {HAILORT_PACKAGE_NAME_RPI}"
                )
                return HAILORT_PACKAGE_NAME_RPI

    hailo_logger.debug(f"Using default HailoRT package: {HAILORT_PACKAGE_NAME}")
    return HAILORT_PACKAGE_NAME
