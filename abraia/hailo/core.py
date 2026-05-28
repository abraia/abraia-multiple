from __future__ import annotations
"""Core helpers: arch detection, buffer utils, model resolution."""

import os
import sys
import yaml
import shlex
import requests
import subprocess

from pathlib import Path
from typing import Any, Optional, Tuple, Dict

from ..utils import download_url

import logging
logger = logging.getLogger(__name__)

# Base Defaults
HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
HAILO10H_ARCH = "hailo10h"
HAILO_FILE_EXTENSION = ".hef"
HAILO_MODEL_ZOO_DEFAULT_VERSION = "v2.17.0"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"
RESOURCES_MODELS_DIR_NAME = "models"

# Core defaults
RPI_POSSIBLE_NAME = "Raspberry Pi"
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
USB_CAMERA = "usb"

# Queue and async inference defaults
MAX_INPUT_QUEUE_SIZE = 60
MAX_OUTPUT_QUEUE_SIZE = 60
MAX_ASYNC_INFER_JOBS = 20

# Image / camera defaults
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}

# Base project paths
DEFAULT_COCO_LABELS_PATH = str(Path(__file__).parent / "coco.txt")


def _get_config_path(filename: str) -> str:
    """Get absolute path to a configuration file within the package."""
    return str(Path(__file__).parent / filename)


DEFAULT_RESOURCES_CONFIG_PATH = _get_config_path("resources_config.yaml")


def load_config(path: Path | str) -> dict:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _extract_model_entries(entries: Any, app_type_filter: Optional[str] = None) -> list[dict]:
    """Extract model entry dictionaries from config entries."""
    entries_list = entries if isinstance(entries, list) else [entries]
    models = []

    for entry in entries_list:
        if isinstance(entry, dict):
            name = entry.get("name")
            if name is not None and not (isinstance(name, str) and name.lower() == "none"):
                # Parse app_type field - default to both if not specified
                app_type_raw = entry.get("app_type", ["pipeline", "standalone"])
                if isinstance(app_type_raw, str):
                    app_type = (app_type_raw,)
                else:
                    app_type = tuple(app_type_raw) if app_type_raw else ("pipeline", "standalone")
                
                model = {
                    "name": name,
                    "source": entry.get("source", "mz"),
                    "url": entry.get("url"),
                    "app_type": app_type,
                }
                
                # Filter by app_type if requested
                if app_type_filter is None or app_type_filter in model["app_type"]:
                    models.append(model)
        elif isinstance(entry, str) and entry.lower() != "none":
            model = {
                "name": entry,
                "source": "mz",
                "url": None,
                "app_type": ("pipeline", "standalone"),
            }
            # String entries default to both types, so always include unless filtered out
            if app_type_filter is None or app_type_filter in model["app_type"]:
                models.append(model)

    return models


def get_default_model_name(app_name: str, arch: str, app_type: Optional[str] = None) -> Optional[str]:
    """Get the first default model name for an app and architecture."""
    config = load_config(DEFAULT_RESOURCES_CONFIG_PATH)
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    models = _extract_model_entries(arch_models.get("default"), app_type_filter=app_type)
    return models[0]["name"] if models else None


DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get Content-Length of a remote file via HEAD request."""
    try:
        r = requests.head(url, headers={'User-Agent': DEFAULT_USER_AGENT}, timeout=timeout, allow_redirects=True)
        length = r.headers.get('Content-Length')
        return int(length) if length else None
    except Exception:
        return None


def get_model_url(app_name, model_name, hailo_arch):
    """Return a download task tuple for a specific app model, or None if not found."""
    config = load_config(DEFAULT_RESOURCES_CONFIG_PATH)
    app_cfg = config.get(app_name, {}).get("models", {}).get(hailo_arch, {})
    for tier in ["default", "extra"]:
        entries = app_cfg.get(tier, [])
        entries_list = entries if isinstance(entries, list) else [entries]
        for e in entries_list:
            name = e.get("name") if isinstance(e, dict) else {}
            if name == model_name:
                url = e.get("url")
                if not url:
                    s3_arch = "h8l" if hailo_arch == HAILO8L_ARCH else "h8"
                    source = e.get("source", "mz")
                    if source == "s3":
                        url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
                    elif source == "mz":
                        url = f"{MODEL_ZOO_URL}/{HAILO_MODEL_ZOO_DEFAULT_VERSION}/{hailo_arch}/{name}{HAILO_FILE_EXTENSION}"
                if url:
                    dest_name = name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION
                    dest = Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_MODELS_DIR_NAME / hailo_arch / dest_name
                    return url, dest
    logger.warning(f"Model '{model_name}' not found for app '{app_name}'")
    return None, None


def execute_download(url, dest_path):
    """Execute a single download task."""
    remote_size = get_remote_file_size(url)

    if dest_path.exists() and remote_size and dest_path.stat().st_size == remote_size:
        logger.info("File already exists and valid")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading: {url}")
        download_url(url, str(dest_path))
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        logger.warning(f"Failed to download {url}: {e}")


def get_resource_path(resource_type: str, name: str, arch: str | None = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    
    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch:
            raise RuntimeError("Could not detect Hailo architecture for model path.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)
        
    return root / resource_type / name


def resolve_hef_path(hef_path: Optional[str], app_name: str, arch: Optional[str] = None, 
                     app_type: Optional[str] = "standalone") -> Optional[Path]:
    """Resolve HEF path, downloading it if necessary."""
    arch = arch or detect_hailo_arch()
    if not arch:
        raise RuntimeError("Could not detect Hailo architecture.")
    
    if hef_path is None:
        default_model = get_default_model_name(app_name, arch, app_type=app_type)
        if not default_model:
            logger.error(f"No default model found for {app_name}/{arch}")
            return None
        hef_path = default_model
        logger.info(f"Using default model: {default_model}")

    # Case 1: Existing local path
    path = Path(hef_path)
    if path.exists():
        return path.resolve()
    
    if not hef_path.endswith(HAILO_FILE_EXTENSION):
        path_with_ext = path.with_suffix(HAILO_FILE_EXTENSION)
        if path_with_ext.exists():
            return path_with_ext.resolve()

    # Case 2: Known model in resources
    model_name = path.stem
    resource_path = get_resource_path(RESOURCES_MODELS_DIR_NAME, model_name, arch)
    if resource_path.exists():
        return resource_path

    # Case 3: Missing model - download if known
    logger.warning(f"\n⚠️  WARNING: Model '{model_name}' not found. Downloading for {app_name}/{arch}...")
    try:
        url, dest = get_model_url(app_name, model_name, arch)
        if url and dest:
            execute_download(url, dest)
            if resource_path.exists():
                return resource_path
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")

    logger.error(f"Model '{model_name}' not found.")
    return None


def resolve_output_resolution_arg(res_arg: Optional[list[str]]) -> Optional[Tuple[int, int]]:
    """Parse --output-resolution argument."""
    if not res_arg:
        return None

    if len(res_arg) == 1:
        key = res_arg[0]
        if key in CAMERA_RESOLUTION_MAP:
            return CAMERA_RESOLUTION_MAP[key]
        raise ValueError(f"Invalid resolution preset: {key}")

    if len(res_arg) == 2 and all(x.isdigit() for x in res_arg):
        w, h = map(int, res_arg)
        if w > 0 and h > 0:
            return (w, h)

    raise ValueError(f"Invalid resolution argument: {res_arg}")


def handle_and_resolve_args(args: Any, app_name: str) -> None:
    """Common CLI argument resolver for standalone apps."""
    args.hef_path = resolve_hef_path(args.hef_path, app_name)

    if hasattr(args, "output_dir") and args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(args, "output_resolution"):
        args.output_resolution = resolve_output_resolution_arg(args.output_resolution)


# =============================================================================
# Hardware Detection
# =============================================================================

def is_raspberry_pi() -> bool:
    """Check if the current host is a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return RPI_POSSIBLE_NAME in f.read()
    except Exception:
        return False


def detect_hailo_arch() -> Optional[str]:
    """Detect the connected Hailo device architecture."""
    logger.debug("Detecting Hailo architecture...")
    try:
        res = subprocess.run(shlex.split(HAILO_FW_CONTROL_CMD), capture_output=True, text=True)
        if res.returncode != 0:
            return None
        
        stdout = res.stdout.upper()
        if HAILO8L_ARCH_CAPS in stdout:
            return HAILO8L_ARCH
        if HAILO8_ARCH_CAPS in stdout:
            return HAILO8_ARCH
        if HAILO10H_ARCH_CAPS in stdout or HAILO15H_ARCH_CAPS in stdout:
            return HAILO10H_ARCH
    except Exception as e:
        logger.error(f"Error detecting Hailo architecture: {e}")
        
    logger.warning("Could not determine Hailo architecture.")
    return None
