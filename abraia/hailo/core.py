from __future__ import annotations
"""Core helpers: arch detection, buffer utils, model resolution."""

import os
import sys
import time
import yaml
import shlex
import requests
import subprocess

from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# Core defaults
RPI_POSSIBLE_NAME = "Raspberry Pi"
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
USB_CAMERA = "usb"

# Resources directory structure
RESOURCES_MODELS_DIR_NAME = "models"
RESOURCES_VIDEOS_DIR_NAME = "videos"
RESOURCES_PHOTOS_DIR_NAME = "images"
RESOURCES_JSON_DIR_NAME = "json"
RESOURCES_NPY_DIR_NAME = "npy"

# Queue and async inference defaults
MAX_INPUT_QUEUE_SIZE = 60
MAX_OUTPUT_QUEUE_SIZE = 60
MAX_ASYNC_INFER_JOBS = 20

# Video format defaults
HAILO_RGB_VIDEO_FORMAT = "RGB"
HAILO_YUYV_VIDEO_FORMAT = "YUYV"
HAILO_NV12_VIDEO_FORMAT = "NV12"

# Image / camera defaults
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
CAMERA_RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}

# Resource types supported
RESOURCE_TYPE_MODEL = "model"
RESOURCE_TYPE_IMAGE = "image"
RESOURCE_TYPE_VIDEO = "video"
CAMERA_KEYWORDS = ["usb", "rpi"]
STANDALONE_SUFFIX = "_standalone"

# Base project paths
DEFAULT_COCO_LABELS_PATH = str(Path(__file__).parent / "coco.txt")


def _get_config_path(filename: str) -> str:
    """Get absolute path to a configuration file within the package."""
    return str(Path(__file__).parent / filename)


DEFAULT_RESOURCES_CONFIG_PATH = _get_config_path("resources_config.yaml")

# =============================================================================
# Exceptions & Type Definitions
# =============================================================================

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


@dataclass(frozen=True)
class ModelEntry:
    """Represents a model entry from resources config."""
    name: str
    source: str  # "mz" | "s3" | "gen-ai-mz"
    url: Optional[str] = None
    app_type: tuple[str, ...] = ("pipeline", "standalone")


# =============================================================================
# Configuration Loading Utilities
# =============================================================================

@lru_cache(maxsize=8)
def _load_config_cached(path: str) -> dict:
    """Internal helper to load and cache YAML configuration."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")


def load_config(path: Path | str, use_cache: bool = True) -> dict:
    """Load configuration from a YAML file."""
    if use_cache:
        return _load_config_cached(str(path))
    return _load_config_cached.__wrapped__(str(path))


def get_resources_config(use_cache: bool = True) -> dict:
    """Get the resources configuration (resources_config.yaml)."""
    return load_config(DEFAULT_RESOURCES_CONFIG_PATH, use_cache)


# =============================================================================
# Resources Config API
# =============================================================================

def _extract_model_entries(entries: Any, app_type_filter: Optional[str] = None) -> list[ModelEntry]:
    """Extract ModelEntry objects from config entries."""
    if is_none_value(entries):
        return []

    entries_list = entries if isinstance(entries, list) else [entries]
    models = []

    for entry in entries_list:
        if is_none_value(entry):
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if name and not is_none_value(name):
                # Parse app_type field - default to both if not specified
                app_type_raw = entry.get("app_type", ["pipeline", "standalone"])
                if isinstance(app_type_raw, str):
                    app_type = (app_type_raw,)
                else:
                    app_type = tuple(app_type_raw) if app_type_raw else ("pipeline", "standalone")
                
                model = ModelEntry(
                    name=name,
                    source=entry.get("source", "mz"),
                    url=entry.get("url"),
                    app_type=app_type,
                )
                
                # Filter by app_type if requested
                if app_type_filter is None or app_type_filter in model.app_type:
                    models.append(model)
        elif isinstance(entry, str) and not is_none_value(entry):
            model = ModelEntry(name=entry, source="mz")
            # String entries default to both types, so always include unless filtered out
            if app_type_filter is None or app_type_filter in model.app_type:
                models.append(model)

    return models


def get_default_models(app_name: str, arch: str, app_type: Optional[str] = None) -> list[ModelEntry]:
    """Get default model entries for an app and architecture."""
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("default"), app_type_filter=app_type)


def get_extra_models(app_name: str, arch: str, app_type: Optional[str] = None) -> list[ModelEntry]:
    """Get extra model entries for an app and architecture."""
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("extra"), app_type_filter=app_type)


def get_all_models(app_name: str, arch: str, app_type: Optional[str] = None) -> list[ModelEntry]:
    """Get all model entries (default + extra) for an app and architecture."""
    return get_default_models(app_name, arch, app_type) + get_extra_models(app_name, arch, app_type)


def get_model_names(app_name: str, arch: str, tier: str = "all", app_type: Optional[str] = None) -> list[str]:
    """Get model names for an app and architecture."""
    if tier == "default":
        models = get_default_models(app_name, arch, app_type)
    elif tier == "extra":
        models = get_extra_models(app_name, arch, app_type)
    else:
        models = get_all_models(app_name, arch, app_type)

    return [m.name for m in models]


def get_default_model_name(app_name: str, arch: str, app_type: Optional[str] = None) -> Optional[str]:
    """Get the first default model name for an app and architecture."""
    models = get_default_models(app_name, arch, app_type)
    return models[0].name if models else None


# =============================================================================
# Inputs Config API (images/videos per app based on tags)
# =============================================================================

def _get_resources_by_tag(section: str, app_name: str) -> list[dict]:
    """Get resources from a section that are tagged for an app."""
    config = get_resources_config()
    all_resources = config.get(section, [])
    
    matching = []
    for resource in all_resources:
        if not isinstance(resource, dict):
            continue
        tags = resource.get("tag", [])
        if isinstance(tags, str):
            tags = [tags]
        if app_name in tags:
            matching.append(resource)
    
    return matching


def get_videos_for_app(app_name: str) -> list[dict]:
    """Get videos that are tagged for a specific application."""
    return _get_resources_by_tag("videos", app_name)


def get_images_for_app(app_name: str) -> list[dict]:
    """Get images that are tagged for a specific application."""
    return _get_resources_by_tag("images", app_name)


# =============================================================================
# Standalone helpers
# =============================================================================

def is_standalone_app_name(app_name: str) -> bool:
    """Return True if the app name uses the _standalone suffix."""
    return app_name.endswith(STANDALONE_SUFFIX)


def base_app_name(app_name: str) -> str:
    """Strip the _standalone suffix if present."""
    return app_name[: -len(STANDALONE_SUFFIX)] if is_standalone_app_name(app_name) else app_name


# =============================================================================
# Download Infrastructure
# =============================================================================

DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


@dataclass
class DownloadConfig:
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 300
    parallel_workers: int = 4
    show_progress: bool = True
    dry_run: bool = False
    force_redownload: bool = False
    include_gen_ai: bool = False


@dataclass(frozen=True)
class DownloadTask:
    url: str
    dest_path: Path
    resource_type: str
    name: str
    expected_size: Optional[int] = None


@dataclass
class DownloadResult:
    task: DownloadTask
    success: bool
    message: str
    skipped: bool = False
    file_size: int = 0


def is_none_value(value: Any) -> bool:
    """Check if a value is effectively None (including string 'none')."""
    return value is None or (isinstance(value, str) and value.lower() == "none")


def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get Content-Length of a remote file via HEAD request."""
    try:
        r = requests.head(url, headers={'User-Agent': DEFAULT_USER_AGENT}, timeout=timeout, allow_redirects=True)
        length = r.headers.get('Content-Length')
        return int(length) if length else None
    except Exception:
        return None


class ResourceDownloader:
    """Manages downloading of models and resources from various sources."""
    RESOURCE_MAP = {
        RESOURCE_TYPE_IMAGE: (RESOURCES_PHOTOS_DIR_NAME, "images"),
        RESOURCE_TYPE_VIDEO: (RESOURCES_VIDEOS_DIR_NAME, "video"),
        "json": (RESOURCES_JSON_DIR_NAME, "configs"),
        "npy": (RESOURCES_NPY_DIR_NAME, "npy"),
    }

    def __init__(self, config: dict, hailo_arch: str, resource_root: Path, download_config: Optional[DownloadConfig] = None):
        self.config = config
        self.hailo_arch = hailo_arch
        self.resource_root = Path(resource_root)
        self.download_config = download_config or DownloadConfig()
        self.model_zoo_version = HAILO_MODEL_ZOO_DEFAULT_VERSION
        self._tasks: set[DownloadTask] = set()

    def _should_download(self, dest_path: Path, expected_size: Optional[int]) -> tuple[bool, str]:
        if self.download_config.force_redownload:
            return True, "Force redownload requested"
        if not dest_path.exists():
            return True, "File does not exist"
        if dest_path.stat().st_size == 0:
            return True, "File is empty"
        if expected_size and dest_path.stat().st_size != expected_size:
            return True, f"Size mismatch (local: {dest_path.stat().st_size}, remote: {expected_size})"
        return False, "File already exists and valid"

    def _download_file_with_retry(self, task: DownloadTask) -> DownloadResult:
        remote_size = task.expected_size or get_remote_file_size(task.url)
        should_download, reason = self._should_download(task.dest_path, remote_size)
        
        if not should_download:
            return DownloadResult(task, True, reason, True, task.dest_path.stat().st_size if task.dest_path.exists() else 0)
        if self.download_config.dry_run:
            logger.info(f"[DRY RUN] Would download: {task.url} -> {task.dest_path}")
            return DownloadResult(task, True, "Dry run", True)

        task.dest_path.parent.mkdir(parents=True, exist_ok=True)
        last_err = None
        for attempt in range(self.download_config.max_retries):
            try:
                logger.info(f"Downloading: {task.url}")
                download_url(task.url, str(task.dest_path))
                
                if remote_size and task.dest_path.stat().st_size != remote_size:
                    raise ValueError(f"Size mismatch: got {task.dest_path.stat().st_size}, expected {remote_size}")
                
                return DownloadResult(task, True, "Success", False, task.dest_path.stat().st_size)
            except Exception as e:
                last_err = e
                if task.dest_path.exists():
                    task.dest_path.unlink()
                if attempt < self.download_config.max_retries - 1:
                    time.sleep(self.download_config.retry_delay * (2 ** attempt))

        return DownloadResult(task, False, str(last_err))

    def _add_task(self, url: str, dest: Path, resource_type: str, name: str):
        """Helper to add a unique download task."""
        if url:
            self._tasks.add(DownloadTask(url, dest, resource_type, name))

    def _add_resource_task(self, entry: Any, resource_type: str):
        if is_none_value(entry):
            return
        name = entry.get("name") if isinstance(entry, dict) else Path(entry).name
        url = entry.get("url") if isinstance(entry, dict) else (entry if entry.startswith("http") else None)
        source = entry.get("source") if isinstance(entry, dict) else None
        if not name or (not url and source != "s3"):
            return

        dest_dir, s3_folder = self.RESOURCE_MAP.get(resource_type, (resource_type, resource_type))
        dest = self.resource_root / dest_dir / name
        if not url and source == "s3":
            url = f"{S3_RESOURCES_BASE_URL}/{s3_folder}/{name}"
        self._add_task(url, dest, resource_type, name)

    def _add_model_task(self, model_entry: Any, is_gen_ai_allowed: bool = False):
        if is_none_value(model_entry):
            return
        name = model_entry.get("name") if isinstance(model_entry, dict) else model_entry
        source = model_entry.get("source", "mz") if isinstance(model_entry, dict) else "mz"
        if source == "gen-ai-mz" and not is_gen_ai_allowed:
            return

        url = model_entry.get("url") if isinstance(model_entry, dict) else None
        if not url:
            s3_arch = "h8l" if self.hailo_arch == HAILO8L_ARCH else "h8"
            if source == "s3":
                url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
            elif source == "mz":
                url = f"{MODEL_ZOO_URL}/{self.model_zoo_version}/{self.hailo_arch}/{name}{HAILO_FILE_EXTENSION}"
            elif source == "gen-ai-mz":
                base = self.config.get("metadata", {}).get("s3_endpoints", {}).get("gen_ai_mz", "https://dev-public.hailo.ai")
                url = f"{base}/{self.model_zoo_version}/blob/{name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION}"
        
        if url:
            dest_name = name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION
            dest = self.resource_root / RESOURCES_MODELS_DIR_NAME / self.hailo_arch / dest_name
            self._add_task(url, dest, "model", name)

    def collect_resources(self, resource_types: list[str]):
        """Collect all resources for given types."""
        for rt in resource_types:
            for entry in self.config.get(rt, []):
                self._add_resource_task(entry, rt)

    def collect_specific_resource_for_app(self, app_name: str, name: str, resource_type: str):
        """Collect a specific resource tagged for an app."""
        config_func = get_images_for_app if resource_type == RESOURCE_TYPE_IMAGE else get_videos_for_app
        for entry in config_func(app_name=app_name):
            entry_name = entry.get("name") if isinstance(entry, dict) else Path(entry).name
            if entry_name == name:
                self._add_resource_task(entry, resource_type)
                return
        logger.warning(f"{resource_type.capitalize()} '{name}' not found for app '{app_name}'")

    def collect_specific_model_for_app(self, app_name: str, model_name: str):
        """Collect a specific model entry for an app."""
        app_cfg = self.config.get(app_name, {}).get("models", {}).get(self.hailo_arch, {})
        for tier in ["default", "extra"]:
            entries = app_cfg.get(tier, [])
            entries_list = entries if isinstance(entries, list) else [entries]
            for e in entries_list:
                name = e.get("name") if isinstance(e, dict) else e
                if name == model_name:
                    self._add_model_task(e, True)
                    return
        logger.warning(f"Model '{model_name}' not found for app '{app_name}'")

    def collect_models_for_app(self, app_name: str, include_extra: bool = False, is_gen_ai_allowed: bool = False):
        """Collect models associated with an application."""
        app_cfg = self.config.get(app_name, {}).get("models", {}).get(self.hailo_arch, {})
        if not app_cfg:
            return
        
        tiers = ["default"]
        if include_extra:
            tiers.append("extra")
            
        for tier in tiers:
            entries = app_cfg.get(tier, [])
            entries_list = entries if isinstance(entries, list) else [entries]
            for m in entries_list:
                self._add_model_task(m, is_gen_ai_allowed)

    def collect_all_default_models(self, include_extra: bool = False, exclude_gen_ai_apps: bool = True):
        """Collect default models for all apps in the configuration."""
        for app_name, app_config in self.config.items():
            if not isinstance(app_config, dict) or "models" not in app_config:
                continue
            
            is_gen_ai = False
            for arch in app_config["models"].values():
                for tier in ["default", "extra"]:
                    entries = arch.get(tier, [])
                    entries_list = entries if isinstance(entries, list) else [entries]
                    if any(isinstance(m, dict) and m.get("source") == "gen-ai-mz" for m in entries_list):
                        is_gen_ai = True
                        break
            
            if exclude_gen_ai_apps and is_gen_ai:
                continue
            self.collect_models_for_app(app_name, include_extra, not exclude_gen_ai_apps)

    def execute(self, parallel: bool = True) -> list[DownloadResult]:
        """Execute all collected download tasks."""
        if not self._tasks:
            return []
            
        logger.info(f"Executing {len(self._tasks)} download tasks...")
        results = []
        
        if parallel and len(self._tasks) > 1 and not self.download_config.dry_run:
            orig_show = self.download_config.show_progress
            self.download_config.show_progress = False
            with ThreadPoolExecutor(max_workers=self.download_config.parallel_workers) as pool:
                futures = {pool.submit(self._download_file_with_retry, t): t for t in self._tasks}
                for i, f in enumerate(as_completed(futures), 1):
                    res = f.result()
                    results.append(res)
                    logger.info(f"[{i}/{len(self._tasks)}] {'✓' if res.success else '✗'} {res.task.name}")
            self.download_config.show_progress = orig_show
        else:
            for i, t in enumerate(self._tasks, 1):
                logger.info(f"[{i}/{len(self._tasks)}] Processing {t.name}...")
                results.append(self._download_file_with_retry(t))
                
        failed = [r for r in results if not r.success]
        logger.info(f"Summary: {sum(1 for r in results if r.success and not r.skipped)} downloaded, "
                          f"{sum(1 for r in results if r.skipped)} skipped, {len(failed)} failed")
        for r in failed:
            logger.warning(f"  - {r.task.name}: {r.message}")
        return results


def _create_downloader(resource_config_path, arch, dry_run, force, include_gen_ai):
    cfg_path = Path(resource_config_path or DEFAULT_RESOURCES_CONFIG_PATH)
    hailo_arch = arch or detect_hailo_arch()
    if not hailo_arch:
        print("\n❌ ERROR: Could not detect Hailo device architecture.", file=sys.stderr)
        sys.exit(1)
    return ResourceDownloader(
        load_config(cfg_path), hailo_arch, Path(RESOURCES_ROOT_PATH_DEFAULT),
        DownloadConfig(dry_run=dry_run, force_redownload=force, include_gen_ai=include_gen_ai)
    )


def download_resources(resource_config_path=None, arch=None, group=None, all_models=False, 
                       resource_name=None, resource_type=None, dry_run=False, force=False, 
                       parallel=True, include_gen_ai=False):
    """Public API for downloading resources."""
    dl = _create_downloader(resource_config_path, arch, dry_run, force, include_gen_ai)
    
    if resource_name:
        if not resource_type or not group:
            logger.error("Targeted download requires resource_type and group")
            return
        if resource_type == RESOURCE_TYPE_MODEL:
            dl.collect_specific_model_for_app(group, resource_name)
        else:
            dl.collect_specific_resource_for_app(group, resource_name, resource_type)
    elif group and group.lower() != "default":
        if group not in dl.config:
            logger.error(f"Group '{group}' not found")
            return
        dl.collect_models_for_app(group, True, True)
        dl.collect_resources([RESOURCE_TYPE_VIDEO, RESOURCE_TYPE_IMAGE, "json", "npy"])
    else:
        dl.collect_resources([RESOURCE_TYPE_VIDEO, RESOURCE_TYPE_IMAGE, "json", "npy"])
        dl.collect_all_default_models(include_extra=all_models, exclude_gen_ai_apps=not include_gen_ai)
    
    dl.execute(parallel)


# =============================================================================
# Resource Resolution API
# =============================================================================

def get_resource_path(resource_type: str, name: str, arch: str | None = None) -> Path:
    """Map a resource type and name to its local filesystem path."""
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    
    if resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = arch or detect_hailo_arch()
        if not arch:
            raise RuntimeError("Could not detect Hailo architecture for model path.")
        model_path = root / RESOURCES_MODELS_DIR_NAME / arch / name
        return model_path if name.endswith(HAILO_FILE_EXTENSION) else model_path.with_suffix(HAILO_FILE_EXTENSION)

    # Standard resource directory mapping
    type_dirs = {
        RESOURCES_VIDEOS_DIR_NAME: RESOURCES_VIDEOS_DIR_NAME,
        RESOURCES_PHOTOS_DIR_NAME: RESOURCES_PHOTOS_DIR_NAME,
        RESOURCES_JSON_DIR_NAME: RESOURCES_JSON_DIR_NAME,
        RESOURCES_NPY_DIR_NAME: RESOURCES_NPY_DIR_NAME,
    }
    
    if resource_type in type_dirs:
        return root / type_dirs[resource_type] / name
        
    return root / resource_type / name


def resolve_hef_path(hef_path: Optional[str], app_name: str, arch: Optional[str] = None, 
                     app_type: Optional[str] = "standalone") -> Optional[Path]:
    """Resolve HEF path, downloading it if necessary."""
    arch = arch or detect_hailo_arch()
    if not arch:
        raise RuntimeError("Could not detect Hailo architecture.")

    available_models = get_model_names(app_name, arch, app_type=app_type)
    default_model = get_default_model_name(app_name, arch, app_type=app_type)

    if hef_path is None:
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
    if model_name in available_models:
        print(f"\n⚠️  WARNING: Model '{model_name}' not found. Downloading for {app_name}/{arch}...")
        if _download_resource(model_name, RESOURCE_TYPE_MODEL, app_name, arch):
            if resource_path.exists():
                return resource_path
        logger.error(f"Failed to download model: {model_name}")

    logger.error(f"Model '{model_name}' not found. Available: {', '.join(available_models)}")
    return None


def _download_resource(name: str, resource_type: str, app_name: str, arch: Optional[str] = None) -> bool:
    """Internal helper to download a specific resource."""
    try:
        download_resources(arch=arch, group=app_name, resource_name=name, 
                           resource_type=resource_type, parallel=False)
        return True
    except Exception as e:
        logger.error(f"Failed to download resource {name}: {e}")
        return False


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
