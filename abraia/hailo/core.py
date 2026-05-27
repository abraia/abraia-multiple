from __future__ import annotations
"""Core helpers: arch detection, buffer utils, model resolution."""

import os
import sys
import time
import yaml
import shlex
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .hailo_logger import get_logger
hailo_logger = get_logger(__name__)

# Base Defaults
HAILO8_ARCH = "hailo8"
HAILO8L_ARCH = "hailo8l"
HAILO10H_ARCH = "hailo10h"
HAILO_FILE_EXTENSION = ".hef"
MODEL_ZOO_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"
S3_RESOURCES_BASE_URL = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"
RESOURCES_ROOT_PATH_DEFAULT = "/usr/local/hailo/resources"  # Do Not Change!

# Core defaults
RPI_POSSIBLE_NAME = "Raspberry Pi"
HAILO8_ARCH_CAPS = "HAILO8"
HAILO8L_ARCH_CAPS = "HAILO8L"
HAILO10H_ARCH_CAPS = "HAILO10H"
HAILO15H_ARCH_CAPS = "HAILO15H"
HAILO_FW_CONTROL_CMD = "hailortcli fw-control identify"
USB_CAMERA = "usb"

# Base project paths
REPO_ROOT = Path(__file__).resolve().parents[4]


def _get_package_config_path(filename: str) -> Path | None:
    """Get config file path from package location (when installed via pip)."""
    package_dir = Path(__file__).parent
    config_path = package_dir / filename
    if config_path.exists():
        return config_path


def _get_repo_config_path(filename: str) -> Path:
    """Get config file path from repo location (when running from source)."""
    return Path(__file__).parent / filename


def _get_config_path(filename: str) -> str:
    """Get config path: first check package location, then fall back to repo location."""
    # First try package location (when installed via pip)
    package_path = _get_package_config_path(filename)
    if package_path:
        return str(package_path)
    # Fall back to repo location (when running from source)
    return str(_get_repo_config_path(filename))


DEFAULT_RESOURCES_CONFIG_PATH = _get_config_path("resources_config.yaml")

# Symlink, dotenv, local resources defaults
DEFAULT_LOCAL_RESOURCES_PATH = str(REPO_ROOT / "local_resources")  # bundled GIFs, JSON, etc.

# Config key constants
HAILO_ARCH_KEY = "hailo_arch"

# Resources directory structure
RESOURCES_MODELS_DIR_NAME = "models"
RESOURCES_VIDEOS_DIR_NAME = "videos"
RESOURCES_SO_DIR_NAME = "so"
RESOURCES_PHOTOS_DIR_NAME = "images"  # Changed from "photos" to match actual directory name
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

DEFAULT_COCO_LABELS_PATH = str(Path(__file__).parent / "coco.txt")

# Standalone app naming convention
STANDALONE_SUFFIX = "_standalone"

# =============================================================================
# Exceptions
# =============================================================================

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


# =============================================================================
# Type Definitions (Dataclasses)
# =============================================================================

@dataclass(frozen=True)
class ModelEntry:
    """Represents a model entry from resources config.
    
    Attributes:
        name: Model name (e.g., "yolov8m")
        source: Source type - "mz" (Model Zoo), "s3", or "gen-ai-mz"
        url: Optional explicit URL for download
        app_type: List of supported app types - "pipeline", "standalone", or both
    """
    name: str
    source: str  # "mz" | "s3" | "gen-ai-mz"
    url: Optional[str] = None
    app_type: tuple[str, ...] = ("pipeline", "standalone")  # Default: supports both


# =============================================================================
# Path Resolution
# =============================================================================

class ConfigPaths:
    """Centralized path resolution for configuration files."""

    @classmethod
    def _get_config_dir(cls) -> Path:
        """Get the configuration directory."""
        return Path(__file__).parent

    @classmethod
    def resources_config(cls) -> Path:
        """Get path to resources_config.yaml."""
        return cls._get_config_dir() / "resources_config.yaml"


# =============================================================================
# Base Loading Utilities
# =============================================================================

@lru_cache(maxsize=8)
def _load_yaml_cached(path: str) -> dict:
    """Load YAML file with caching."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path_obj, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    return data if data else {}


def load_config(path: Path, use_cache: bool = True) -> dict:
    """Load YAML file, optionally with caching."""
    if use_cache:
        return _load_yaml_cached(str(path))

    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    return data if data else {}


# =============================================================================
# Resources Config API (resources_config.yaml)
# =============================================================================

def get_resources_config(use_cache: bool = True) -> dict:
    """Get the resources configuration (resources_config.yaml)."""
    return load_config(ConfigPaths.resources_config(), use_cache)


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


def get_inputs_for_app(app_name: str, is_standalone: bool = False) -> dict:
    """Get input resources for an application based on app type."""
    # Standalone apps use the base app tag names (without _standalone suffix)
    app_name = base_app_name(app_name) if is_standalone or is_standalone_app_name(app_name) else app_name
    result = {
        "videos": get_videos_for_app(app_name),
    }
    
    if is_standalone:
        result["images"] = get_images_for_app(app_name)
    
    return result


# =============================================================================
# Standalone helpers
# =============================================================================

def is_standalone_app_name(app_name: str) -> bool:
    """Return True if the app name uses the _standalone suffix."""
    return app_name.endswith(STANDALONE_SUFFIX)


def base_app_name(app_name: str) -> str:
    """Strip the _standalone suffix if present."""
    return app_name[: -len(STANDALONE_SUFFIX)] if is_standalone_app_name(app_name) else app_name


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

def is_none_value(value) -> bool:
    return value is None or (isinstance(value, str) and value.lower() == "none")

def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': DEFAULT_USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get('Content-Length')
            return int(length) if length else None
    except Exception:
        return None

class ProgressTracker:
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self._last_percent = -1

    def update(self, downloaded: int, total_size: int):
        if not self.show_progress: return
        percent = min(100, (downloaded * 100) // total_size) if total_size > 0 else 0
        if percent == self._last_percent: return
        self._last_percent = percent
        bar = '=' * (percent // 2.5) + '-' * (40 - int(percent // 2.5))
        print(f"\r[{bar}] {percent}% ({downloaded/1e6:.2f}/{total_size/1e6:.2f} MB)", end='', flush=True)

    def finish(self):
        if self.show_progress: print()

class ResourceDownloader:
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
        self.model_zoo_version = "v2.17.0"
        self._tasks: set[DownloadTask] = set()

    def _should_download(self, dest_path: Path, expected_size: Optional[int]) -> tuple[bool, str]:
        if self.download_config.force_redownload: return True, "Force redownload requested"
        if not dest_path.exists(): return True, "File does not exist"
        if dest_path.stat().st_size == 0: return True, "File is empty"
        if expected_size and dest_path.stat().st_size != expected_size:
            return True, f"Size mismatch (local: {dest_path.stat().st_size}, remote: {expected_size})"
        return False, "File already exists and valid"

    def _download_file_with_retry(self, task: DownloadTask) -> DownloadResult:
        remote_size = task.expected_size or get_remote_file_size(task.url)
        should_download, reason = self._should_download(task.dest_path, remote_size)
        
        if not should_download:
            return DownloadResult(task, True, reason, True, task.dest_path.stat().st_size if task.dest_path.exists() else 0)
        if self.download_config.dry_run:
            hailo_logger.info(f"[DRY RUN] Would download: {task.url} → {task.dest_path}")
            return DownloadResult(task, True, "Dry run", True)

        task.dest_path.parent.mkdir(parents=True, exist_ok=True)
        last_err = None
        for attempt in range(self.download_config.max_retries):
            temp_path = None
            try:
                fd, temp_path = tempfile.mkstemp(dir=task.dest_path.parent, prefix=f".{task.name}.", suffix=".tmp")
                os.close(fd)
                temp_path = Path(temp_path)
                
                req = urllib.request.Request(task.url, headers={'User-Agent': DEFAULT_USER_AGENT})
                progress = ProgressTracker(self.download_config.show_progress)
                hailo_logger.info(f"Downloading: {task.url}")
                
                with urllib.request.urlopen(req, timeout=self.download_config.timeout) as resp:
                    total = int(resp.headers.get('Content-Length', 0))
                    with open(temp_path, 'wb') as f:
                        downloaded = 0
                        while chunk := resp.read(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(downloaded, total)
                progress.finish()

                if remote_size and temp_path.stat().st_size != remote_size:
                    raise ValueError(f"Size mismatch: got {temp_path.stat().st_size}, expected {remote_size}")
                
                if task.dest_path.exists(): task.dest_path.unlink()
                temp_path.rename(task.dest_path)
                return DownloadResult(task, True, "Success", False, task.dest_path.stat().st_size)
            except Exception as e:
                last_err = e
                if temp_path and temp_path.exists(): temp_path.unlink()
                if attempt < self.download_config.max_retries - 1:
                    time.sleep(self.download_config.retry_delay * (2 ** attempt))

        return DownloadResult(task, False, str(last_err))

    def _add_resource_task(self, entry, resource_type: str):
        if is_none_value(entry): return
        name = entry.get("name") if isinstance(entry, dict) else Path(entry).name
        url = entry.get("url") if isinstance(entry, dict) else (entry if entry.startswith("http") else None)
        source = entry.get("source") if isinstance(entry, dict) else None
        if not name or (not url and source != "s3"): return

        dest_dir, s3_folder = self.RESOURCE_MAP.get(resource_type, (resource_type, resource_type))
        dest = self.resource_root / dest_dir / name
        if not url and source == "s3": url = f"{S3_RESOURCES_BASE_URL}/{s3_folder}/{name}"
        if url: self._tasks.add(DownloadTask(url, dest, resource_type, name))

    def _add_model_task(self, model_entry, is_gen_ai_allowed: bool = False):
        if is_none_value(model_entry): return
        name = model_entry.get("name") if isinstance(model_entry, dict) else model_entry
        source = model_entry.get("source", "mz") if isinstance(model_entry, dict) else "mz"
        if source == "gen-ai-mz" and not is_gen_ai_allowed: return

        url = model_entry.get("url") if isinstance(model_entry, dict) else None
        if not url:
            s3_arch = "h8l" if self.hailo_arch == HAILO8L_ARCH else "h8"
            if source == "s3": url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
            elif source == "mz": url = f"{MODEL_ZOO_URL}/{self.model_zoo_version}/{self.hailo_arch}/{name}{HAILO_FILE_EXTENSION}"
            elif source == "gen-ai-mz":
                base = self.config.get("metadata", {}).get("s3_endpoints", {}).get("gen_ai_mz", "https://dev-public.hailo.ai")
                url = f"{base}/{self.model_zoo_version}/blob/{name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION}"
        if url:
            dest = self.resource_root / RESOURCES_MODELS_DIR_NAME / self.hailo_arch / (name if name.endswith(HAILO_FILE_EXTENSION) else name + HAILO_FILE_EXTENSION)
            self._tasks.add(DownloadTask(url, dest, "model", name))

    def _add_onnx_task(self, onnx_name: str):
        s3_arch = "h8l" if self.hailo_arch == HAILO8L_ARCH else "h8"
        url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{onnx_name}"
        dest = self.resource_root / RESOURCES_MODELS_DIR_NAME / self.hailo_arch / onnx_name
        self._tasks.add(DownloadTask(url, dest, "onnx", onnx_name))

    def collect_all_videos(self): [self._add_resource_task(e, RESOURCE_TYPE_VIDEO) for e in self.config.get("videos", [])]
    def collect_all_images(self): [self._add_resource_task(e, RESOURCE_TYPE_IMAGE) for e in self.config.get("images", [])]
    def collect_all_json_files(self): [self._add_resource_task(e, "json") for e in self.config.get("json", [])]
    def collect_all_npy_files(self): [self._add_resource_task(e, "npy") for e in self.config.get("npy", [])]

    def collect_specific_resource_for_app(self, app_name: str, name: str, resource_type: str):
        config_func = get_images_for_app if resource_type == RESOURCE_TYPE_IMAGE else get_videos_for_app
        for entry in config_func(app_name=app_name):
            if (isinstance(entry, dict) and entry.get("name") == name) or (isinstance(entry, str) and Path(entry).name == name):
                self._add_resource_task(entry, resource_type); return
        hailo_logger.warning(f"{resource_type.capitalize()} '{name}' not found for app '{app_name}'")

    def collect_specific_model_for_app(self, app_name: str, model_name: str):
        app_cfg = self.config.get(app_name, {}).get("models", {}).get(self.hailo_arch, {})
        for k in ["default", "extra"]:
            if k in app_cfg:
                entries = app_cfg[k] if isinstance(app_cfg[k], list) else [app_cfg[k]]
                for e in entries:
                    if (isinstance(e, dict) and e.get("name") == model_name) or (isinstance(e, str) and e == model_name):
                        self._add_model_task(e, True); return
        hailo_logger.warning(f"Model '{model_name}' not found for app '{app_name}'")

    def collect_specific_onnx_for_app(self, app_name: str, onnx_name: str): self._add_onnx_task(onnx_name)

    def collect_models_for_app(self, app_name: str, include_extra: bool = False, is_gen_ai_allowed: bool = False):
        app_cfg = self.config.get(app_name, {}).get("models", {}).get(self.hailo_arch, {})
        if not app_cfg: return
        models = (app_cfg.get("default", []) if isinstance(app_cfg.get("default"), list) else ([app_cfg["default"]] if "default" in app_cfg else [])) + (app_cfg.get("extra", []) if include_extra else [])
        for m in models: self._add_model_task(m, is_gen_ai_allowed)

    def collect_all_default_models(self, include_extra: bool = False, exclude_gen_ai_apps: bool = True):
        for app_name, app_config in self.config.items():
            if not isinstance(app_config, dict) or "models" not in app_config: continue
            is_gen_ai = any(m.get("source") == "gen-ai-mz" for arch in app_config["models"].values() for m in (arch.get("default", []) if isinstance(arch.get("default"), list) else [arch.get("default")]) + arch.get("extra", []) if isinstance(m, dict))
            if exclude_gen_ai_apps and is_gen_ai: continue
            self.collect_models_for_app(app_name, include_extra, not exclude_gen_ai_apps)

    def collect_group_resources(self, group_name: str):
        if group_name not in self.config: hailo_logger.error(f"Group '{group_name}' not found"); return
        self.collect_models_for_app(group_name, True, True)
        self.collect_all_videos(); self.collect_all_images(); self.collect_all_json_files(); self.collect_all_npy_files()

    def execute(self, parallel: bool = True) -> list[DownloadResult]:
        if not self._tasks: return []
        hailo_logger.info(f"Executing {len(self._tasks)} tasks...")
        results = []
        if parallel and len(self._tasks) > 1 and not self.download_config.dry_run:
            orig_show = self.download_config.show_progress
            self.download_config.show_progress = False
            with ThreadPoolExecutor(max_workers=self.download_config.parallel_workers) as pool:
                futures = {pool.submit(self._download_file_with_retry, t): t for t in self._tasks}
                for i, f in enumerate(as_completed(futures), 1):
                    res = f.result(); results.append(res)
                    hailo_logger.info(f"[{i}/{len(self._tasks)}] {'✓' if res.success else '✗'} {res.task.name}")
            self.download_config.show_progress = orig_show
        else:
            for i, t in enumerate(self._tasks, 1):
                hailo_logger.info(f"[{i}/{len(self._tasks)}] Processing {t.name}...")
                results.append(self._download_file_with_retry(t))
        failed = [r for r in results if not r.success]
        hailo_logger.info(f"Summary: {sum(1 for r in results if r.success and not r.skipped)} downloaded, {sum(1 for r in results if r.skipped)} skipped, {len(failed)} failed")
        for r in failed: hailo_logger.warning(f"  - {r.task.name}: {r.message}")
        return results

def _create_downloader(resource_config_path, arch, dry_run, force, include_gen_ai):
    cfg_path = Path(resource_config_path or DEFAULT_RESOURCES_CONFIG_PATH)
    if not cfg_path.is_file(): hailo_logger.error(f"Config not found: {cfg_path}"); return None
    hailo_arch = arch or detect_hailo_arch()
    if not hailo_arch: print("\n❌ ERROR: Could not detect Hailo device architecture.", file=sys.stderr); sys.exit(1)
    return ResourceDownloader(load_config(cfg_path), hailo_arch, Path(RESOURCES_ROOT_PATH_DEFAULT), DownloadConfig(dry_run=dry_run, force_redownload=force, include_gen_ai=include_gen_ai))

def download_resources(resource_config_path=None, arch=None, group=None, all_models=False, resource_name=None, resource_type=None, dry_run=False, force=False, parallel=True, include_gen_ai=False):
    if resource_name:
        if not resource_type or not group: hailo_logger.error("Targeted download requires resource_type and group"); return
        dl = _create_downloader(resource_config_path, arch, dry_run, force, include_gen_ai)
        if dl:
            if resource_type == RESOURCE_TYPE_MODEL: dl.collect_specific_model_for_app(group, resource_name)
            else: dl.collect_specific_resource_for_app(group, resource_name, resource_type)
            dl.execute(parallel)
        return
    dl = _create_downloader(resource_config_path, arch, dry_run, force, include_gen_ai)
    if not dl: return
    if group and group.lower() != "default": dl.collect_group_resources(group)
    else:
        dl.collect_all_videos(); dl.collect_all_images(); dl.collect_all_json_files(); dl.collect_all_npy_files()
        dl.collect_all_default_models(include_extra=all_models, exclude_gen_ai_apps=not include_gen_ai)
    dl.execute(parallel)


def get_resource_path(
    pipeline_name: str, resource_type: str, arch: str | None = None, model: str | None = None
) -> Path | None:
    hailo_logger.debug(
        f"Getting resource path for pipeline={pipeline_name}, resource_type={resource_type}, model={model}"
    )
    root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    # Auto-detect arch if not provided and needed for RESOURCES_MODELS_DIR_NAME
    if arch is None and resource_type == RESOURCES_MODELS_DIR_NAME:
        arch = os.getenv(HAILO_ARCH_KEY) or detect_hailo_arch()
        hailo_logger.debug(f"Auto-detected arch: {arch}")

    if not arch and resource_type == RESOURCES_MODELS_DIR_NAME:
        hailo_logger.error("Could not detect Hailo architecture.")
        assert False, "Could not detect Hailo architecture."

    if resource_type == RESOURCES_SO_DIR_NAME and model:
        return root / RESOURCES_SO_DIR_NAME / model
    if resource_type == RESOURCES_VIDEOS_DIR_NAME and model:
        return root / RESOURCES_VIDEOS_DIR_NAME / model
    if resource_type == RESOURCES_PHOTOS_DIR_NAME and model:
        return root / RESOURCES_PHOTOS_DIR_NAME / model
    if resource_type == RESOURCES_JSON_DIR_NAME and model:
        return root / RESOURCES_JSON_DIR_NAME / model
    if resource_type == RESOURCES_NPY_DIR_NAME and model:
        return root / RESOURCES_NPY_DIR_NAME / model
    if resource_type == DEFAULT_LOCAL_RESOURCES_PATH and model:
        return root / DEFAULT_LOCAL_RESOURCES_PATH / model

    if resource_type == RESOURCES_MODELS_DIR_NAME:
        if model:
            model_path = root / RESOURCES_MODELS_DIR_NAME / arch / model
            if "." in model:
                return model_path.with_name(model_path.name + HAILO_FILE_EXTENSION)
            return model_path.with_suffix(HAILO_FILE_EXTENSION)

    return None


def resolve_hef_path(
    hef_path: str | None,
    app_name: str,
    arch: str | None = None,
    app_type: str | None = "standalone",
) -> Path | None:
    """
    Main method for resolving HEF (Hailo Executable Format) file paths.

    Provides intelligent path resolution with automatic model downloading.
    See README.md for detailed documentation and usage examples.

    Args:
        hef_path: User-provided path or model name (None uses default model)
        app_name: Application name from resources config (e.g., DETECTION_PIPELINE)
        arch: Hailo architecture ('hailo8', 'hailo8l', or 'hailo10h')
        app_type: Filter by app type ("pipeline" or "standalone").
                  If None, auto-detects from caller's location.

    Returns:
        Path to the HEF file, or None if not found
    """
    resources_root = Path(RESOURCES_ROOT_PATH_DEFAULT)

    # Auto-detect arch if not provided.
    if arch is None:
        arch = os.getenv(HAILO_ARCH_KEY) or detect_hailo_arch()
        if not arch:
            hailo_logger.error("Could not detect Hailo architecture.")
            assert False, "Could not detect Hailo architecture."
        hailo_logger.debug(f"Auto-detected arch: {arch}")

    models_dir = resources_root / RESOURCES_MODELS_DIR_NAME / arch

    # Get available models for this app/arch (filtered by app_type if detected)
    available_models = get_model_names(app_name, arch, tier="all", app_type=app_type)
    default_model = get_default_model_name(app_name, arch, app_type=app_type)
    is_using_default = False

    # Case 1: No hef_path provided - use default model
    if hef_path is None:
        if default_model:
            hef_path = default_model
            is_using_default = True
            hailo_logger.info(f"Using default model: {default_model}")
        else:
            hailo_logger.error(f"No default model found for {app_name}/{arch}")
            return None
    
    # Normalize model name (extract basename and remove only the .hef suffix, keep dots in name)
    candidate_name = Path(hef_path).name
    if candidate_name.endswith(HAILO_FILE_EXTENSION):
        model_name = candidate_name[: -len(HAILO_FILE_EXTENSION)]
    else:
        model_name = candidate_name
    
    # Case 2: Treat an existing path (absolute or relative) as a file path
    hef_full_path = Path(hef_path)
    if hef_full_path.exists():
        resolved = hef_full_path.resolve()
        hailo_logger.info(f"Using HEF from path: {resolved}")
        return resolved

    # Also check with .hef extension
    if not hef_path.endswith(HAILO_FILE_EXTENSION):
        hef_full_path = Path(hef_path + HAILO_FILE_EXTENSION)
        if hef_full_path.exists():
            hailo_logger.info(f"Using HEF from path: {hef_full_path}")
            return hef_full_path

    # Case 3: Check in resources folder
    resource_path = models_dir / f"{model_name}{HAILO_FILE_EXTENSION}"
    if resource_path.exists():
        hailo_logger.info(f"Found HEF in resources: {resource_path}")
        return resource_path

    # Case 4: Model not found locally - check if it's in the available models list
    if model_name in available_models:

        # Show warning before downloading
        if is_using_default:
            print(f"\n⚠️  WARNING: Default model '{model_name}' is not downloaded.")
            print(f"   Downloading model for {app_name}/{arch}...")
            print(f"   This may take a while depending on your internet connection.\n")
        else:
            print(f"\n⚠️  WARNING: Model '{model_name}' is not downloaded.")
            print(f"   Downloading model for {app_name}/{arch}...")
            print(f"   This may take a while depending on your internet connection.\n")

        if _download_resource(model_name, RESOURCE_TYPE_MODEL, app_name, arch):
            if resource_path.exists():
                hailo_logger.info(f"Model downloaded successfully: {resource_path}")
                return resource_path
            else:
                hailo_logger.error(f"Download succeeded but file not found: {resource_path}")
                return None
        else:
            hailo_logger.error(f"Failed to download model: {model_name}")
            return None

    # Model not in available list - don't auto-download unknown models
    app_type_info = f" ({app_type})" if app_type else ""
    hailo_logger.error(
        f"Model '{model_name}' not found and not in available models list. "
        f"Available models for {app_name}/{arch}{app_type_info}: {', '.join(available_models) if available_models else 'None'}"
    )
    return None


def _download_resource(resource_name: str, resource_type: str, app_name: str, arch: str | None = None) -> bool:
    """
    Download a specific resource using the download_resources module.

    Args:
        resource_name: Name of the resource to download
        resource_type: Type of the resource.
                       Supported values: "model", "image", "video", "onnx"
        app_name: Application/group name used by the downloader
        arch: Optional Hailo architecture override

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        print(f"Downloading resource: {resource_name} (type: {resource_type})")
        download_resources(
            arch=arch,
            group=app_name,
            resource_name=resource_name,
            resource_type=resource_type,
            dry_run=False,
            force=False,
            parallel=False  # Sequential for single model
        )
        return True
    except Exception as e:
        hailo_logger.error(f"Failed to download resource: {e}")
        return False


def resolve_input_arg(app: str, input_arg: str | None) -> str:
    """
    Resolve the CLI `--input` argument into a concrete input source.

    Supported inputs (in priority order):
      1) If `--input` is NOT provided:
         - Try to use the first available input defined in the resources YAML that already exists locally.
         - If no input exists locally, try to download the default input from the YAML and use it.
         - If download fails, exit with an actionable error message.

      2) If `--input` IS provided:
         - If `--input` is 'usb' or 'rpi': treat it as a camera source keyword and return as-is.
         - If it is an existing local file or directory path: return it as-is.
         - Otherwise, treat it as a resource name defined in the resources YAML:
             * If it exists locally under the resources folder, return it.
             * If it is missing, download it and return its resolved local path.
         - If nothing matches, exit with a clear error.

    """

    def resolve_tagged_resource(
        app_name: str,
        preferred_name: str | None = None,
        allow_download_default: bool = False
    ) -> str | None:
        """
        Resolve an input resource defined in resources_config.yaml.

        Behavior:
          - If `preferred_name` is provided, resolve that exact resource name.
          - If `preferred_name` is not provided, pick the first available resource.
          - Images are preferred over videos.
          - Resources are downloaded automatically if required.
        """

        resource_app_name = _map_app_to_resource_group(app_name)
        inputs = get_inputs_for_app(resource_app_name, is_standalone=True)

        def pick(section: str) -> str | None:
            """
            Resolve a resource from a specific YAML section ('images' or 'videos').
            """
            resource_type = RESOURCES_PHOTOS_DIR_NAME if section == "images" else RESOURCES_VIDEOS_DIR_NAME

            def download_if_missing(name: str) -> str | None:
                """
                Download a resource by name and return its resolved local path (or None).
                """
                print(f"\n⚠️  WARNING: Input '{preferred_name}' is not downloaded.")
                print(f"   Downloading input for {app_name}...")
                print("   This may take a while depending on your internet connection.\n")

                if _download_resource(
                    resource_name=name,
                    resource_type=RESOURCE_TYPE_IMAGE if section == "images" else RESOURCE_TYPE_VIDEO,
                    app_name=resource_app_name,
                ):
                    resolved = get_resource_path(
                        pipeline_name=None,
                        resource_type=resource_type,
                        arch=None,
                        model=name,
                    )
                    if resolved and resolved.exists():
                        return str(resolved)
                return None


            # Iterate over YAML-defined resources in this section (images/videos)
            for entry in inputs.get(section, []):
                name = entry.get("name")
                if not name:
                    continue

                # If user asked for a specific resource, skip others
                if preferred_name and preferred_name != name:
                    continue

                # Try local resource first
                resolved = get_resource_path(
                    pipeline_name=None,
                    resource_type=resource_type,
                    arch=None,
                    model=name,
                )
                if resolved and resolved.exists():
                    return str(resolved)

                # Download requested resource if missing
                if preferred_name:
                    return download_if_missing(preferred_name)

                # Download default resource if required
                if allow_download_default:
                    return download_if_missing(name)

            return None

        # Prefer images first, then videos
        resolved = pick("images")
        if resolved:
            return resolved
        return pick("videos")

    # ------------------------------------------------
    # Case A: No --input provided
    # ------------------------------------------------
    if input_arg is None:
        resolved = resolve_tagged_resource(app_name=app, preferred_name=None, allow_download_default=False)
        if resolved:
            hailo_logger.info("No input provided; using default bundled input resource for %s: %s", app, resolved)
            return resolved

        hailo_logger.info("No input was provided and no input was found locally; downloading default input...")

        resolved = resolve_tagged_resource(app_name=app, preferred_name=None, allow_download_default=True)
        if resolved:
            hailo_logger.info("Default input downloaded successfully for: %s: %s", app, resolved)
            return resolved

        hailo_logger.error("No --input was provided and no bundled resource was found. "
            "Default input download failed.\n"
            "Please specify -i/--input with a file name, full file path, directory path, or camera'usb/rpi'."
        )
        sys.exit(1)

    # ------------------------------------------------
    # Case B: Camera input
    # ------------------------------------------------
    # Accept camera-related inputs:
    #   - Keywords: "usb", "rpi"
    #   - Linux device path: "/dev/videoX"
    #   - Windows camera index: "0", "1", ...
    if input_arg in CAMERA_KEYWORDS or input_arg.startswith("/dev/video") or input_arg.isdigit():
        return input_arg

    path_candidate = Path(input_arg)

    # ------------------------------------------------
    # Case C: Local path
    # ------------------------------------------------
    if path_candidate.exists():
        return str(path_candidate)

    # ------------------------------------------------
    # Case D: YAML resource name
    # ------------------------------------------------
    resource_path = resolve_tagged_resource(app_name=app, preferred_name=path_candidate.name, allow_download_default=False)
    if resource_path:
        return resource_path

    # ------------------------------------------------
    # Case E: Invalid input
    # ------------------------------------------------
    hailo_logger.error(
        f"Input '{input_arg}' does not exist as a local file or directory, "
        "and was not found as a downloadable resource.\n"
        "Please provide a full file path, directory path, a resource name, or a camera source: 'usb' / 'rpi'."
    )
    sys.exit(1)


def _map_app_to_resource_group(app_name: str) -> str:
    app_mapping = {
        "object_detection": "detection",
        "object_detection_onnx_postproc": "detection",
        "pose_estimation_onnx_postproc": "pose_estimation",
        "simple_detection": "simple_detection",
        "instance_segmentation": "instance_segmentation",
        "super_resolution": "super_resolution",
    }
    return app_mapping.get(app_name, app_name)


def resolve_output_resolution_arg(res_arg: Optional[list[str]]) -> Optional[Tuple[int, int]]:
    """
    Parse --output-resolution argument.

    Supported:
      --output-resolution sd|hd|fhd
      --output-resolution 1920 1080
    """
    if res_arg is None:
        return None

    # Single token: preset name (sd/hd/fhd)
    if len(res_arg) == 1:
        key = res_arg[0]
        if key in CAMERA_RESOLUTION_MAP:
            return CAMERA_RESOLUTION_MAP[key]
        raise ValueError(
            f"Invalid --output-resolution value '{key}'. "
            "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
        )

    # Two tokens: custom width/height
    if len(res_arg) == 2 and all(x.isdigit() for x in res_arg):
        w, h = map(int, res_arg)
        if w <= 0 or h <= 0:
            raise ValueError("Custom --output-resolution width/height must be positive integers.")
        return (w, h)

    raise ValueError(
        f"Invalid --output-resolution value: {res_arg}. "
        "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
    )


def handle_and_resolve_args(args, APP_NAME: str) -> None:
    """
    Handle common CLI argument logic for Hailo applications.

    This function:
    - Resolves the HEF path for the given application
    - Resolves the input source (camera / video / image)
    - Resolves output resolution if the flag exists
    - Ensures a valid output directory exists

    Notes:
    - This helper is intended mainly for standalone applications.

    Args:
        args: Parsed args from the application
        APP_NAME: The application name for model/input resolution
    """
    args.hef_path = resolve_hef_path(hef_path=args.hef_path, app_name=APP_NAME)
    if args.hef_path is None:
        hailo_logger.error("Failed to resolve HEF path for %s", APP_NAME)
        sys.exit(1)

    args.input = resolve_input_arg(APP_NAME, args.input)
    if args.input is None:
        hailo_logger.error("Failed to resolve input source for %s", APP_NAME)
        sys.exit(1)

    if hasattr(args, "output_dir"):
        try:
            if args.output_dir is None:
                args.output_dir = os.path.join(os.getcwd(), "output")
                os.makedirs(args.output_dir, exist_ok=True)
        except ValueError as e:
            hailo_logger.error(str(e))
            sys.exit(1)

    if hasattr(args, "output_resolution"):
        try:
            args.output_resolution = resolve_output_resolution_arg(args.output_resolution)
        except ValueError as e:
            hailo_logger.error(str(e))
            sys.exit(1)


def is_raspberry_pi():
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return RPI_POSSIBLE_NAME in model
    except:
        return False


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
