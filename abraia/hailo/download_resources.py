from __future__ import annotations
"""
Resource Download Manager for Hailo Apps Infrastructure.
Optimized system for downloading ML models, videos, images, and config files.
"""

import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .hailo_logger import get_logger
from .config_manager import get_images_for_app, get_videos_for_app, _load_yaml as load_config
from .defines import (
    DEFAULT_RESOURCES_CONFIG_PATH,
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO_FILE_EXTENSION,
    MODEL_ZOO_URL,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_NPY_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_PHOTOS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
    S3_RESOURCES_BASE_URL,
    RESOURCE_TYPE_MODEL,
    RESOURCE_TYPE_IMAGE,
    RESOURCE_TYPE_VIDEO,
    RESOURCE_TYPES,
)
from .installation_utils import detect_hailo_arch

hailo_logger = get_logger(__name__)
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
