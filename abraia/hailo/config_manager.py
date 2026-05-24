#!/usr/bin/env python3
"""
Unified Configuration Manager for Hailo Apps Infrastructure.

This module provides a single, consistent interface for loading and
querying configuration files used throughout the project.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


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

def _is_none_value(value: Any) -> bool:
    """Check if a value represents None (handles YAML None parsing)."""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == "none":
        return True
    return False


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


def _load_yaml(path: Path, use_cache: bool = True) -> dict:
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
    return _load_yaml(ConfigPaths.resources_config(), use_cache)


def get_supported_architectures(app_name: str) -> list[str]:
    """Get architectures that have valid models for an app."""
    config = get_resources_config()
    app_config = config.get(app_name, {})
    models_config = app_config.get("models", {})

    supported = []
    for arch, arch_models in models_config.items():
        if isinstance(arch_models, dict):
            default = arch_models.get("default")
            extra = arch_models.get("extra", [])
            if not _is_none_value(default) or extra:
                supported.append(arch)

    return sorted(supported)


def _extract_model_entries(entries: Any, app_type_filter: Optional[str] = None) -> list[ModelEntry]:
    """Extract ModelEntry objects from config entries."""
    if _is_none_value(entries):
        return []

    entries_list = entries if isinstance(entries, list) else [entries]
    models = []

    for entry in entries_list:
        if _is_none_value(entry):
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            if name and not _is_none_value(name):
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
        elif isinstance(entry, str) and not _is_none_value(entry):
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
