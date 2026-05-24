#!/usr/bin/env python3
"""
Unified Configuration Manager for Hailo Apps Infrastructure.

This module provides a single, consistent interface for loading and
querying all configuration files used throughout the project.

Configuration Files Managed:
    - config.yaml: Main installation/runtime settings
    - resources_config.yaml: Model, video, image resource definitions
    - test_definition_config.yaml: Test framework structure and suites
    - test_control.yaml: Test execution control settings

Usage:
    from hailo_apps.config import config_manager

    # Load main config
    main_config = config_manager.get_main_config()

    # Query resources
    models = config_manager.get_models_for_app("detection", "hailo8")

    # Query test definitions
    app_def = config_manager.get_app_definition("detection")
    test_suites = config_manager.get_test_suites_for_app("detection", "default")

    # Query test control
    run_time = config_manager.get_control_parameter("default_run_time")
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


@dataclass(frozen=True)
class AppDefinition:
    """Represents an application definition from test definition config.
    
    Attributes:
        name: Display name of the application
        description: Brief description
        module: Python module path (e.g., "hailo_apps.python.pipeline_apps.detection.detection")
        script: Script path relative to repo root
        cli: CLI command name (e.g., "hailo-detect")
        default_test_suites: Test suites to run by default
        extra_test_suites: Additional test suites for extended testing
    """
    name: str
    description: str
    module: str
    script: str
    cli: str
    default_test_suites: tuple[str, ...]
    extra_test_suites: tuple[str, ...]


@dataclass(frozen=True)
class TestSuite:
    """Represents a test suite definition.
    
    Attributes:
        name: Suite identifier
        description: Brief description of what the suite tests
        flags: Command-line flags to use for this suite
    """
    name: str
    description: str
    flags: tuple[str, ...]


# =============================================================================
# Path Resolution
# =============================================================================

class ConfigPaths:
    """Centralized path resolution for all configuration files.
    
    This class provides a single source of truth for locating configuration
    files, handling both installed package and development scenarios.
    """

    _repo_root: Optional[Path] = None

    @classmethod
    def _get_repo_root(cls) -> Path:
        """Get the repository root directory."""
        if cls._repo_root is None:
            # This file is at: hailo_apps/config/config_manager.py
            # Repo root is 2 levels up
            cls._repo_root = Path(__file__).resolve().parents[2]
        return cls._repo_root

    @classmethod
    def _get_config_dir(cls) -> Path:
        """Get the configuration directory.
        
        First tries package location (when installed via pip),
        then falls back to repo location (development mode).
        """
        # First try package location
        try:
            
            package_dir = Path(__file__).parent
            if package_dir.exists():
                return package_dir
        except (ImportError, AttributeError):
            pass
        # Fallback to repo location
        return Path(__file__).parent

    @classmethod
    def repo_root(cls) -> Path:
        """Get the repository root path."""
        return cls._get_repo_root()

    @classmethod
    def main_config(cls) -> Path:
        """Get path to config.yaml."""
        return cls._get_config_dir() / "config.yaml"

    @classmethod
    def resources_config(cls) -> Path:
        """Get path to resources_config.yaml."""
        return cls._get_config_dir() / "resources_config.yaml"

    @classmethod
    def test_definition_config(cls) -> Path:
        """Get path to test_definition_config.yaml."""
        return cls._get_config_dir() / "test_definition_config.yaml"

    @classmethod
    def test_control_config(cls) -> Path:
        """Get path to test_control.yaml (in tests directory)."""
        return cls._get_repo_root() / "tests" / "test_control.yaml"


# =============================================================================
# Base Loading Utilities
# =============================================================================

def _is_none_value(value: Any) -> bool:
    """Check if a value represents None (handles YAML None parsing).
    
    YAML files may contain None as:
    - Python None (null in YAML)
    - String "None" or "none"
    
    Args:
        value: The value to check
        
    Returns:
        True if the value represents None
    """
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == "none":
        return True
    return False


@lru_cache(maxsize=8)
def _load_yaml_cached(path: str) -> dict:
    """Load YAML file with caching.
    
    Args:
        path: Path to YAML file (string for hashability with lru_cache)
        
    Returns:
        Parsed YAML as dictionary
        
    Raises:
        ConfigError: If file not found or invalid YAML
    """
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
    """Load YAML file, optionally with caching.
    
    Args:
        path: Path to YAML file
        use_cache: Whether to use cached result if available
        
    Returns:
        Parsed YAML as dictionary
    """
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
# Main Config API (config.yaml)
# =============================================================================

def get_main_config(use_cache: bool = True) -> dict:
    """Get the main configuration (config.yaml).
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete configuration dictionary
    """
    return _load_yaml(ConfigPaths.main_config(), use_cache)


def get_valid_versions(key: str) -> list[str]:
    """Get valid versions for a configuration key.
    
    Args:
        key: Version key (e.g., "hailort", "tappas", "hailo_arch")
        
    Returns:
        List of valid version strings
    """
    config = get_main_config()
    valid_versions = config.get("valid_versions", {})
    return valid_versions.get(key, [])


def get_model_zoo_version_for_arch(arch: str) -> str:
    """Get the Model Zoo version for a given architecture.
    
    Args:
        arch: Hailo architecture ("hailo8", "hailo8l", "hailo10h")
        
    Returns:
        Model Zoo version string (e.g., "v2.17.0")
    """
    config = get_main_config()
    mapping = config.get("model_zoo_mapping", {})
    return mapping.get(arch, "v2.17.0")


def get_venv_config() -> dict:
    """Get virtual environment configuration.
    
    Returns:
        Dictionary with 'name' and 'use_system_site_packages' keys
    """
    config = get_main_config()
    return config.get("venv", {"name": "venv_hailo_apps", "use_system_site_packages": True})


def get_resources_path_config() -> dict:
    """Get resources path configuration.
    
    Returns:
        Dictionary with 'path', 'root', 'env_file', 'download_group', 'dirs' keys
    """
    config = get_main_config()
    return config.get("resources", {})


def get_model_zoo_mapping() -> dict:
    """Get the model zoo version mapping for architectures.
    
    Returns:
        Dictionary mapping architecture names to Model Zoo versions
        e.g., {"hailo8": "v2.17.0", "hailo8l": "v2.17.0", "hailo10h": "v2.17.0"}
    """
    config = get_main_config()
    return config.get("model_zoo_mapping", {})


# =============================================================================
# Resources Config API (resources_config.yaml)
# =============================================================================

def get_resources_config(use_cache: bool = True) -> dict:
    """Get the resources configuration (resources_config.yaml).
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete resources configuration dictionary
    """
    return _load_yaml(ConfigPaths.resources_config(), use_cache)


def get_available_apps() -> list[str]:
    """Get list of all available application names from resources config.
    
    Returns:
        Sorted list of application names (excludes 'videos' and 'images' sections)
    """
    config = get_resources_config()
    shared_keys = {"videos", "images"}
    return sorted(
        k for k, v in config.items() if isinstance(v, dict) and k not in shared_keys
    )


def get_supported_architectures(app_name: str) -> list[str]:
    """Get architectures that have valid models for an app.
    
    Args:
        app_name: Application name (e.g., "detection")
        
    Returns:
        Sorted list of supported architecture names
    """
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
    """Extract ModelEntry objects from config entries.
    
    Handles various config formats:
    - Single model dict: {"name": "model", "source": "mz"}
    - List of models: [{"name": "model1"}, {"name": "model2"}]
    - None/null values
    
    Args:
        entries: Raw entries from config
        app_type_filter: If provided, only return models supporting this app type
                        ("pipeline" or "standalone"). If None, return all models.
        
    Returns:
        List of ModelEntry objects
    """
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
    """Get default model entries for an app and architecture.
    
    Args:
        app_name: Application name (e.g., "detection")
        arch: Hailo architecture (e.g., "hailo8")
        app_type: Filter by app type ("pipeline" or "standalone"). If None, return all.
        
    Returns:
        List of default ModelEntry objects
    """
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("default"), app_type_filter=app_type)


def get_extra_models(app_name: str, arch: str, app_type: Optional[str] = None) -> list[ModelEntry]:
    """Get extra model entries for an app and architecture.
    
    Args:
        app_name: Application name (e.g., "detection")
        arch: Hailo architecture (e.g., "hailo8")
        app_type: Filter by app type ("pipeline" or "standalone"). If None, return all.
        
    Returns:
        List of extra ModelEntry objects
    """
    config = get_resources_config()
    app_config = config.get(app_name, {})
    arch_models = app_config.get("models", {}).get(arch, {})
    return _extract_model_entries(arch_models.get("extra"), app_type_filter=app_type)


def get_all_models(app_name: str, arch: str, app_type: Optional[str] = None) -> list[ModelEntry]:
    """Get all model entries (default + extra) for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        app_type: Filter by app type ("pipeline" or "standalone"). If None, return all.
        
    Returns:
        Combined list of default and extra ModelEntry objects
    """
    return get_default_models(app_name, arch, app_type) + get_extra_models(app_name, arch, app_type)


def get_model_names(app_name: str, arch: str, tier: str = "all", app_type: Optional[str] = None) -> list[str]:
    """Get model names for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        tier: "default", "extra", or "all"
        app_type: Filter by app type ("pipeline" or "standalone"). If None, return all.
        
    Returns:
        List of model name strings
    """
    if tier == "default":
        models = get_default_models(app_name, arch, app_type)
    elif tier == "extra":
        models = get_extra_models(app_name, arch, app_type)
    else:
        models = get_all_models(app_name, arch, app_type)

    return [m.name for m in models]


def get_default_model_name(app_name: str, arch: str, app_type: Optional[str] = None) -> Optional[str]:
    """Get the first default model name for an app and architecture.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        app_type: Filter by app type ("pipeline" or "standalone"). If None, return all.
        
    Returns:
        Model name string or None if no default model
    """
    models = get_default_models(app_name, arch, app_type)
    return models[0].name if models else None


def get_model_info(
    app_name: str,
    arch: str,
    model_name: str,
    app_type: Optional[str] = None,
) -> Optional[ModelEntry]:
    """Get full model info for a specific model.
    
    Args:
        app_name: Application name
        arch: Hailo architecture
        model_name: Name of the model to find
        
    Returns:
        ModelEntry if found, None otherwise
    """
    for model in get_all_models(app_name, arch, app_type=app_type):
        if model.name == model_name:
            return model
    return None


def get_videos() -> list[str]:
    """Get list of video filenames from resources config.
    
    Returns:
        List of video filename strings
    """
    config = get_resources_config()
    videos = []
    for entry in config.get("videos", []):
        if isinstance(entry, dict) and entry.get("name"):
            videos.append(entry["name"])
        elif isinstance(entry, str):
            videos.append(entry)
    return videos


def get_images() -> list[str]:
    """Get list of image filenames from resources config.
    
    Returns:
        List of image filename strings
    """
    config = get_resources_config()
    images = []
    for entry in config.get("images", []):
        if isinstance(entry, dict) and entry.get("name"):
            images.append(entry["name"])
        elif isinstance(entry, str):
            images.append(entry)
    return images


def get_npy_files() -> list[str]:
    """Get list of NPY filenames from resources config.
    
    Returns:
        List of NPY filename strings
    """
    config = get_resources_config()
    npy_files = []
    for entry in config.get("npy", []):
        if isinstance(entry, dict) and entry.get("name"):
            npy_files.append(entry["name"])
        elif isinstance(entry, str):
            npy_files.append(entry)
    return npy_files


def get_json_files(app_name: str = None) -> list[str]:
    """Get JSON config filenames from the shared json section.
    
    Args:
        app_name: Ignored (kept for backward compatibility). All JSON files are shared.
        
    Returns:
        List of JSON filename strings
    """
    config = get_resources_config()
    json_files = []
    for entry in config.get("json", []):
        if isinstance(entry, dict) and entry.get("name"):
            json_files.append(entry["name"])
        elif isinstance(entry, str):
            json_files.append(entry)
    return json_files


def get_all_json_files() -> list[str]:
    """Get all JSON files from the shared json section.
    
    Returns:
        List of JSON filename strings
    """
    return get_json_files()


def is_gen_ai_app(app_name: str) -> bool:
    """Check if an app uses gen-ai-mz source models.
    
    Args:
        app_name: Application name
        
    Returns:
        True if any model has source "gen-ai-mz"
    """
    for arch in get_supported_architectures(app_name):
        for model in get_all_models(app_name, arch):
            if model.source == "gen-ai-mz":
                return True
    return False


# =============================================================================
# Inputs Config API (images/videos per app based on tags)
# =============================================================================

def _get_resources_by_tag(section: str, app_name: str) -> list[dict]:
    """Get resources from a section that are tagged for an app.
    
    Args:
        section: Section name ("videos", "images", "json")
        app_name: Application name to match in tags
        
    Returns:
        List of resource entry dicts that match the app
    """
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
    """Get videos that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        
    Returns:
        List of video entry dicts that match the app
    """
    return _get_resources_by_tag("videos", app_name)


def get_images_for_app(app_name: str) -> list[dict]:
    """Get images that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        
    Returns:
        List of image entry dicts that match the app
    """
    return _get_resources_by_tag("images", app_name)


def get_json_for_app(app_name: str) -> list[dict]:
    """Get JSON config files that are tagged for a specific application.
    
    Args:
        app_name: Application name (e.g., "detection", "tiling")
        
    Returns:
        List of JSON entry dicts that match the app
    """
    return _get_resources_by_tag("json", app_name)


def get_inputs_for_app(app_name: str, is_standalone: bool = False) -> dict:
    """Get input resources for an application based on app type.
    
    - Standalone apps (is_standalone=True): Returns both videos AND images
    - Pipeline apps (is_standalone=False): Returns only videos
    
    Args:
        app_name: Application name (e.g., "detection", "pose_estimation")
        is_standalone: If True, include images; if False, only videos
        
    Returns:
        Dictionary with 'videos' and optionally 'images' lists for the app
    """
    # Standalone apps use the base app tag names (without _standalone suffix)
    app_name = base_app_name(app_name) if is_standalone or is_standalone_app_name(app_name) else app_name
    result = {
        "videos": get_videos_for_app(app_name),
    }
    
    if is_standalone:
        result["images"] = get_images_for_app(app_name)
    
    return result


def get_all_tags() -> set[str]:
    """Get all unique tags used across videos, images, and json.
    
    Returns:
        Set of all tag strings found in resources config
    """
    config = get_resources_config()
    tags = set()
    
    for section in ["videos", "images", "json"]:
        for entry in config.get(section, []):
            if isinstance(entry, dict):
                entry_tags = entry.get("tag", [])
                if isinstance(entry_tags, str):
                    entry_tags = [entry_tags]
                tags.update(entry_tags)
    
    return tags


def get_apps_with_inputs() -> set[str]:
    """Get all app names that have tagged inputs (videos/images/json).
    
    Returns:
        Set of app names that have at least one tagged resource
    """
    return get_all_tags()


# =============================================================================
# Standalone helpers
# =============================================================================

def is_standalone_app_name(app_name: str) -> bool:
    """Return True if the app name uses the _standalone suffix."""
    return app_name.endswith(STANDALONE_SUFFIX)


def base_app_name(app_name: str) -> str:
    """Strip the _standalone suffix if present."""
    return app_name[: -len(STANDALONE_SUFFIX)] if is_standalone_app_name(app_name) else app_name


def get_defined_standalone_apps() -> list[str]:
    """List app definitions that are marked as standalone (_standalone suffix)."""
    return [name for name in get_defined_apps() if is_standalone_app_name(name)]


def get_standalone_app_definition(app_name: str) -> Optional[AppDefinition]:
    """Get standalone app definition (expects _standalone suffix)."""
    return get_app_definition(app_name)


def get_standalone_test_suites_for_app(app_name: str, mode: str = "default") -> list[str]:
    """Get test suites for a standalone app."""
    return get_test_suites_for_app(app_name, mode)


def get_standalone_model_names(app_name: str, arch: str, tier: str = "default") -> list[str]:
    """Get model names for a standalone app, mapped to its base app resources."""
    base = base_app_name(app_name)
    return get_model_names(base, arch, tier)


def get_standalone_default_model_name(app_name: str, arch: str) -> Optional[str]:
    """Get default model name for a standalone app."""
    base = base_app_name(app_name)
    return get_default_model_name(base, arch)


# =============================================================================
# Test Definition Config API (test_definition_config.yaml)
# =============================================================================

def get_test_definition_config(use_cache: bool = True) -> dict:
    """Get the test definition configuration.
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete test definition configuration dictionary
    """
    return _load_yaml(ConfigPaths.test_definition_config(), use_cache)


def get_app_definition(app_name: str) -> Optional[AppDefinition]:
    """Get application definition by name.
    
    Args:
        app_name: Application name (e.g., "detection")
        
    Returns:
        AppDefinition dataclass or None if not found
    """
    config = get_test_definition_config()
    apps = config.get("apps", {})
    app_data = apps.get(app_name)

    if not app_data:
        return None

    return AppDefinition(
        name=app_data.get("name", app_name),
        description=app_data.get("description", ""),
        module=app_data.get("module", ""),
        script=app_data.get("script", ""),
        cli=app_data.get("cli", ""),
        default_test_suites=tuple(app_data.get("default_test_suites", [])),
        extra_test_suites=tuple(app_data.get("extra_test_suites", [])),
    )


def get_defined_apps() -> list[str]:
    """Get list of all defined application names from test definition config.
    
    Returns:
        List of application name strings
    """
    config = get_test_definition_config()
    return list(config.get("apps", {}).keys())


def get_test_suite(suite_name: str) -> Optional[TestSuite]:
    """Get test suite definition by name.
    
    Args:
        suite_name: Test suite name (e.g., "basic_show_fps")
        
    Returns:
        TestSuite dataclass or None if not found
    """
    config = get_test_definition_config()
    suites = config.get("test_suites", {})
    suite_data = suites.get(suite_name)

    if not suite_data:
        return None

    return TestSuite(
        name=suite_name,
        description=suite_data.get("description", ""),
        flags=tuple(suite_data.get("flags", [])),
    )


def get_all_test_suites() -> list[str]:
    """Get all test suite names.
    
    Returns:
        List of test suite name strings
    """
    config = get_test_definition_config()
    return list(config.get("test_suites", {}).keys())


def get_test_suites_for_app(app_name: str, mode: str = "default") -> list[str]:
    """Get test suites for an app based on mode.
    
    Args:
        app_name: Application name
        mode: "default", "extra", or "all"
        
    Returns:
        List of test suite name strings
    """
    app_def = get_app_definition(app_name)
    if not app_def:
        return []

    if mode == "default":
        return list(app_def.default_test_suites)
    elif mode == "extra":
        return list(app_def.extra_test_suites)
    else:  # "all"
        return list(app_def.default_test_suites) + list(app_def.extra_test_suites)


def get_test_run_combination(name: str) -> Optional[dict]:
    """Get a test run combination by name.
    
    Args:
        name: Combination name (e.g., "ci_run", "all_default")
        
    Returns:
        Combination configuration dict or None if not found
    """
    config = get_test_definition_config()
    combinations = config.get("test_run_combinations", {})
    return combinations.get(name)


def get_all_test_run_combinations() -> list[str]:
    """Get all test run combination names.
    
    Returns:
        List of combination name strings
    """
    config = get_test_definition_config()
    return list(config.get("test_run_combinations", {}).keys())


def get_test_resources() -> dict:
    """Get test resources configuration.
    
    Returns:
        Dictionary with 'videos', 'photos', 'json_files' keys
    """
    config = get_test_definition_config()
    return config.get("resources", {})


# =============================================================================
# Test Control Config API (test_control.yaml)
# =============================================================================

def get_test_control_config(use_cache: bool = True) -> dict:
    """Get the test control configuration.
    
    Args:
        use_cache: Whether to use cached result
        
    Returns:
        Complete test control configuration dictionary
    """
    try:
        return _load_yaml(ConfigPaths.test_control_config(), use_cache)
    except ConfigError:
        # test_control.yaml may not exist in all environments
        return {}


def get_control_parameter(key: str, default: Any = None) -> Any:
    """Get a control parameter value.
    
    Args:
        key: Parameter name (e.g., "default_run_time", "term_timeout")
        default: Default value if not found
        
    Returns:
        Parameter value
    """
    config = get_test_control_config()
    params = config.get("control_parameters", {})
    return params.get(key, default)


def get_logging_config() -> dict:
    """Get logging configuration from test control.
    
    Returns:
        Dictionary with logging settings
    """
    config = get_test_control_config()
    return config.get("logging", {})


def get_enabled_run_methods() -> list[str]:
    """Get list of enabled run methods.
    
    Returns:
        List of enabled run method names (e.g., ["pythonpath", "cli"])
    """
    config = get_test_control_config()
    run_methods = config.get("run_methods", {})
    return [name for name, cfg in run_methods.items() if cfg.get("enabled", False)]


def get_custom_test_apps() -> dict[str, dict]:
    """Get custom test configuration for all apps.
    
    Returns:
        Dictionary mapping app names to their custom test config
    """
    config = get_test_control_config()
    custom_tests = config.get("custom_tests", {})
    if not custom_tests.get("enabled", False):
        return {}
    return custom_tests.get("apps", {})


def is_special_test_enabled(test_name: str) -> bool:
    """Check if a special test is enabled.
    
    Args:
        test_name: Special test name (e.g., "h8l_on_h8", "sanity_checks")
        
    Returns:
        True if the special test is enabled
    """
    config = get_test_control_config()
    special_tests = config.get("special_tests", {})
    return special_tests.get(test_name, {}).get("enabled", False)


def get_enabled_test_combinations() -> list[str]:
    """Get list of enabled test combinations.
    
    Returns:
        List of enabled combination names
    """
    config = get_test_control_config()
    test_combinations = config.get("test_combinations", {})
    return [name for name, cfg in test_combinations.items() if cfg.get("enabled", False)]


def get_custom_standalone_tests() -> dict[str, dict]:
    """Get standalone test configuration for standalone apps.
    
    Returns:
        Dictionary mapping standalone app names (_standalone suffix) to their config.
    """
    config = get_test_control_config()
    standalone_tests = config.get("standalone_tests", {})
    if not standalone_tests.get("enabled", False):
        return {}
    return standalone_tests.get("apps", {})


# =============================================================================
# Cache Management
# =============================================================================

def clear_cache():
    """Clear all cached configuration data.
    
    Call this when configuration files have been modified and you
    need to reload them.
    """
    _load_yaml_cached.cache_clear()


def reload_all():
    """Clear cache and reload all configurations.
    
    Useful for testing or when configs have been modified.
    """
    clear_cache()
    # Trigger reloads
    get_main_config()
    get_resources_config()
    get_test_definition_config()
    get_test_control_config()
