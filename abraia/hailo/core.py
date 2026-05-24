from __future__ import annotations
"""Core helpers: arch detection, buffer utils, model resolution."""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from .defines import (
    DEFAULT_LOCAL_RESOURCES_PATH,
    HAILO_ARCH_KEY,
    HAILO_FILE_EXTENSION,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_NPY_DIR_NAME,
    # for get_resource_path
    RESOURCES_PHOTOS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    CAMERA_RESOLUTION_MAP,
    RESOURCE_TYPE_IMAGE,
    RESOURCE_TYPE_ONNX,
    RESOURCE_TYPE_VIDEO,
    RESOURCE_TYPE_MODEL,
    CAMERA_KEYWORDS,
)

from .hailo_logger import get_logger
from .installation_utils import detect_hailo_arch
from .config_manager import get_default_models, get_inputs_for_app, get_supported_architectures
hailo_logger = get_logger(__name__)


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
    from .config_manager import (
        get_model_names,
        get_default_model_name,
    )

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
        from .download_resources import download_resources

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
