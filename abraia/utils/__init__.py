"""Public utility facade.

Implementations live in focused modules, while this facade exposes the
supported SDK utility API.
"""

from .arrays import (
    array_copy,
    array_from_image_buffer,
    array_ndim,
    array_shape,
    array_size,
    array_squeeze,
    array_to_list,
    as_array,
    compose_mask_layers,
    encode_mask_overlay,
    mask_array,
    merge_masks,
    resize_mask,
    zeros_array,
)
from .filesystem import (
    NumpyEncoder,
    get_type,
    list_dir,
    load_data,
    load_json,
    load_text,
    make_dirs,
    md5sum,
    save_data,
    save_json,
    save_text,
)
from .image import encode_image, image_base64, load_image, save_image, show_image
from .remote import (
    API_URL,
    HEADERS,
    download_file,
    download_url,
    get_remote_file_size,
    is_url,
    load_url,
    load_url_bytes,
    temporal_src,
    url_path,
)


def get_providers():
    """Return available ONNX Runtime providers in preferred order."""
    import onnxruntime as ort

    preferred = (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    available = set(ort.get_available_providers())
    return [provider for provider in preferred if provider in available]


_LAZY_EXPORTS = {
    "FrameContext": (".pipeline", "FrameContext"),
    "LineCounterStage": (".pipeline", "LineCounterStage"),
    "Pipeline": (".pipeline", "Pipeline"),
    "RegionFilterStage": (".pipeline", "RegionFilterStage"),
    "RegionTimerStage": (".pipeline", "RegionTimerStage"),
    "TrackerStage": (".pipeline", "TrackerStage"),
    "Sketcher": (".display", "Sketcher"),
    "Window": (".display", "Window"),
    "Video": (".video", "Video"),
    "VideoDisplay": (".stream", "VideoDisplay"),
    "VideoInput": (".stream", "VideoInput"),
    "get_color": (".draw", "get_color"),
    "render_counter": (".draw", "render_counter"),
    "render_region": (".draw", "render_region"),
    "render_results": (".draw", "render_results"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "API_URL",
    "HEADERS",
    "FrameContext",
    "LineCounterStage",
    "NumpyEncoder",
    "Pipeline",
    "RegionFilterStage",
    "RegionTimerStage",
    "Sketcher",
    "TrackerStage",
    "Video",
    "VideoDisplay",
    "VideoInput",
    "Window",
    "array_copy",
    "array_from_image_buffer",
    "array_ndim",
    "array_shape",
    "array_size",
    "array_squeeze",
    "array_to_list",
    "as_array",
    "compose_mask_layers",
    "download_file",
    "download_url",
    "encode_image",
    "encode_mask_overlay",
    "get_color",
    "get_providers",
    "get_remote_file_size",
    "get_type",
    "image_base64",
    "is_url",
    "list_dir",
    "load_data",
    "load_image",
    "load_json",
    "load_text",
    "load_url",
    "load_url_bytes",
    "make_dirs",
    "mask_array",
    "md5sum",
    "merge_masks",
    "render_counter",
    "render_region",
    "render_results",
    "resize_mask",
    "save_data",
    "save_image",
    "save_json",
    "save_text",
    "show_image",
    "temporal_src",
    "url_path",
    "zeros_array",
]
