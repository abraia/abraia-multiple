"""Public utility facade.

Implementations live in focused modules, while this facade exposes the
supported SDK utility API.
"""

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
    "array_copy": (".arrays", "array_copy"),
    "array_from_image_buffer": (".arrays", "array_from_image_buffer"),
    "array_ndim": (".arrays", "array_ndim"),
    "array_shape": (".arrays", "array_shape"),
    "array_size": (".arrays", "array_size"),
    "array_squeeze": (".arrays", "array_squeeze"),
    "array_to_list": (".arrays", "array_to_list"),
    "as_array": (".arrays", "as_array"),
    "combined_mask": (".arrays", "combined_mask"),
    "compose_mask_layers": (".arrays", "compose_mask_layers"),
    "copy_mask": (".arrays", "copy_mask"),
    "encode_mask_overlay": (".arrays", "encode_mask_overlay"),
    "mask_array": (".arrays", "mask_array"),
    "mask_matches_image": (".arrays", "mask_matches_image"),
    "merge_masks": (".arrays", "merge_masks"),
    "resize_mask": (".arrays", "resize_mask"),
    "zeros_array": (".arrays", "zeros_array"),
    "NumpyEncoder": (".filesystem", "NumpyEncoder"),
    "get_type": (".filesystem", "get_type"),
    "list_dir": (".filesystem", "list_dir"),
    "load_data": (".filesystem", "load_data"),
    "load_json": (".filesystem", "load_json"),
    "load_text": (".filesystem", "load_text"),
    "make_dirs": (".filesystem", "make_dirs"),
    "md5sum": (".filesystem", "md5sum"),
    "save_data": (".filesystem", "save_data"),
    "save_json": (".filesystem", "save_json"),
    "save_text": (".filesystem", "save_text"),
    "encode_image": (".image", "encode_image"),
    "image_base64": (".image", "image_base64"),
    "load_image": (".image", "load_image"),
    "save_image": (".image", "save_image"),
    "show_image": (".image", "show_image"),
    "API_URL": (".remote", "API_URL"),
    "ARTIFACT_RESOLVER": (".remote", "ARTIFACT_RESOLVER"),
    "ArtifactReference": (".remote", "ArtifactReference"),
    "ArtifactResolver": (".remote", "ArtifactResolver"),
    "HEADERS": (".remote", "HEADERS"),
    "download_file": (".remote", "download_file"),
    "download_url": (".remote", "download_url"),
    "get_remote_file_size": (".remote", "get_remote_file_size"),
    "is_managed_model_path": (".remote", "is_managed_model_path"),
    "is_url": (".remote", "is_url"),
    "load_url": (".remote", "load_url"),
    "load_url_bytes": (".remote", "load_url_bytes"),
    "resolve_model_file": (".remote", "resolve_model_file"),
    "temporal_src": (".remote", "temporal_src"),
    "url_path": (".remote", "url_path"),
    "Sketcher": (".display", "Sketcher"),
    "Window": (".display", "Window"),
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
    "ARTIFACT_RESOLVER",
    "ArtifactReference",
    "ArtifactResolver",
    "HEADERS",
    "NumpyEncoder",
    "Sketcher",
    "Window",
    "array_copy",
    "array_from_image_buffer",
    "array_ndim",
    "array_shape",
    "array_size",
    "array_squeeze",
    "array_to_list",
    "as_array",
    "combined_mask",
    "compose_mask_layers",
    "copy_mask",
    "download_file",
    "download_url",
    "encode_image",
    "encode_mask_overlay",
    "get_color",
    "get_providers",
    "get_remote_file_size",
    "get_type",
    "image_base64",
    "is_managed_model_path",
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
    "mask_matches_image",
    "md5sum",
    "merge_masks",
    "render_counter",
    "render_region",
    "render_results",
    "resize_mask",
    "resolve_model_file",
    "save_data",
    "save_image",
    "save_json",
    "save_text",
    "show_image",
    "temporal_src",
    "url_path",
    "zeros_array",
]
