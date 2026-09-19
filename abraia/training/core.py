"""Shared dataset state and remote dataset helpers."""

import re

from ..client import Abraia


def _resolve_client(client=None):
    """Return an injected client or create one lazily for this operation."""
    return client if client is not None else Abraia()


def next_model_version(client, project, model_name):
    """Return the next version in the flat model filename sequence."""
    pattern = re.compile(rf"^{re.escape(model_name)}_v(\d+)\.onnx$")
    files, _ = client.list_files(f"{project}/")
    versions = []
    for file_data in files or []:
        match = pattern.fullmatch(file_data.get("name", ""))
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def versioned_model_paths(client, project, model_name):
    """Return collision-checked paths for the next versioned model."""
    version = next_model_version(client, project, model_name)
    while True:
        versioned_name = f"{model_name}_v{version}"
        model_path = f"{project}/{versioned_name}.onnx"
        metadata_path = f"{project}/{versioned_name}.json"
        check_file = getattr(client, "check_file", None)
        if not callable(check_file):
            break
        if not check_file(model_path) and not check_file(metadata_path):
            break
        version += 1
    return {
        "version": version,
        "name": f"{versioned_name}.onnx",
        "onnx": model_path,
        "metadata": metadata_path,
    }


def save_versioned_model(client, project, model_name, source, metadata):
    """Upload an ONNX file and matching metadata using a safe version name."""
    check_file = getattr(client, "check_file", None)
    for _attempt in range(3):
        paths = versioned_model_paths(client, project, model_name)
        try:
            client.upload_file(source, paths["onnx"])
        except Exception:
            # A concurrent save may win the preflight check between version
            # allocation and upload. Retry only when the target now exists.
            if not callable(check_file) or not check_file(paths["onnx"]):
                raise
            continue
        try:
            client.save_json(paths["metadata"], metadata)
        except Exception:
            remove_file = getattr(client, "remove_file", None)
            if callable(remove_file):
                try:
                    remove_file(paths["onnx"])
                except Exception:
                    pass
            raise
        return paths
    raise RuntimeError("Could not allocate a unique model version")


__all__ = [
    "next_model_version",
    "save_versioned_model",
    "versioned_model_paths",
]
