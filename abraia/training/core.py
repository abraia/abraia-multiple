"""Shared dataset state and remote dataset helpers."""

import re
from pathlib import Path

from ..client import Abraia


HAILO_EXPORT_TARGETS = (
    "hailo8",
    "hailo8l",
    "hailo10h",
    "hailo15h",
    "hailo15l",
)


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


def save_hailo_bundle(
    client,
    project,
    model_name,
    source_dir,
    target,
    version=None,
    metadata=None,
):
    """Upload a native Ultralytics Hailo export and its sidecar files.

    The Hailo exporter returns a directory rather than a single model file.
    Keep that directory intact for Ultralytics consumers, while also exposing
    a flat HEF path for Abraia's existing Hailo runtime and model pairing.
    """
    if target not in HAILO_EXPORT_TARGETS:
        choices = ", ".join(HAILO_EXPORT_TARGETS)
        raise ValueError(f"Unsupported Hailo target '{target}'. Use: {choices}")

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Hailo export did not produce a directory: {source_dir}")

    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    hef_files = [path for path in files if path.suffix.lower() == ".hef"]
    if not hef_files:
        raise ValueError(f"Hailo export contains no HEF file: {source_dir}")
    if len(hef_files) > 1:
        raise ValueError(
            f"Hailo export contains multiple HEF files: {', '.join(map(str, hef_files))}"
        )

    versioned_name = model_name if version is None else f"{model_name}_v{version}"
    remote_dir = f"{project}/{versioned_name}_{target}"
    remote_hef = f"{project}/{versioned_name}_{target}.hef"
    uploaded = []

    for source in files:
        relative = source.relative_to(source_dir).as_posix()
        destination = f"{remote_dir}/{relative}"
        client.upload_file(str(source), destination)
        uploaded.append(destination)

    # Keep a flat HEF path compatible with paired_hailo_uri(). The complete
    # bundle above remains available for Ultralytics' own Hailo loader.
    client.upload_file(str(hef_files[0]), remote_hef)
    uploaded.append(remote_hef)

    manifest_path = f"{remote_dir}/abraia.json"
    if metadata is not None and callable(getattr(client, "save_json", None)):
        client.save_json(manifest_path, metadata)
        uploaded.append(manifest_path)

    return {
        "format": "hailo",
        "target": target,
        "version": version,
        "bundle": remote_dir,
        "hef": remote_hef,
        "files": uploaded,
        "manifest": manifest_path if metadata is not None else None,
    }


__all__ = [
    "next_model_version",
    "HAILO_EXPORT_TARGETS",
    "save_hailo_bundle",
    "save_versioned_model",
    "versioned_model_paths",
]
