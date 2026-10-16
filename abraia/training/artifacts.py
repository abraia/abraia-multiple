"""Model catalog, versioning, and export helpers."""

import importlib
from importlib import metadata
import os
import re
import platform
from pathlib import Path
import subprocess
import sys

from ..dataset import _resolve_client
from ..tasks import normalize_task
from .model_data import metric_average


HAILO_EXPORT_TARGETS = (
    "hailo8",
    "hailo8l",
    "hailo10h",
    "hailo15h",
    "hailo15l",
)

HAILO_DFC_WHEELS = {
    "hailo8": (
        "multiple/hailo8/"
        "hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl"
    ),
    "hailo8l": (
        "multiple/hailo8/"
        "hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl"
    ),
    "hailo10h": (
        "multiple/hailo15/"
        "hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl"
    ),
    "hailo15h": (
        "multiple/hailo15/"
        "hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl"
    ),
    "hailo15l": (
        "multiple/hailo15/"
        "hailo_dataflow_compiler-5.2.0-py3-none-linux_x86_64.whl"
    ),
}

HAILO_DFC_VERSIONS = {
    "hailo8": "3.33.1",
    "hailo8l": "3.33.1",
    "hailo10h": "5.2.0",
    "hailo15h": "5.2.0",
    "hailo15l": "5.2.0",
}


def list_models(project, client=None, sort_by_accuracy=False):
    """Return ONNX model names stored in a remote project.

    When ``sort_by_accuracy`` is true, models with saved accuracy metrics are
    returned first, from highest to lowest score. Models without a score keep
    their relative order after the ranked models.
    """
    client = _resolve_client(client)
    if sort_by_accuracy:
        records = list_model_records(project, client, sort_by_accuracy=True)
        return [
            record["name"] for record in records
            if str(record["name"]).lower().endswith(".onnx")
        ]

    files = client.list_files(f"{project}/")[0]
    return [
        file_data["name"]
        for file_data in files
        if file_data["name"].endswith(".onnx")
    ]


def list_model_records(project, client=None, sort_by_accuracy=False):
    """Return model records with optional metadata and accuracy ordering."""
    client = _resolve_client(client)
    records = []
    files = client.list_files(f"{project}/")[0]
    names = [
        file_data["name"]
        for file_data in files
        if file_data["name"].lower().endswith((".onnx", ".joblib"))
    ]
    for name in names:
        record = {"name": name, "metrics": {}}
        if hasattr(client, "load_json"):
            stem = os.path.splitext(str(name))[0]
            try:
                metadata = client.load_json(f"{project}/{stem}.json")
            except Exception:
                metadata = None
            if isinstance(metadata, dict):
                record.update({
                    "metrics": metadata.get("metrics") or {},
                    "classes": metadata.get("classes") or [],
                    "task": normalize_task(metadata.get("task")),
                    "kind": metadata.get("kind"),
                    "backend": metadata.get("backend"),
                })
                if str(name).lower().endswith(".joblib"):
                    record["kind"] = metadata.get("kind") or "multispectral"
                    record["task"] = normalize_task(
                        metadata.get("task"), default="classification"
                    )
        if str(name).lower().endswith(".joblib"):
            record.setdefault("classes", [])
            record.setdefault("task", "classification")
            record.setdefault("kind", "multispectral")
        records.append(record)
    if sort_by_accuracy:
        def accuracy(record):
            metrics = record.get("metrics") or {}
            for key in ("acc", "accuracy", "mAP"):
                score = metric_average(metrics.get(key))
                if score is not None:
                    return score
            return None

        ranked = [(accuracy(record), record) for record in records]
        ranked.sort(key=lambda item: (item[0] is None, -(item[0] or 0.0)))
        records = [record for _, record in ranked]
    return records


def ensure_hailo_dfc(target):
    """Install the matching DFC wheel when local compilation needs it.

    Hailo compilation is supported only on Linux x86_64. On other hosts this
    function leaves the existing exporter error path untouched. The wheel is
    fetched from the global ``multiple`` namespace and installed into the
    interpreter running the SDK.
    """
    target = str(target or "").strip().lower()
    if target not in HAILO_DFC_WHEELS:
        raise ValueError(f"Unsupported Hailo target: {target}")
    if not _supports_hailo_dfc():
        return False

    expected_version = HAILO_DFC_VERSIONS[target]
    installed_version = _installed_dfc_version()
    if installed_version is None:
        needs_install = not _dfc_import_available()
    else:
        needs_install = installed_version != expected_version
    if not needs_install:
        return False

    from ..utils.remote import download_file

    wheel_path = download_file(HAILO_DFC_WHEELS[target])
    command = [sys.executable, "-m", "pip", "install"]
    if installed_version is not None:
        command.append("--force-reinstall")
    command.append(wheel_path)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            "Unable to install the Hailo Dataflow Compiler {} for {}. {}"
            .format(expected_version, target, detail)
        )

    importlib.invalidate_caches()
    if not _dfc_import_available():
        raise RuntimeError(
            "The Hailo Dataflow Compiler installation completed, but "
            "hailo_sdk_client is still unavailable."
        )
    return True


def _supports_hailo_dfc():
    return (
        sys.platform.startswith("linux")
        and platform.machine().lower() in ("x86_64", "amd64")
    )


def _dfc_import_available():
    try:
        return importlib.util.find_spec("hailo_sdk_client") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _installed_dfc_version():
    if not _dfc_import_available():
        return None
    for package_name in ("hailo-dataflow-compiler", "hailo_dataflow_compiler"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


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
    progress_callback=None,
):
    """Upload a Hailo export and its sidecar files.

    Preserve the complete export directory and expose a flat HEF path for
    Abraia's Hailo runtime and model pairing.
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
    saves_manifest = metadata is not None and callable(
        getattr(client, "save_json", None)
    )
    total_uploads = len(files) + 1 + int(saves_manifest)
    completed_uploads = 0
    if progress_callback:
        progress_callback(95, 100, "Uploading HEF bundle", "Running")

    def report_upload(destination):
        nonlocal completed_uploads
        uploaded.append(destination)
        completed_uploads += 1
        if progress_callback:
            current = 95 + round(5 * completed_uploads / total_uploads)
            progress_callback(
                current, 100,
                f"Uploading Hailo artifacts {completed_uploads}/{total_uploads}",
                "Running",
            )

    for source in files:
        relative = source.relative_to(source_dir).as_posix()
        destination = f"{remote_dir}/{relative}"
        client.upload_file(str(source), destination)
        report_upload(destination)

    # Keep a flat HEF path compatible with paired_hailo_uri(). The complete
    # bundle above remains available for Ultralytics Hailo loading.
    client.upload_file(str(hef_files[0]), remote_hef)
    report_upload(remote_hef)

    manifest_path = f"{remote_dir}/abraia.json"
    if saves_manifest:
        client.save_json(manifest_path, metadata)
        report_upload(manifest_path)
    if progress_callback:
        progress_callback(100, 100, "HEF bundle uploaded", "Complete")

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
    "list_models",
    "list_model_records",
    "next_model_version",
    "HAILO_EXPORT_TARGETS",
    "HAILO_DFC_WHEELS",
    "HAILO_DFC_VERSIONS",
    "ensure_hailo_dfc",
    "save_hailo_bundle",
    "save_versioned_model",
    "versioned_model_paths",
]
