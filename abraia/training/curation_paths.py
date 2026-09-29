"""Remote project and annotation path normalization for dataset curation."""

import posixpath
from typing import Any, Mapping, Set


def basename(value: Any) -> str:
    """Return a normalized path's final component."""
    return posixpath.basename(
        str(value or "").splitlines()[0].strip().replace("\\", "/")
    )


def annotation_relative_path(annotation: Mapping[str, Any], project: str) -> str:
    """Normalize an annotation's path relative to its project."""
    value = annotation.get("path") or annotation.get("filename")
    return project_relative_path(value, project) if value else ""


def is_annotated_path(
    relative: str,
    annotated: Set[str],
    basename_counts,
) -> bool:
    """Match exact paths and conservatively support legacy basename metadata."""
    if relative in annotated:
        return True
    name = basename(relative)
    return basename_counts.get(name, 0) > 0 and name in annotated


def is_deleted_annotation(
    annotation: Mapping[str, Any],
    deleted_relatives: Set[str],
    basename_counts,
    project: str,
) -> bool:
    """Match annotation cleanup to the same path identity used for protection."""
    relative = annotation_relative_path(annotation, project)
    if relative in deleted_relatives:
        return True
    name = basename(relative)
    deleted_basenames = {basename(path) for path in deleted_relatives}
    return (
        "/" not in relative
        and basename_counts.get(name, 0) == 1
        and relative in deleted_basenames
    )


def project_relative_path(remote_path: str, project: str) -> str:
    """Remove a project prefix while normalizing remote path separators."""
    path = posixpath.normpath(
        str(remote_path or "").strip().replace("\\", "/").strip("/")
    )
    root = str(project).strip("/") + "/"
    return path[len(root):] if path.startswith(root) else path


__all__ = [
    "annotation_relative_path",
    "basename",
    "is_annotated_path",
    "is_deleted_annotation",
    "project_relative_path",
]
