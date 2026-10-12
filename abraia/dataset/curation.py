"""Image dataset curation using the SDK's existing image dependencies.

The curation workflow deliberately uses the SDK's existing Pillow and NumPy
image helpers.  It reports conservative recommendations: exact and near
duplicates, unreadable files, and blurry files can be selected for removal;
outliers and exposure issues remain review-only by default.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Union,
)

from PIL import Image

from ..utils import load_image
from .quality import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    analyze_rgb_quality,
)


PathLike = Union[os.PathLike, str]


def _finding(
    kind: str,
    path: str,
    *,
    score: Optional[float] = None,
    group_id: Optional[str] = None,
    neighbor: Optional[str] = None,
    recommended: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    finding = {
        "kind": kind,
        "path": path,
        "score": score,
        "group_id": group_id,
        "neighbor": neighbor,
        "recommended": recommended,
    }
    if details:
        finding["details"] = details
    return finding


def _build_report(
    paths: List[str],
    local_paths: Mapping[str, Path],
    exact_groups: Mapping[str, List[str]],
    quality_findings: List[Dict[str, Any]],
    protected: Set[str],
    *,
    remove_outliers: bool,
    remove_invalid: bool,
    work_dir: Optional[PathLike],
) -> Dict[str, Any]:
    """Convert an image-path scan into review and removal findings."""
    findings: List[Dict[str, Any]] = []
    preserved_paths = set()
    exact_paths = {
        path
        for group in exact_groups.values()
        if len(group) > 1
        for path in group
    }
    similar_edges = _similar_edges(quality_findings, exact_paths)
    metrics_by_path = {
        candidate["path"]: candidate["metrics"]
        for candidate in quality_findings
    }
    grouped_paths = set(exact_paths)
    grouped_paths.update(
        path
        for first, second, _details in similar_edges
        for path in (first, second)
    )
    source_resolutions = {}
    for path in grouped_paths:
        width, height = _image_dimensions(local_paths[path])
        source_resolutions[path] = width * height

    for group_id, group in exact_groups.items():
        if len(group) < 2:
            continue
        keeper = _choose_keeper(
            group, protected, metrics_by_path, source_resolutions
        )
        preserved_paths.add(keeper)
        for path in group:
            if path == keeper:
                continue
            findings.append(_finding(
                "duplicate",
                path,
                score=1.0,
                group_id=group_id,
                neighbor=keeper,
                recommended=_is_removable(path, protected),
                details={
                    "method": "sha256",
                    "keep": keeper,
                    "keep_resolution": source_resolutions[keeper],
                },
            ))

    for group_index, members in enumerate(
        _connected_components(similar_edges),
        start=1,
    ):
        keeper = _choose_keeper(
            members, protected, metrics_by_path, source_resolutions
        )
        preserved_paths.add(keeper)
        group_id = "perceptual-{}".format(group_index)
        for path in members:
            if path == keeper:
                continue
            related = [edge for edge in similar_edges if path in edge[:2]]
            best_edge = max(
                related,
                key=lambda edge: edge[2].get("similarity") or 0.0,
            )
            neighbor = (
                best_edge[1] if best_edge[0] == path else best_edge[0]
            )
            findings.append(_finding(
                "duplicate"
                if best_edge[2].get("reason") == "duplicate"
                else "similar",
                path,
                score=best_edge[2].get("similarity"),
                group_id=group_id,
                neighbor=neighbor,
                recommended=_is_removable(path, protected),
                details={
                    "method": "perceptual",
                    "keep": keeper,
                    "keep_resolution": source_resolutions[keeper],
                },
            ))

    for candidate in quality_findings:
        path = candidate["path"]
        reasons = set(candidate["reasons"])
        if "unreadable" in reasons:
            findings.append(_finding(
                "invalid",
                path,
                recommended=(
                    remove_invalid
                    and _is_removable(path, protected)
                ),
                details=candidate["metrics"],
            ))

        for reason, kind, metric in (
            ("blurry", "blurry", "sharpness"),
            ("dark", "dark", "brightness"),
            ("bright", "bright", "brightness"),
            ("outlier", "outlier", "outlier_score"),
        ):
            if reason not in reasons:
                continue
            score = candidate["metrics"].get(metric)
            findings.append(_finding(
                kind,
                path,
                score=None if score is None else float(score),
                recommended=(
                    reason == "blurry"
                    or (reason == "outlier" and remove_outliers)
                )
                and path not in preserved_paths
                and _is_removable(path, protected),
                details=candidate["metrics"],
            ))

    counts = Counter(finding["kind"] for finding in findings)
    recommended_paths = list(dict.fromkeys(
        finding["path"]
        for finding in findings
        if finding["recommended"] and finding["path"]
    ))
    return {
        "scanned": len(paths),
        "findings": findings,
        "counts": dict(counts),
        "recommended_paths": recommended_paths,
        "deleted": [],
        "work_dir": str(work_dir) if work_dir is not None else None,
    }


def analyze_curation(
    paths: Iterable[PathLike],
    *,
    path_loader=None,
    work_dir: Optional[PathLike] = None,
    protected_paths: Iterable[PathLike] = (),
    cache_namespace: Optional[str] = None,
    progress_callback=None,
    is_cancelled=None,
    duplicate_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    blur_threshold: Optional[float] = None,
    blur_percentile: Optional[float] = 0.05,
    outlier_percentile: Optional[float] = 0.05,
    nearest_neighbors_k: int = 2,
    outlier_mode: str = "one",
    feature_extractor=None,
    mode: str = "fast",
    remove_outliers: bool = False,
    remove_invalid: bool = True,
) -> Dict[str, Any]:
    """Analyze image paths and return findings without changing the dataset.

    ``mode='fast'`` uses bounded previews and hash-based candidate buckets.
    ``mode='thorough'`` performs exhaustive comparisons and richer analysis.
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError("paths must be an iterable of image paths")
    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be between 0 and 1")
    for name, percentile in (
        ("blur_percentile", blur_percentile),
        ("outlier_percentile", outlier_percentile),
    ):
        if percentile is not None and not 0 < percentile <= 1:
            raise ValueError(
                "{} must be greater than 0 and at most 1".format(name)
            )
    if nearest_neighbors_k < 1:
        raise ValueError("nearest_neighbors_k must be at least 1")
    if outlier_mode not in {"one", "all"}:
        raise ValueError("outlier_mode must be 'one' or 'all'")
    if mode not in {"fast", "thorough"}:
        raise ValueError("mode must be 'fast' or 'thorough'")

    thorough = mode == "thorough"
    is_cancelled = is_cancelled or (lambda: False)
    paths = [os.fspath(path) for path in paths]
    path_loader = path_loader or (lambda path: path)
    local_paths = {
        path: Path(path_loader(path)).resolve()
        for path in paths
    }
    records = []
    for path in paths:
        local_path = local_paths[path]
        try:
            file_size = local_path.stat().st_size
        except OSError:
            file_size = 0
        records.append({
            "name": path,
            "path": path,
            "file_size": file_size,
            "cache_key": _file_cache_key(local_path),
        })

    exact_groups = (
        _hash_groups(paths, local_paths, is_cancelled)
        if thorough
        else _quick_hash_groups(paths, local_paths, is_cancelled=is_cancelled)
    )
    hash_by_path = {
        path: digest
        for digest, group in exact_groups.items()
        for path in group
    }
    for record in records:
        record["cache_key"] = hash_by_path.get(
            record["path"], record["cache_key"]
        )

    def load_local_image(path):
        max_size = None if thorough else 512
        return load_image(local_paths[path], max_size=max_size)

    quality_findings = analyze_rgb_quality(
        records,
        load_local_image,
        similarity_threshold=duplicate_threshold,
        duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
        blur_threshold=blur_threshold,
        blur_percentile=blur_percentile,
        outlier_percentile=(
            outlier_percentile if thorough else None
        ),
        nearest_neighbors_k=nearest_neighbors_k if thorough else 1,
        outlier_mode=outlier_mode,
        include_all=True,
        feature_extractor=feature_extractor,
        cache_dir=work_dir,
        cache_namespace=cache_namespace or "dataset",
        cache_context={
            "mode": mode,
            "preview_size": None if thorough else 512,
        },
        candidate_mode="all" if thorough else "hash",
        compute_stats=thorough,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )
    protected = {os.fspath(path) for path in protected_paths}
    return _build_report(
        paths,
        local_paths,
        exact_groups,
        quality_findings,
        protected,
        remove_outliers=remove_outliers,
        remove_invalid=remove_invalid,
        work_dir=work_dir,
    )


def _image_dimensions(path: Path):
    try:
        with Image.open(path) as opened:
            return int(opened.width), int(opened.height)
    except Exception:
        return 0, 0


def _hash_file(path: Path, is_cancelled) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if is_cancelled():
                raise RuntimeError("Operation canceled")
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_groups(
    paths: Iterable[str],
    local_paths: Mapping[str, Path],
    is_cancelled,
) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for path in paths:
        if is_cancelled():
            raise RuntimeError("Operation canceled")
        try:
            digest = _hash_file(local_paths[path], is_cancelled)
        except OSError:
            continue
        groups[digest].append(path)
    return dict(groups)


def _quick_hash_groups(
    paths: Iterable[str],
    local_paths: Mapping[str, Path],
    sample_size: int = 65536,
    is_cancelled=None,
) -> Dict[str, List[str]]:
    """Group files by a cheap size plus head/tail fingerprint.

    The fingerprint is only used to find possible exact duplicates.  Candidate
    groups are verified with a full SHA-256 read, so sampled-byte collisions
    cannot produce a false duplicate recommendation.
    """
    candidates = defaultdict(list)
    is_cancelled = is_cancelled or (lambda: False)
    for source_path in paths:
        if is_cancelled():
            raise RuntimeError("Operation canceled")
        path = local_paths[source_path]
        try:
            size = path.stat().st_size
            with path.open("rb") as stream:
                head = stream.read(sample_size)
                if size > sample_size:
                    stream.seek(max(0, size - sample_size))
                    tail = stream.read(sample_size)
                else:
                    tail = b""
            digest = hashlib.sha256(
                str(size).encode("ascii") + head + tail
            ).hexdigest()
            candidates[digest].append(source_path)
        except OSError:
            continue

    groups = {}
    for fingerprint, candidate_paths in candidates.items():
        if len(candidate_paths) < 2:
            continue
        verified = defaultdict(list)
        for source_path in candidate_paths:
            if is_cancelled():
                raise RuntimeError("Operation canceled")
            try:
                digest = _hash_file(local_paths[source_path], is_cancelled)
            except OSError:
                continue
            verified[digest].append(source_path)
        for digest, source_paths in verified.items():
            if len(source_paths) > 1:
                groups[digest] = source_paths
    return groups


def _file_cache_key(path: Path) -> str:
    """Return a cheap cache identity without reading the complete file."""
    try:
        stat = path.stat()
        return "{}:{}".format(stat.st_size, stat.st_mtime_ns)
    except OSError:
        return "missing"


def _choose_keeper(
    paths: Iterable[str],
    protected: Set[str],
    metrics_by_path: Mapping[str, Mapping[str, Any]],
    source_resolutions: Mapping[str, int],
) -> str:
    """Choose the best representative, favoring protection and quality."""
    return min(
        paths,
        key=lambda path: (
            -int(path in protected),
            -source_resolutions[path],
            -_number(metrics_by_path.get(path, {}).get("quality_score")),
            -_number(metrics_by_path.get(path, {}).get("sharpness")),
            path,
        ),
    )


def _similar_edges(quality_findings, exact_paths):
    edges = []
    for candidate in quality_findings:
        path = candidate["path"]
        reasons = set(candidate["reasons"])
        if not ({"duplicate", "highly similar"} & reasons):
            continue
        for neighbor in candidate["metrics"].get("similar_neighbors", ()):
            neighbor_path = neighbor.get("name")
            if (
                not neighbor_path
                or path in exact_paths
                or neighbor_path in exact_paths
            ):
                continue
            edges.append((path, neighbor_path, {
                "reason": neighbor["reason"],
                "similarity": neighbor.get("similarity"),
            }))
    return edges


def _connected_components(edges):
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second, _details in edges:
        union(first, second)
    groups = defaultdict(list)
    for value in parent:
        groups[find(value)].append(value)
    ordered_groups = sorted(groups.values(), key=lambda values: min(values))
    return [sorted(values) for values in ordered_groups]


def _number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_removable(path: Optional[str], protected: Set[str]) -> bool:
    return bool(path) and path not in protected


__all__ = [
    "analyze_curation",
]
