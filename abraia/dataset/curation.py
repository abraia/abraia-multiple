"""Image dataset curation using the SDK's existing image dependencies.

The curation workflow deliberately uses the SDK's existing Pillow and NumPy
image helpers.  It reports conservative recommendations: exact and near
duplicates, unreadable files, and blurry files can be selected for removal;
outliers and exposure issues remain review-only by default.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
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
    source_resolutions = {
        path: int(metrics_by_path[path].get("resolution") or 0)
        for path in grouped_paths
    }

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
                    "method": "pixels",
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
    images: Iterable[Mapping[str, Any]],
    *,
    image_loader,
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
    """Analyze image records through an image loader without local files."""
    if isinstance(images, (str, bytes, os.PathLike)):
        raise TypeError("images must be an iterable of image records")
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
    records = []
    for image in images:
        record = dict(image)
        path = os.fspath(record.get("path") or record.get("name") or "")
        record["path"] = path
        record["name"] = record.get("name") or path
        record.setdefault("file_size", record.get("size"))
        record["cache_key"] = record.get("cache_key") or json.dumps(
            {
                key: record.get(key)
                for key in ("etag", "checksum", "date", "updated_at")
                if record.get(key) is not None
            },
            sort_keys=True,
            default=str,
        )
        records.append(record)
    paths = [record["path"] for record in records]

    quality_findings = analyze_rgb_quality(
        records,
        image_loader,
        similarity_threshold=duplicate_threshold,
        duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
        blur_threshold=blur_threshold,
        blur_percentile=blur_percentile,
        outlier_percentile=outlier_percentile if thorough else None,
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
    exact_groups = defaultdict(list)
    for candidate in quality_findings:
        digest = candidate["metrics"].pop("pixel_hash", None)
        if digest:
            exact_groups[digest].append(candidate["path"])
    return _build_report(
        paths,
        dict(exact_groups),
        quality_findings,
        {os.fspath(path) for path in protected_paths},
        remove_outliers=remove_outliers,
        remove_invalid=remove_invalid,
        work_dir=work_dir,
    )


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
