"""Dependency-free image dataset curation.

The curation workflow deliberately uses the SDK's existing Pillow and NumPy
image helpers.  It reports conservative recommendations: exact and near
duplicates, unreadable files, and blurry files can be selected for removal;
outliers and exposure issues remain review-only by default.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import posixpath
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Union

from PIL import Image

from ..utils import load_image
from .quality import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_NEAREST_NEIGHBORS,
    DEFAULT_SIMILARITY_THRESHOLD,
    analyze_rgb_quality,
    is_rgb_dataset,
)


PathLike = Union[os.PathLike, str]


@dataclass
class CurationFinding:
    """One image-level curation finding."""

    kind: str
    path: str
    score: Optional[float] = None
    group_id: Optional[str] = None
    neighbor: Optional[str] = None
    recommended: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "kind": self.kind,
            "path": self.path,
            "score": self.score,
            "group_id": self.group_id,
            "neighbor": self.neighbor,
            "recommended": self.recommended,
        }
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class CurationReport:
    """Serializable result of a curation scan."""

    scanned: int = 0
    findings: List[CurationFinding] = field(default_factory=list)
    work_dir: Optional[str] = None
    deleted: List[str] = field(default_factory=list)

    @property
    def recommended_paths(self) -> List[str]:
        """Return unique paths selected by the configured curation policy."""
        return _unique(
            finding.path
            for finding in self.findings
            if finding.recommended
        )

    def by_kind(self) -> Dict[str, List[CurationFinding]]:
        grouped = defaultdict(list)
        for finding in self.findings:
            grouped[finding.kind].append(finding)
        return dict(grouped)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned": self.scanned,
            "findings": [finding.to_dict() for finding in self.findings],
            "counts": {
                kind: len(findings)
                for kind, findings in self.by_kind().items()
            },
            "recommended_paths": self.recommended_paths,
            "deleted": list(self.deleted),
            "work_dir": self.work_dir,
        }

    def write_json(self, filename: PathLike) -> None:
        """Write the report in a machine-readable format."""
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)


@dataclass
class CurationScan:
    """Local analysis inputs and results shared by curation policies."""

    input_path: Path
    image_paths: List[Path]
    exact_groups: Dict[str, List[str]]
    quality_findings: List[Dict[str, Any]]
    metrics_by_path: Dict[str, Mapping[str, Any]]
    work_dir: Optional[PathLike] = None


class ImageDatasetScanner:
    """Discover local images and calculate reusable quality measurements."""

    def __init__(
        self,
        duplicate_threshold: float,
        blur_threshold: Optional[float],
        blur_percentile: Optional[float],
        outlier_percentile: Optional[float],
        nearest_neighbors_k: int,
        outlier_mode: str,
        feature_extractor,
        mode: str,
    ) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.blur_threshold = blur_threshold
        self.blur_percentile = blur_percentile
        self.outlier_percentile = outlier_percentile
        self.nearest_neighbors_k = nearest_neighbors_k
        self.outlier_mode = outlier_mode
        self.feature_extractor = feature_extractor
        self.mode = mode

    def scan(
        self,
        input_dir: PathLike,
        work_dir: Optional[PathLike] = None,
        cache_namespace: Optional[str] = None,
        progress_callback=None,
        is_cancelled=None,
    ) -> CurationScan:
        input_path = Path(input_dir).resolve()
        if not input_path.is_dir():
            raise ValueError("input_dir must be an existing directory")

        image_paths = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and _is_image_path(path)
        )
        relative_paths = [
            path.relative_to(input_path).as_posix() for path in image_paths
        ]
        records = [
            {
                "name": relative,
                "path": relative,
                "type": "image/jpeg",
                "file_size": (input_path / relative).stat().st_size,
                "cache_key": _file_cache_key(input_path / relative),
            }
            for relative in relative_paths
        ]
        if self.mode == "thorough":
            exact_groups = _hash_groups(image_paths, input_path)
        else:
            exact_groups = _quick_hash_groups(image_paths, input_path)

        hash_by_path = {
            path: digest
            for digest, paths in exact_groups.items()
            for path in paths
        }
        for record in records:
            record["cache_key"] = hash_by_path.get(
                record["path"], record["cache_key"]
            )

        def loader(relative_path):
            max_size = None if self.mode == "thorough" else 512
            return load_image(input_path / relative_path, max_size=max_size)

        quality_findings = analyze_rgb_quality(
            records,
            loader,
            similarity_threshold=self.duplicate_threshold,
            duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
            blur_threshold=self.blur_threshold,
            blur_percentile=self.blur_percentile,
            outlier_percentile=(
                self.outlier_percentile if self.mode == "thorough" else None
            ),
            nearest_neighbors_k=(
                self.nearest_neighbors_k if self.mode == "thorough" else 1
            ),
            outlier_mode=self.outlier_mode,
            include_all=True,
            feature_extractor=self.feature_extractor,
            cache_dir=work_dir,
            cache_namespace=cache_namespace or str(input_path),
            cache_context={
                "mode": self.mode,
                "preview_size": 512 if self.mode == "fast" else None,
            },
            candidate_mode="all" if self.mode == "thorough" else "hash",
            compute_stats=self.mode == "thorough",
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        return CurationScan(
            input_path=input_path,
            image_paths=image_paths,
            exact_groups=exact_groups,
            quality_findings=quality_findings,
            metrics_by_path={
                candidate["path"]: candidate["metrics"]
                for candidate in quality_findings
            },
            work_dir=work_dir,
        )


class CurationPolicy:
    """Convert a local scan into protected, review, and removal findings."""

    def __init__(self, remove_outliers: bool, remove_invalid: bool) -> None:
        self.remove_outliers = remove_outliers
        self.remove_invalid = remove_invalid

    def build_report(
        self,
        scan: CurationScan,
        protected: Set[str],
    ) -> CurationReport:
        findings: List[CurationFinding] = []
        preserved_paths = set()
        exact_paths = {
            path
            for paths in scan.exact_groups.values()
            if len(paths) > 1
            for path in paths
        }

        for group_id, paths in scan.exact_groups.items():
            if len(paths) < 2:
                continue
            keeper = _choose_keeper(
                paths, protected, scan.input_path, scan.metrics_by_path
            )
            preserved_paths.add(keeper)
            for path in paths:
                if path == keeper:
                    continue
                findings.append(CurationFinding(
                    kind="duplicate",
                    path=path,
                    score=1.0,
                    group_id=group_id,
                    neighbor=keeper,
                    recommended=_is_removable(
                        path, protected, scan.input_path
                    ),
                    details={
                        "method": "sha256",
                        "keep": keeper,
                        "keep_resolution": _resolution(
                            scan.metrics_by_path.get(keeper)
                        ),
                    },
                ))

        similar_edges = _similar_edges(scan.quality_findings, exact_paths)
        for group_index, members in enumerate(
            _connected_components(similar_edges),
            start=1,
        ):
            keeper = _choose_keeper(
                members, protected, scan.input_path, scan.metrics_by_path
            )
            preserved_paths.add(keeper)
            group_id = "perceptual-{}".format(group_index)
            for path in members:
                if path == keeper:
                    continue
                related = [
                    edge for edge in similar_edges if path in edge[:2]
                ]
                best_edge = max(
                    related,
                    key=lambda edge: edge[2].get("similarity") or 0.0,
                )
                findings.append(CurationFinding(
                    kind=(
                        "duplicate"
                        if any(
                            edge[2].get("reason") == "duplicate"
                            for edge in related
                        )
                        else "similar"
                    ),
                    path=path,
                    score=best_edge[2].get("similarity"),
                    group_id=group_id,
                    neighbor=keeper,
                    recommended=_is_removable(
                        path, protected, scan.input_path
                    ),
                    details={
                        "method": "perceptual",
                        "keep": keeper,
                        "keep_resolution": _resolution(
                            scan.metrics_by_path.get(keeper)
                        ),
                    },
                ))

        for candidate in scan.quality_findings:
            path = candidate["path"]
            reasons = set(candidate["reasons"])
            if "unreadable" in reasons:
                findings.append(CurationFinding(
                    kind="invalid",
                    path=path,
                    recommended=(
                        self.remove_invalid
                        and _is_removable(path, protected, scan.input_path)
                    ),
                    details=candidate["metrics"],
                ))

            for reason, kind in (
                ("blurry", "blurry"),
                ("dark", "dark"),
                ("bright", "bright"),
                ("outlier", "outlier"),
            ):
                if reason not in reasons:
                    continue
                findings.append(CurationFinding(
                    kind=kind,
                    path=path,
                    score=_metric_for_reason(candidate["metrics"], reason),
                    recommended=(
                        reason == "blurry"
                        or (
                            reason == "outlier" and self.remove_outliers
                        )
                    )
                    and path not in preserved_paths
                    and _is_removable(path, protected, scan.input_path),
                    details=candidate["metrics"],
                ))

        return CurationReport(
            scanned=len(scan.image_paths),
            findings=findings,
            work_dir=(
                str(scan.work_dir) if scan.work_dir is not None else None
            ),
        )


class CurationAnalyzer:
    """Analyze a local image directory without modifying its contents.

    The analyzer combines byte hashes for exact duplicates with perceptual
    similarity and image-quality checks from :mod:`abraia.training.quality`.
    Protected paths are never recommended for deletion.  When a protected
    image is part of a duplicate pair, the unprotected image is selected
    instead. ``mode='fast'`` uses bounded previews, sampled duplicate
    fingerprints, and hash/coarse-feature candidate buckets. It skips the
    full-image quality statistics and NN outlier pass. ``mode='thorough'``
    performs exhaustive comparisons and the richer analysis.
    """

    def __init__(
        self,
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
    ) -> None:
        if not 0 <= duplicate_threshold <= 1:
            raise ValueError("duplicate_threshold must be between 0 and 1")
        self.duplicate_threshold = duplicate_threshold
        self.blur_threshold = blur_threshold
        self.blur_percentile = _validate_percentile(
            "blur_percentile", blur_percentile
        )
        self.outlier_percentile = _validate_percentile(
            "outlier_percentile", outlier_percentile
        )
        if nearest_neighbors_k < 1:
            raise ValueError("nearest_neighbors_k must be at least 1")
        if outlier_mode not in {"one", "all"}:
            raise ValueError("outlier_mode must be 'one' or 'all'")
        if mode not in {"fast", "thorough"}:
            raise ValueError("mode must be 'fast' or 'thorough'")
        self.nearest_neighbors_k = nearest_neighbors_k
        self.outlier_mode = outlier_mode
        self.feature_extractor = feature_extractor
        self.mode = mode
        self.remove_outliers = remove_outliers
        self.remove_invalid = remove_invalid

    def analyze(
        self,
        input_dir: PathLike,
        work_dir: Optional[PathLike] = None,
        protected_paths: Iterable[PathLike] = (),
        cache_namespace: Optional[str] = None,
        progress_callback=None,
        is_cancelled=None,
    ) -> CurationReport:
        """Analyze a local image directory without changing its contents."""
        scan = ImageDatasetScanner(
            duplicate_threshold=self.duplicate_threshold,
            blur_threshold=self.blur_threshold,
            blur_percentile=self.blur_percentile,
            outlier_percentile=self.outlier_percentile,
            nearest_neighbors_k=self.nearest_neighbors_k,
            outlier_mode=self.outlier_mode,
            feature_extractor=self.feature_extractor,
            mode=self.mode,
        ).scan(
            input_dir,
            work_dir=work_dir,
            cache_namespace=cache_namespace,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        protected = {
            _local_key(_resolve_local_path(path, scan.input_path))
            for path in protected_paths
        }
        return CurationPolicy(
            remove_outliers=self.remove_outliers,
            remove_invalid=self.remove_invalid,
        ).build_report(
            scan,
            protected,
        )


class RemoteCurationService:
    """Stage, analyze, and optionally mutate a remote image dataset."""

    def run(
        self,
        project: str,
        client: Any = None,
        work_dir: Optional[PathLike] = None,
        apply: bool = False,
        analyzer: Optional[CurationAnalyzer] = None,
        confirm=None,
        progress_callback=None,
        is_cancelled=None,
    ) -> CurationReport:
        from ..training.dataset import load_dataset
        from ..training.core import _resolve_client

        dataset = load_dataset(project, client=_resolve_client(client))
        client = dataset.client
        analyzer = analyzer or CurationAnalyzer()
        image_records = list(dataset.images or [])
        annotated = {
            _annotation_relative_path(annotation, project)
            for annotation in dataset.annotations or []
            if isinstance(annotation, Mapping)
        }
        image_relatives = [
            _project_relative_path(
                image.get("path") or "{}/{}".format(
                    str(project).strip("/"), image.get("name", "")
                ),
                project,
            )
            for image in image_records
        ]
        basename_counts = defaultdict(int)
        for relative in image_relatives:
            basename_counts[_basename(relative)] += 1

        with tempfile.TemporaryDirectory(prefix="abraia-curation-") as staging:
            staging_path = Path(staging)
            local_to_remote, protected = self._stage_images(
                client,
                project,
                image_records,
                image_relatives,
                annotated,
                basename_counts,
                staging_path,
                progress_callback=self._phase_progress(
                    progress_callback, 0, 50
                ),
                is_cancelled=is_cancelled,
            )
            report = analyzer.analyze(
                staging_path,
                **self._analyzer_kwargs(
                    analyzer,
                    client,
                    project,
                    work_dir,
                    protected,
                    progress_callback=self._phase_progress(
                        progress_callback, 50, 50
                    ),
                    is_cancelled=is_cancelled,
                ),
            )
            report = self._to_remote_report(
                report, staging_path, local_to_remote
            )
            if apply:
                paths = report.recommended_paths
                if paths and confirm is not None and not confirm(paths):
                    return report
                self._apply_report(
                    report, client, dataset, project, basename_counts
                )
            if progress_callback:
                progress_callback(100, 100, "Curation complete", True)
            return report

    @staticmethod
    def _phase_progress(callback, offset, span):
        """Scale one curation phase into the overall 0–100 progress range."""
        if callback is None:
            return None

        def report(current, total, detail, completed):
            total = max(1, int(total or 1))
            current = max(0, min(int(current), total))
            progress = offset + round(span * current / total)
            callback(progress, 100, detail, completed)

        return report

    @staticmethod
    def _stage_images(
        client,
        project,
        image_records,
        image_relatives,
        annotated,
        basename_counts,
        staging_path,
        progress_callback=None,
        is_cancelled=None,
    ):
        local_to_remote = {}
        protected = []
        total = max(1, len(image_records))
        for index, (image, relative) in enumerate(
            zip(image_records, image_relatives), start=1
        ):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Operation canceled")
            remote_path = image.get("path") or "{}/{}".format(
                str(project).strip("/"), image.get("name", "")
            )
            if progress_callback:
                progress_callback(index - 1, total, remote_path, False)
            local_path = staging_path / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(remote_path, str(local_path))
            local_to_remote[_local_key(local_path)] = remote_path
            if _is_annotated_path(relative, annotated, basename_counts):
                protected.append(local_path)
            if progress_callback:
                progress_callback(index, total, remote_path, True)
        return local_to_remote, protected

    @staticmethod
    def _analyzer_kwargs(
        analyzer,
        client,
        project,
        work_dir,
        protected,
        progress_callback=None,
        is_cancelled=None,
    ):
        kwargs = {
            "work_dir": work_dir,
            "protected_paths": protected,
        }
        if isinstance(analyzer, CurationAnalyzer):
            owner = str(getattr(client, "userid", "") or "").strip("/")
            project_name = str(project).strip("/")
            kwargs["cache_namespace"] = f"{owner}/{project_name}"
            kwargs["progress_callback"] = progress_callback
            kwargs["is_cancelled"] = is_cancelled
        return kwargs

    @staticmethod
    def _to_remote_report(report, staging_path, local_to_remote):
        remote_findings = []
        for finding in report.findings:
            local_path = staging_path / finding.path
            remote_path = local_to_remote.get(_local_key(local_path))
            if remote_path is None:
                continue
            finding.path = remote_path
            if finding.neighbor:
                neighbor = local_to_remote.get(
                    _local_key(staging_path / finding.neighbor)
                )
                finding.neighbor = neighbor or finding.neighbor
            if finding.details.get("keep"):
                keep = local_to_remote.get(
                    _local_key(staging_path / finding.details["keep"])
                )
                finding.details["keep"] = keep or finding.details["keep"]
            remote_findings.append(finding)
        report.findings = remote_findings
        return report

    @staticmethod
    def _apply_report(
        report,
        client,
        dataset,
        project,
        basename_counts,
    ):
        paths = report.recommended_paths
        for path in paths:
            client.remove_file(path)
        if paths:
            deleted_relatives = {
                _project_relative_path(path, project) for path in paths
            }
            dataset.annotations = [
                annotation
                for annotation in dataset.annotations or []
                if (
                    not isinstance(annotation, Mapping)
                    or not _is_deleted_annotation(
                        annotation,
                        deleted_relatives,
                        basename_counts,
                        project,
                    )
                )
            ]
            dataset.save()
        report.deleted = paths


def curate_dataset(
    project: str,
    client: Any = None,
    work_dir: Optional[PathLike] = None,
    apply: bool = False,
    analyzer: Optional[CurationAnalyzer] = None,
    confirm=None,
    progress_callback=None,
    is_cancelled=None,
) -> CurationReport:
    """Analyze a remote Abraia dataset and optionally remove recommendations.

    ``apply`` remains opt-in. Callers such as Studio can use the report to
    present and review recommendations, then perform a separate user-directed
    delete operation without allowing curation itself to mutate the dataset.
    """
    return RemoteCurationService().run(
        project,
        client=client,
        work_dir=work_dir,
        apply=apply,
        analyzer=analyzer,
        confirm=confirm,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp",
        ".tif", ".tiff", ".heic", ".heif",
    }


def _image_dimensions(path: Path):
    try:
        with Image.open(path) as opened:
            return int(opened.width), int(opened.height)
    except Exception:
        return 0, 0


def _hash_groups(paths: Iterable[Path], input_path: Path) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(path.relative_to(input_path).as_posix())
    return dict(groups)


def _quick_hash_groups(
    paths: Iterable[Path], input_path: Path, sample_size: int = 65536
) -> Dict[str, List[str]]:
    """Group files by a cheap size plus head/tail fingerprint.

    The fingerprint is only used to find possible exact duplicates.  Candidate
    groups are verified with a full SHA-256 read, so sampled-byte collisions
    cannot produce a false duplicate recommendation.
    """
    candidates = defaultdict(list)
    for path in paths:
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
            candidates[digest].append(path)
        except OSError:
            continue

    groups = {}
    for fingerprint, candidate_paths in candidates.items():
        if len(candidate_paths) < 2:
            continue
        verified = defaultdict(list)
        for path in candidate_paths:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            verified[digest].append(path.relative_to(input_path).as_posix())
        for digest, relative_paths in verified.items():
            if len(relative_paths) > 1:
                groups[digest] = relative_paths
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
    input_path: Path,
    metrics_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    """Choose the best representative, favoring protection and quality."""
    paths = list(paths)
    for path in paths:
        metrics = metrics_by_path.get(path)
        if metrics is None or "source_resolution" in metrics:
            continue
        width, height = _image_dimensions(input_path / path)
        metrics["source_resolution"] = int(width * height)
        if width and height:
            metrics["width"] = width
            metrics["height"] = height
            metrics["resolution"] = int(width * height)
    return sorted(
        paths,
        key=lambda path: (
            -int(_local_key(input_path / path) in protected),
            -_resolution(metrics_by_path.get(path)),
            -_number(metrics_by_path.get(path, {}).get("quality_score")),
            -_number(metrics_by_path.get(path, {}).get("sharpness")),
            path,
        ),
    )[0]


def _similar_edges(quality_findings, exact_paths):
    edges = []
    for candidate in quality_findings:
        path = candidate["path"]
        reasons = set(candidate["reasons"])
        neighbors = candidate["metrics"].get("similar_neighbors")
        if not neighbors:
            neighbor = candidate["metrics"].get("similar_to")
            neighbors = [{
                "name": neighbor,
                "reason": "duplicate" if "duplicate" in reasons else "highly similar",
                "similarity": candidate["metrics"].get("similarity"),
            }]
        for neighbor in neighbors:
            neighbor_path = neighbor.get("name")
            if (
                not neighbor_path
                or path in exact_paths
                or neighbor_path in exact_paths
                or not ({"duplicate", "highly similar"} & reasons)
            ):
                continue
            edges.append((path, neighbor_path, {
                "reason": neighbor.get("reason") or (
                    "duplicate" if "duplicate" in reasons else "highly similar"
                ),
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
    return [sorted(values) for values in sorted(groups.values(), key=lambda value: value[0])]


def _resolution(metrics):
    return int((metrics or {}).get("resolution") or 0)


def _number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_removable(path: Optional[str], protected: Set[str], input_path: Path) -> bool:
    return bool(path) and _local_key(input_path / path) not in protected


def _metric_for_reason(metrics: Mapping[str, Any], reason: str) -> Optional[float]:
    keys = {
        "blurry": "sharpness",
        "dark": "brightness",
        "bright": "brightness",
        "outlier": "outlier_score",
    }
    value = metrics.get(keys[reason])
    return None if value is None else float(value)


def _validate_percentile(name: str, value: Optional[float]) -> Optional[float]:
    if value is not None and not 0 < value <= 1:
        raise ValueError("{} must be greater than 0 and at most 1".format(name))
    return value


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _local_key(path: PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _resolve_local_path(value: PathLike, input_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_path / path


def _basename(value: Any) -> str:
    return posixpath.basename(str(value or "").splitlines()[0].strip().replace("\\", "/"))


def _annotation_relative_path(annotation: Mapping[str, Any], project: str) -> str:
    """Normalize an annotation's path relative to its project."""
    value = annotation.get("path") or annotation.get("filename")
    return _project_relative_path(value, project) if value else ""


def _is_annotated_path(
    relative: str,
    annotated: Set[str],
    basename_counts: Mapping[str, int],
) -> bool:
    """Match exact paths and conservatively support legacy basename metadata."""
    if relative in annotated:
        return True
    basename = _basename(relative)
    return basename_counts.get(basename, 0) > 0 and basename in annotated


def _is_deleted_annotation(
    annotation: Mapping[str, Any],
    deleted_relatives: Set[str],
    basename_counts: Mapping[str, int],
    project: str,
) -> bool:
    """Match annotation cleanup to the same path identity used for protection."""
    relative = _annotation_relative_path(annotation, project)
    if relative in deleted_relatives:
        return True
    basename = _basename(relative)
    deleted_basenames = {_basename(path) for path in deleted_relatives}
    return (
        "/" not in relative
        and basename_counts.get(basename, 0) == 1
        and relative in deleted_basenames
    )


def _project_relative_path(remote_path: str, project: str) -> str:
    path = posixpath.normpath(str(remote_path or "").strip().replace("\\", "/").strip("/"))
    root = str(project).strip("/") + "/"
    return path[len(root):] if path.startswith(root) else path


__all__ = [
    "CurationAnalyzer",
    "CurationFinding",
    "CurationReport",
    "curate_dataset",
    "is_rgb_dataset",
]
