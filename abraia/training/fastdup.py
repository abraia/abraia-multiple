"""Optional FastDup-backed image dataset curation.

FastDup is deliberately imported lazily.  Dataset analysis is an optional
workflow and should not make the base SDK depend on FastDup or pandas.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
)


PathLike = Union[os.PathLike, str]


class FastdupUnavailable(RuntimeError):
    """Raised when FastDup is requested but is not installed."""


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


class FastdupAnalyzer:
    """Run FastDup and normalize its dataframes into SDK findings.

    ``blur_percentile`` and ``outlier_percentile`` are relative thresholds.
    For example, ``0.05`` considers the lowest five percent of blur scores or
    FastDup outlier results.  Outliers are reported but are not recommended
    for deletion unless ``remove_outliers`` is explicitly enabled.
    """

    def __init__(
        self,
        duplicate_threshold: float = 0.96,
        blur_threshold: Optional[float] = None,
        blur_percentile: Optional[float] = 0.05,
        outlier_percentile: Optional[float] = 0.05,
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
        self.remove_outliers = remove_outliers
        self.remove_invalid = remove_invalid

    def analyze(
        self,
        input_dir: PathLike,
        work_dir: Optional[PathLike] = None,
        protected_paths: Iterable[PathLike] = (),
    ) -> CurationReport:
        """Analyze a local image directory without modifying its contents."""
        fastdup = _load_fastdup()
        input_path = Path(input_dir).resolve()
        if not input_path.is_dir():
            raise ValueError("FastDup input_dir must be an existing directory")

        temporary_work_dir = None
        if work_dir is None:
            temporary_work_dir = tempfile.TemporaryDirectory(prefix="abraia-fastdup-")
            work_path = Path(temporary_work_dir.name)
        else:
            work_path = Path(work_dir)
            work_path.mkdir(parents=True, exist_ok=True)

        try:
            controller = fastdup.create(
                input_dir=str(input_path),
                work_dir=str(work_path),
            )
            result = controller.run(
                ccthreshold=self.duplicate_threshold,
                lower_threshold=self.outlier_percentile or 0,
                run_cc=1,
                run_stats=1,
                overwrite=True,
                print_summary=False,
            )
            if result not in (None, 0):
                raise RuntimeError("FastDup analysis failed with status {}".format(result))

            protected = {
                _local_key(_resolve_local_path(path, input_path))
                for path in protected_paths
            }
            findings = []
            findings.extend(
                self._duplicate_findings(controller, input_path, protected)
            )
            findings.extend(self._outlier_findings(controller, input_path))
            findings.extend(self._blur_findings(controller, input_path, protected))
            findings.extend(self._invalid_findings(controller, input_path, protected))
            scanned = _count_files(input_path)
            return CurationReport(
                scanned=scanned,
                findings=findings,
                work_dir=str(work_path) if work_dir is not None else None,
            )
        finally:
            if temporary_work_dir is not None:
                temporary_work_dir.cleanup()

    def _duplicate_findings(
        self,
        controller: Any,
        input_path: Path,
        protected: Set[str],
    ) -> List[CurationFinding]:
        grouped = controller.connected_components_grouped(
            sort_by="comp_size",
            ascending=False,
        )
        findings = []
        for row in _records(grouped):
            paths = [
                _relative_path(filename, input_path)
                for filename in _as_list(row.get("files"))
            ]
            paths = _unique(paths)
            if len(paths) < 2:
                continue
            protected_group = [
                path for path in paths
                if _local_key(input_path / path) in protected
            ]
            keeper = (protected_group or paths)[0]
            group_id = _string_value(row.get("component_id"))
            score = _number(row.get("distance"))
            for path in paths:
                if path == keeper:
                    continue
                findings.append(CurationFinding(
                    kind="duplicate",
                    path=path,
                    score=score,
                    group_id=group_id,
                    neighbor=keeper,
                    recommended=_local_key(input_path / path) not in protected,
                    details={"keep": keeper},
                ))
        return findings

    def _outlier_findings(
        self,
        controller: Any,
        input_path: Path,
    ) -> List[CurationFinding]:
        if self.outlier_percentile is None:
            return []
        rows = _records(controller.outliers(data=False, fast_mode=True))
        findings = []
        for row in rows:
            filename = _first_value(row, "outlier", "from", "filename")
            if filename is None:
                continue
            findings.append(CurationFinding(
                kind="outlier",
                path=_relative_path(filename, input_path),
                score=_number(_first_value(row, "distance", "outlier_distance", "score")),
                neighbor=_relative_optional_path(
                    _first_value(row, "nearest_neighbor", "to"), input_path
                ),
                recommended=self.remove_outliers,
                details={"review_required": not self.remove_outliers},
            ))
        return findings

    def _blur_findings(
        self,
        controller: Any,
        input_path: Path,
        protected: Set[str],
    ) -> List[CurationFinding]:
        stats = _records(controller.img_stats(data=False))
        values = [
            _number(_first_value(row, "blur", "bluriness"))
            for row in stats
        ]
        values = [value for value in values if value is not None]
        threshold = self.blur_threshold
        if threshold is None and self.blur_percentile is not None:
            threshold = _percentile(values, self.blur_percentile)
        if threshold is None:
            return []

        findings = []
        for row in stats:
            filename = _first_value(row, "filename", "path")
            blur = _number(_first_value(row, "blur", "bluriness"))
            if filename is None or blur is None or blur > threshold:
                continue
            path = _relative_path(filename, input_path)
            findings.append(CurationFinding(
                kind="blurry",
                path=path,
                score=blur,
                recommended=_local_key(input_path / path) not in protected,
                details={"threshold": threshold},
            ))
        return findings

    def _invalid_findings(
        self,
        controller: Any,
        input_path: Path,
        protected: Set[str],
    ) -> List[CurationFinding]:
        if not self.remove_invalid or not hasattr(controller, "invalid_instances"):
            return []
        findings = []
        for row in _records(controller.invalid_instances()):
            filename = _first_value(row, "filename", "path")
            if filename is None:
                continue
            path = _relative_path(filename, input_path)
            findings.append(CurationFinding(
                kind="invalid",
                path=path,
                recommended=_local_key(input_path / path) not in protected,
            ))
        return findings


def curate_dataset(
    project: str,
    client: Any = None,
    work_dir: Optional[PathLike] = None,
    apply: bool = False,
    analyzer: Optional[FastdupAnalyzer] = None,
) -> CurationReport:
    """Analyze a remote Abraia dataset and optionally remove recommendations.

    The remote dataset is staged locally because FastDup expects local files
    for this workflow.  ``apply=False`` is always safe: it downloads, scans,
    and reports without changing remote files.
    """
    from ..training.dataset import load_dataset
    from ..training.core import _resolve_client

    dataset = load_dataset(project, client=_resolve_client(client))
    client = dataset.client
    analyzer = analyzer or FastdupAnalyzer()
    image_records = list(dataset.images or [])
    annotated = {
        _basename(annotation.get("filename"))
        for annotation in dataset.annotations or []
        if isinstance(annotation, Mapping)
    }

    with tempfile.TemporaryDirectory(prefix="abraia-curation-") as staging:
        staging_path = Path(staging)
        local_to_remote = {}
        protected = []
        for image in image_records:
            remote_path = image.get("path") or "{}/{}".format(
                str(project).strip("/"), image.get("name", "")
            )
            relative = _project_relative_path(remote_path, project)
            local_path = staging_path / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(remote_path, str(local_path))
            local_to_remote[_local_key(local_path)] = remote_path
            if _basename(relative) in annotated:
                protected.append(local_path)

        report = analyzer.analyze(
            staging_path,
            work_dir=work_dir,
            protected_paths=protected,
        )
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

        if apply:
            paths = report.recommended_paths
            for path in paths:
                client.remove_file(path)
            if paths:
                deleted_names = {_basename(path) for path in paths}
                dataset.annotations = [
                    annotation
                    for annotation in dataset.annotations or []
                    if (
                        not isinstance(annotation, Mapping)
                        or _basename(annotation.get("filename")) not in deleted_names
                    )
                ]
                dataset.save()
            report.deleted = paths
        return report


def _load_fastdup() -> Any:
    try:
        import fastdup
    except ImportError as error:
        raise FastdupUnavailable(
            "FastDup is required for curation. Install it with "
            "`pip install fastdup` or `pip install abraia[curation]`."
        ) from error
    return fastdup


def _records(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dict()
    if isinstance(value, Mapping):
        return [dict(value)]
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, os.PathLike)):
        return [value]
    return list(value)


def _first_value(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _string_value(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _validate_percentile(name: str, value: Optional[float]) -> Optional[float]:
    if value is not None and not 0 < value <= 1:
        raise ValueError("{} must be greater than 0 and at most 1".format(name))
    return value


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered)) - 1))
    return ordered[index]


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _local_key(path: PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _resolve_local_path(value: PathLike, input_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_path / path


def _relative_path(value: PathLike, input_path: Path) -> str:
    path = _resolve_local_path(value, input_path)
    try:
        return path.resolve().relative_to(input_path.resolve()).as_posix()
    except ValueError:
        return path.name


def _relative_optional_path(value: Any, input_path: Path) -> Optional[str]:
    return None if value is None else _relative_path(value, input_path)


def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").splitlines()[0].strip())


def _project_relative_path(remote_path: str, project: str) -> str:
    path = str(remote_path).strip("/")
    root = str(project).strip("/") + "/"
    return path[len(root):] if path.startswith(root) else os.path.basename(path)


def _count_files(input_path: Path) -> int:
    return sum(1 for path in input_path.rglob("*") if path.is_file())


__all__ = [
    "CurationFinding",
    "CurationReport",
    "FastdupAnalyzer",
    "FastdupUnavailable",
    "curate_dataset",
]
