"""Remote dataset staging and application for image curation."""

from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional

from .curation import (
    CurationAnalyzer,
    CurationReport,
    PathLike,
    _annotation_relative_path,
    _basename,
    _is_annotated_path,
    _is_deleted_annotation,
    _local_key,
    _project_relative_path,
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
