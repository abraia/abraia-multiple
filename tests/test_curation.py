from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from abraia.training.quality import analyze_rgb_quality
from abraia.training.curation import CurationAnalyzer, curate_dataset


def _write_image(path, value):
    from PIL import Image

    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def test_native_analyzer_reports_exact_duplicates_and_quality_findings(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a_keep.jpg", 120)
    (image_dir / "b_copy.jpg").write_bytes((image_dir / "a_keep.jpg").read_bytes())
    (image_dir / "broken.jpg").write_bytes(b"not an image")

    report = CurationAnalyzer().analyze(image_dir)

    duplicate = next(finding for finding in report.findings if finding.kind == "duplicate")
    assert duplicate.path == "b_copy.jpg"
    assert duplicate.neighbor == "a_keep.jpg"
    assert duplicate.recommended is True
    assert any(finding.kind == "invalid" for finding in report.findings)


def test_native_analyzer_protects_annotated_duplicate(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a_keep.jpg", 120)
    (image_dir / "b_copy.jpg").write_bytes((image_dir / "a_keep.jpg").read_bytes())

    report = CurationAnalyzer().analyze(
        image_dir,
        protected_paths=[image_dir / "b_copy.jpg"],
    )

    duplicate = next(finding for finding in report.findings if finding.kind == "duplicate")
    assert duplicate.path == "a_keep.jpg"
    assert duplicate.neighbor == "b_copy.jpg"
    assert duplicate.recommended is True


def test_native_analyzer_keeps_highest_resolution_similar_image(tmp_path):
    from PIL import Image

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    small[6:26, 8:24] = [180, 90, 40]
    small[12:18, 12:20] = 250
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    Image.fromarray(large).save(image_dir / "a_high.png")
    Image.fromarray(small).save(image_dir / "b_low.png")

    report = CurationAnalyzer().analyze(image_dir)

    duplicate = next(
        finding
        for finding in report.findings
        if finding.kind == "duplicate"
    )
    assert duplicate.path == "b_low.png"
    assert duplicate.neighbor == "a_high.png"
    assert report.recommended_paths == ["b_low.png"]


def test_similarity_analysis_keeps_top_k_neighbor_edges():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = 180
    records = [
        {"name": name, "path": name}
        for name in ("one.jpg", "two.jpg", "three.jpg")
    ]

    findings = analyze_rgb_quality(
        records,
        lambda _path: image,
        nearest_neighbors_k=2,
        include_all=True,
    )

    third = next(finding for finding in findings if finding["path"] == "three.jpg")
    assert len(third["metrics"]["similar_neighbors"]) == 2


def test_curate_dataset_applies_remote_deletions_and_annotation_cleanup():
    class FakeClient:
        def __init__(self):
            self.removals = []

        def download_file(self, source, destination):
            Path(destination).write_bytes(b"image")

        def remove_file(self, path):
            self.removals.append(path)

    client = FakeClient()
    dataset = SimpleNamespace(
        client=client,
        images=[
            {"name": "keep.jpg", "path": "project/keep.jpg"},
            {"name": "copy.jpg", "path": "project/copy.jpg"},
        ],
        annotations=[{"filename": "keep.jpg", "objects": []}],
        save=lambda: setattr(dataset, "saved", True),
    )
    report = SimpleNamespace(
        findings=[],
        recommended_paths=["project/copy.jpg"],
        deleted=[],
    )

    class FakeAnalyzer:
        def analyze(self, *args, **kwargs):
            return report

    with patch("abraia.training.dataset.load_dataset", return_value=dataset):
        result = curate_dataset(
            "project",
            client=client,
            apply=True,
            analyzer=FakeAnalyzer(),
        )

    assert result.deleted == ["project/copy.jpg"]
    assert client.removals == ["project/copy.jpg"]
    assert dataset.saved is True


def test_curate_dataset_uses_relative_paths_for_duplicate_basenames():
    class FakeClient:
        def __init__(self):
            self.removals = []

        def download_file(self, source, destination):
            Path(destination).write_bytes(b"image")

        def remove_file(self, path):
            self.removals.append(path)

    client = FakeClient()
    dataset = SimpleNamespace(
        client=client,
        images=[
            {"name": "duplicate.jpg", "path": "project/set-a/duplicate.jpg"},
            {"name": "duplicate.jpg", "path": "project/set-b/duplicate.jpg"},
        ],
        annotations=[
            {"filename": "set-a/duplicate.jpg", "objects": []},
        ],
        save=lambda: setattr(dataset, "saved", True),
    )
    report = SimpleNamespace(
        findings=[],
        recommended_paths=["project/set-b/duplicate.jpg"],
        deleted=[],
    )

    class FakeAnalyzer:
        seen_kwargs = None

        def analyze(self, *args, **kwargs):
            self.seen_kwargs = kwargs
            return report

    analyzer = FakeAnalyzer()
    with patch("abraia.training.dataset.load_dataset", return_value=dataset):
        curate_dataset(
            "project",
            client=client,
            apply=True,
            analyzer=analyzer,
        )

    assert client.removals == ["project/set-b/duplicate.jpg"]
    assert [annotation["filename"] for annotation in dataset.annotations] == [
        "set-a/duplicate.jpg"
    ]
    assert [path.parts[-2:] for path in analyzer.seen_kwargs["protected_paths"]] == [
        ("set-a", "duplicate.jpg")
    ]
