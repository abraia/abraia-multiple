from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from abraia.training.fastdup import FastdupAnalyzer, curate_dataset


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient=None):
        assert orient == "records"
        return self.rows


class FakeController:
    def __init__(self):
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return 0

    def connected_components_grouped(self, **kwargs):
        return FakeFrame([{
            "component_id": 3,
            "files": ["/tmp/images/keep.jpg", "/tmp/images/copy.jpg"],
            "distance": 0.99,
        }])

    def outliers(self, **kwargs):
        return FakeFrame([{
            "outlier": "/tmp/images/rare.jpg",
            "nearest_neighbor": "/tmp/images/keep.jpg",
            "distance": 0.2,
        }])

    def img_stats(self, **kwargs):
        return FakeFrame([
            {"filename": "/tmp/images/keep.jpg", "blur": 100.0},
            {"filename": "/tmp/images/copy.jpg", "blur": 2.0},
        ])

    def invalid_instances(self):
        return FakeFrame([])


class FakeFastdup:
    def __init__(self):
        self.controller = FakeController()

    def create(self, **kwargs):
        return self.controller


def test_fastdup_analyzer_normalizes_duplicate_outlier_and_blur_results(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for filename in ("keep.jpg", "copy.jpg", "rare.jpg"):
        (image_dir / filename).write_bytes(b"image")
    fake = FakeFastdup()

    with patch(
        "abraia.training.fastdup._load_fastdup",
        return_value=fake,
    ):
        report = FastdupAnalyzer().analyze(image_dir)

    assert report.scanned == 3
    assert [finding.kind for finding in report.findings] == [
        "duplicate",
        "outlier",
        "blurry",
    ]
    assert report.recommended_paths == ["copy.jpg"]
    assert fake.controller.run_kwargs["ccthreshold"] == 0.96


def test_fastdup_analyzer_protects_annotated_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for filename in ("keep.jpg", "copy.jpg", "rare.jpg"):
        (image_dir / filename).write_bytes(b"image")
    fake = FakeFastdup()

    with patch(
        "abraia.training.fastdup._load_fastdup",
        return_value=fake,
    ):
        report = FastdupAnalyzer().analyze(
            image_dir,
            protected_paths=[image_dir / "copy.jpg"],
        )

    duplicate = next(
        finding for finding in report.findings if finding.kind == "duplicate"
    )
    assert duplicate.path == "keep.jpg"
    assert duplicate.recommended is True


def test_curate_dataset_applies_remote_deletions_and_annotation_cleanup(tmp_path):
    class FakeClient:
        userid = "user"

        def __init__(self):
            self.downloads = []
            self.removals = []
            self.saved = None

        def download_file(self, source, destination):
            self.downloads.append((source, destination))
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

    with patch(
        "abraia.training.dataset.load_dataset",
        return_value=dataset,
    ):
        result = curate_dataset(
            "project",
            client=client,
            apply=True,
            analyzer=FakeAnalyzer(),
        )

    assert result.deleted == ["project/copy.jpg"]
    assert client.removals == ["project/copy.jpg"]
    assert dataset.saved is True
