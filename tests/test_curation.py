import numpy as np

from abraia.dataset.curation import analyze_curation
from abraia.dataset.quality import analyze_rgb_quality
from abraia.dataset.remote import Dataset


def _write_image(path, value):
    from PIL import Image

    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def test_native_analyzer_reports_exact_duplicates_and_quality_findings(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a_keep.jpg", 120)
    (image_dir / "b_copy.jpg").write_bytes((image_dir / "a_keep.jpg").read_bytes())
    (image_dir / "broken.jpg").write_bytes(b"not an image")

    report = analyze_curation(
        [path.name for path in image_dir.iterdir()],
        path_loader=lambda path: image_dir / path,
    )

    duplicate = next(
        finding
        for finding in report["findings"]
        if finding["kind"] == "duplicate"
    )
    assert duplicate["path"] == "b_copy.jpg"
    assert duplicate["neighbor"] == "a_keep.jpg"
    assert duplicate["recommended"] is True
    assert any(
        finding["kind"] == "invalid" for finding in report["findings"]
    )


def test_native_analyzer_protects_annotated_duplicate(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a_keep.jpg", 120)
    (image_dir / "b_copy.jpg").write_bytes((image_dir / "a_keep.jpg").read_bytes())

    report = analyze_curation(
        ["a_keep.jpg", "b_copy.jpg"],
        path_loader=lambda path: image_dir / path,
        protected_paths=["b_copy.jpg"],
    )

    duplicate = next(
        finding
        for finding in report["findings"]
        if finding["kind"] == "duplicate"
    )
    assert duplicate["path"] == "a_keep.jpg"
    assert duplicate["neighbor"] == "b_copy.jpg"
    assert duplicate["recommended"] is True


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

    report = analyze_curation(
        ["a_high.png", "b_low.png"],
        path_loader=lambda path: image_dir / path,
    )

    duplicate = next(
        finding
        for finding in report["findings"]
        if finding["kind"] == "duplicate"
    )
    assert duplicate["path"] == "b_low.png"
    assert duplicate["neighbor"] == "a_high.png"
    assert report["recommended_paths"] == ["b_low.png"]


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


def test_dataset_curate_applies_deletions_and_annotation_cleanup():
    class FakeClient:
        def __init__(self):
            self.removals = []
            self.saved = None

        def download_file(self, source):
            return source

        def remove_file(self, path):
            self.removals.append(path)

        def save_json(self, path, value):
            self.saved = (path, value)

    client = FakeClient()
    dataset = Dataset("project", client=client)
    dataset.images = [
        {"name": "keep.jpg", "path": "keep.jpg"},
        {"name": "copy.jpg", "path": "copy.jpg"},
    ]
    dataset.annotations = [
        {"filename": name, "path": name, "objects": []}
        for name in ("keep.jpg", "copy.jpg")
    ]
    report = {
        "findings": [],
        "recommended_paths": ["copy.jpg"],
        "deleted": [],
    }

    analysis = {}

    def fake_analyze(paths, **kwargs):
        analysis["paths"] = paths
        analysis["protected_paths"] = kwargs["protected_paths"]
        for path in paths:
            kwargs["path_loader"](path)
        return report

    result = dataset.curate(apply=True, analyze_fn=fake_analyze)

    assert analysis["paths"] == ["keep.jpg", "copy.jpg"]
    assert analysis["protected_paths"] == ["keep.jpg", "copy.jpg"]
    assert result["deleted"] == ["copy.jpg"]
    assert dataset.curation_findings == result["findings"]
    assert "project/copy.jpg" in client.removals
    assert dataset.images == [{"name": "keep.jpg", "path": "keep.jpg"}]
    assert [annotation["filename"] for annotation in dataset.annotations] == [
        "keep.jpg"
    ]
    assert client.saved == ("project/annotations.json", dataset.annotations)
