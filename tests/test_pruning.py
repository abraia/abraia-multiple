from types import SimpleNamespace

import numpy as np
import pytest

from abraia.training import is_rgb_dataset
from abraia.training.quality import analyze_rgb_quality


def test_rgb_pruning_finds_quality_candidates():
    base = np.zeros((32, 32, 3), dtype=np.uint8)
    base[8:24, 8:24] = 200
    images = {
        "base.jpg": base,
        "duplicate.jpg": base.copy(),
        "dark.jpg": np.full_like(base, 5),
        "bright.jpg": np.full_like(base, 250),
        "outlier.jpg": np.random.default_rng(4).integers(
            0, 256, base.shape, dtype=np.uint8
        ),
    }
    records = [
        {"name": name, "path": name}
        for name in images
    ]

    findings = analyze_rgb_quality(records, images.__getitem__)
    reasons = {
        finding["path"]: set(finding["reasons"])
        for finding in findings
    }

    assert "duplicate" in reasons["duplicate.jpg"]
    assert "dark" in reasons["dark.jpg"]
    assert "bright" in reasons["bright.jpg"]
    assert "blurry" in reasons["dark.jpg"]
    assert any(
        "outlier" in candidate_reasons for candidate_reasons in reasons.values()
    )


def test_rgb_pruning_reports_start_and_completion_for_each_image():
    records = [
        {"name": "one.jpg", "path": "one.jpg"},
        {"name": "two.jpg", "path": "two.jpg"},
    ]
    progress = []
    images = {
        "one.jpg": np.zeros((8, 8, 3), dtype=np.uint8),
        "two.jpg": np.full((8, 8, 3), 100, dtype=np.uint8),
    }

    analyze_rgb_quality(
        records,
        images.__getitem__,
        progress_callback=lambda *event: progress.append(event),
    )

    assert progress == [
        (0, 2, "one.jpg", False),
        (1, 2, "one.jpg", True),
        (1, 2, "two.jpg", False),
        (2, 2, "two.jpg", True),
    ]


def test_rgb_pruning_rejects_spectral_records():
    record = {"name": "scene.tiff", "path": "scene.tiff"}

    with pytest.raises(ValueError, match="only for RGB"):
        analyze_rgb_quality([record], lambda _path: np.zeros((4, 4, 3)))


def test_rgb_pruning_uses_perceptual_similarity_for_brightness_changes():
    base = np.zeros((40, 40, 3), dtype=np.uint8)
    base[8:32, 10:30] = [180, 120, 80]
    brighter = np.clip(base.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    records = [
        {"name": "base.jpg", "path": "base.jpg"},
        {"name": "brighter.jpg", "path": "brighter.jpg"},
    ]

    findings = analyze_rgb_quality(
        records,
        {"base.jpg": base, "brighter.jpg": brighter}.__getitem__,
    )

    assert findings[0]["path"] == "brighter.jpg"
    assert findings[0]["reasons"] == ("highly similar",)


def test_rgb_pruning_detects_small_translations_as_similar():
    base = np.zeros((60, 60, 3), dtype=np.uint8)
    base[12:48, 15:45] = [190, 100, 60]
    translated = np.roll(base, 3, axis=(0, 1))
    records = [
        {"name": "base.jpg", "path": "base.jpg"},
        {"name": "translated.jpg", "path": "translated.jpg"},
    ]

    findings = analyze_rgb_quality(
        records,
        {"base.jpg": base, "translated.jpg": translated}.__getitem__,
    )

    translated_finding = next(
        finding for finding in findings if finding["path"] == "translated.jpg"
    )
    assert "highly similar" in translated_finding["reasons"]


def test_rgb_dataset_capability_requires_only_standard_images():
    assert is_rgb_dataset(SimpleNamespace(images=[{"name": "photo.jpg"}]))
    assert not is_rgb_dataset(SimpleNamespace(images=[{"name": "scene.tiff"}]))
    assert not is_rgb_dataset(SimpleNamespace(images=[]))


def test_feature_cache_is_namespaced_for_different_sources(tmp_path):
    record = {"name": "same.jpg", "path": "same.jpg"}
    cache_dir = tmp_path / "cache"

    first = analyze_rgb_quality(
        [record],
        lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
        cache_dir=cache_dir,
        cache_namespace="dataset-a",
        include_all=True,
    )
    second = analyze_rgb_quality(
        [record],
        lambda _path: np.full((8, 8, 3), 255, dtype=np.uint8),
        cache_dir=cache_dir,
        cache_namespace="dataset-b",
        include_all=True,
    )

    assert first[0]["metrics"]["brightness"] == 0.0
    assert second[0]["metrics"]["brightness"] == 1.0


def test_feature_cache_context_invalidates_statistics_profile(tmp_path):
    record = {"name": "same.jpg", "path": "same.jpg"}
    cache_dir = tmp_path / "cache"
    image = np.full((8, 8, 3), 128, dtype=np.uint8)

    analyze_rgb_quality(
        [record],
        lambda _path: image,
        cache_dir=cache_dir,
        cache_namespace="dataset",
        compute_stats=False,
        include_all=True,
    )
    result = analyze_rgb_quality(
        [record],
        lambda _path: image,
        cache_dir=cache_dir,
        cache_namespace="dataset",
        compute_stats=True,
        include_all=True,
    )

    assert "contrast" in result[0]["metrics"]
