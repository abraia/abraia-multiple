import os

from abraia.utils import load_image
import numpy as np
import pytest

from abraia.editing import build_mask, detect_faces, detect_plates, detect_smartcrop
from abraia.editing.inpaint import stitching
from abraia.editing.removebg import BackgroundRemover
from abraia.editing.smartcrop import best_crop_area


LIVE_TESTS_ENABLED = os.environ.get("ABRAIA_RUN_LIVE_TESTS") == "1"
LIVE_TEST_REASON = (
    "Live editing model tests are disabled; "
    "set ABRAIA_RUN_LIVE_TESTS=1"
)


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason=LIVE_TEST_REASON)
def test_detect_faces():
    img = load_image('images/rolling-stones.jpg')
    results = detect_faces(img)
    assert isinstance(results, list)


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason=LIVE_TEST_REASON)
def test_detect_plates():
    img = load_image('images/car.jpg')
    results = detect_plates(img)
    assert isinstance(results, list)


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason=LIVE_TEST_REASON)
def test_detect_smartcrop():
    img = load_image('images/mick-jagger.jpg')
    roi = detect_smartcrop(img, (150, 300))
    assert isinstance(roi, list)


def test_build_mask_accepts_segmentation_masks_from_plate_detector():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    mask = build_mask(
        image,
        [{'box': [2, 3, 4, 2], 'mask': np.ones((2, 4), dtype=np.uint8)}],
        [],
    )

    assert mask.dtype == np.uint8
    assert np.all(mask[3:5, 2:6] == 255)
    assert mask.sum() == 8 * 255


def test_background_remover_preprocess_uses_fixed_scaling():
    remover = BackgroundRemover.__new__(BackgroundRemover)
    remover.image_size = (2, 2)
    remover.input_mean = (0.5, 0.5, 0.5)
    remover.input_std = (1.0, 1.0, 1.0)

    black = remover.preprocess(np.zeros((1, 1, 3), dtype=np.uint8))
    gray = remover.preprocess(np.full((1, 1, 3), 64, dtype=np.uint8))

    assert np.isfinite(black).all()
    assert not np.array_equal(black, gray)


def test_best_crop_area_falls_back_for_empty_candidate_set():
    saliency = np.zeros((32, 32), dtype=np.uint8)

    assert best_crop_area(saliency, [], saliency, 1.0) == [0, 0, 32, 32]


def test_inpaint_stitching_blends_overlapping_tiles():
    first = np.zeros((2, 4, 3), dtype=np.uint8)
    second = np.full((2, 4, 3), 100, dtype=np.uint8)

    output = stitching([first, second], (6, 2), overlap=2)

    assert output.shape == (2, 6, 3)
    assert np.all(output[:, :2] == 0)
    assert np.all(output[:, 2:4] == 50)
    assert np.all(output[:, 4:] == 100)
