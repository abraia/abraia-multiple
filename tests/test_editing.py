import os

from abraia.utils import load_image
import numpy as np
import pytest

from abraia.editing import build_mask, detect_faces, detect_plates, detect_smartcrop
from abraia.editing.inpaint import stitching
from abraia.editing.removebg import BackgroundRemover
from abraia.editing.smartcrop import best_crop_area
from abraia.utils.draw import draw_mask, draw_overlay, render_results


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


def test_draw_overlay_clips_rectangles_outside_the_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.full((3, 3, 3), 255, dtype=np.uint8)

    result = draw_overlay(image, overlay, rect=(-1, -1, 3, 3))

    assert result.shape == image.shape
    assert np.all(result[:2, :2] == 255)
    assert np.all(result[2:, 2:] == 0)


def test_draw_mask_resizes_to_the_box_dimensions():
    overlay = np.zeros((5, 5, 3), dtype=np.uint8)

    draw_mask(overlay, np.ones((1, 1), dtype=np.uint8), (1, 1, 3, 2), (10, 20, 30))

    np.testing.assert_array_equal(
        overlay[1:3, 1:4], np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8)
    )
    assert np.all(overlay[0] == 0)


def test_render_results_paints_full_image_black_masks():
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.uint8)

    result = render_results(image, [{"mask": mask, "color": "#000000"}])

    assert np.all(result < 255)


def test_render_results_does_not_draw_boxes_for_instance_masks(monkeypatch):
    import abraia.utils.draw as draw

    def fail_if_box_is_rendered(*_args, **_kwargs):
        raise AssertionError("instance masks must not render bounding boxes")

    monkeypatch.setattr(draw, "render_box", fail_if_box_is_rendered)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.uint8)

    draw.render_results(
        image,
        [{"box": [2, 2, 3, 3], "mask": mask, "color": "#FFFFFF"}],
    )


def test_render_results_draws_boxes_for_pose_keypoints(monkeypatch):
    import abraia.utils.draw as draw

    rendered_boxes = []
    monkeypatch.setattr(
        draw,
        "render_box",
        lambda *args, **_kwargs: rendered_boxes.append(args),
    )

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[:3] = [[5, 5], [10, 10], [15, 15]]

    draw.render_results(
        image,
        [{
            "box": [2, 2, 20, 20],
            "keypoints": keypoints,
            "joint_scores": np.ones(17),
            "color": "#FFFFFF",
        }],
    )

    assert len(rendered_boxes) == 1
