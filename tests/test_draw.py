import numpy as np

from abraia.utils.draw import draw_mask, draw_overlay, render_results


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

    result = render_results(image, [{'mask': mask, 'color': '#000000'}])

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


def test_render_results_does_not_draw_boxes_for_pose_keypoints(monkeypatch):
    import abraia.utils.draw as draw

    def fail_if_box_is_rendered(*_args, **_kwargs):
        raise AssertionError("pose results must not render bounding boxes")

    monkeypatch.setattr(draw, "render_box", fail_if_box_is_rendered)
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
