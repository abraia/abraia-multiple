import numpy as np
from abraia.inference import ops, Clip


def test_softmax_values():
    logits = np.array([0, 10, -10])
    assert np.isclose(np.sum(ops.softmax(logits)), 1)


def test_clip_import():
    assert Clip is not None


def test_mask_to_polygon_accepts_boolean_masks():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    polygon = ops.mask_to_polygon(mask)

    assert len(polygon) >= 3


from abraia.utils.stream import load_images_opencv
import cv2

def test_load_images_opencv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.jpg"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)

    images = load_images_opencv(str(p))
    assert len(images) == 1
    assert images[0].shape == (10, 10, 3)

    images_dir = load_images_opencv(str(d))
    assert len(images_dir) == 1
    assert images_dir[0].shape == (10, 10, 3)
