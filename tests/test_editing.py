from abraia.utils import load_image
from abraia.editing import detect_faces, detect_plates, detect_smartcrop, edit_image


def test_detect_faces():
    img = load_image('images/rolling-stones.jpg')
    results = detect_faces(img)
    assert isinstance(results, list)


def test_detect_plates():
    img = load_image('images/car.jpg')
    results = detect_plates(img)
    assert isinstance(results, list)


def test_detect_smartcrop():
    img = load_image('images/mick-jagger.jpg')
    roi = detect_smartcrop(img, (150, 300))
    assert isinstance(roi, list)


def test_edit_smartcrop():
    img = load_image('images/mick-jagger.jpg')
    out = edit_image(img, mode='smartcrop', size=(150, 300))
    assert out.shape == (300, 150, 3)
