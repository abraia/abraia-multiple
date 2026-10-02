import cv2
import numpy as np
from io import BytesIO

from PIL import Image

from abraia.editing.compress import compare_mse, compare_mssim, filter_gaussian
from abraia.editing.compress import convert_mode, getsize, optimal_quality, save_jpeg, save_webp


img1 = cv2.imread('images/person.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.GaussianBlur(img1, (11, 11), 1.5)


def test_mse_uses_the_range_of_float_images():
    image = np.ones((2, 2), dtype=np.float32)
    assert compare_mse(image, np.zeros_like(image)) == 1


def test_mse_supports_explicit_float_data_range():
    image = np.full((2, 2), 255, dtype=np.float32)
    assert compare_mse(image, np.zeros_like(image), data_range=255) == 1


def test_optimal_quality_color():
    im = Image.open('images/lion.jpg')
    q = optimal_quality(im)[0]
    assert type(q) == int


def test_optimal_quality_gray():
    im = Image.open('images/skate_gray.jpg')
    q = optimal_quality(im)[0]
    assert type(q) == int


def test_save_jpeg(tmp_path):
    im = Image.open('images/birds.jpg')
    output = tmp_path / 'optimal.jpg'
    save_jpeg(im, str(output), None)
    assert output.is_file()


def test_save_webp(tmp_path):
    im = Image.open('images/lion.jpg')
    output = tmp_path / 'optimal.webp'
    save_webp(im, str(output))
    assert output.is_file()


def test_convert_mode_returns_the_requested_mode():
    assert convert_mode(Image.new('L', (2, 2)), 'RGB').mode == 'RGB'
    assert convert_mode(Image.new('RGB', (2, 2)), 'RGBA').mode == 'RGBA'


def test_getsize_reports_bytesio_payload_size():
    stream = BytesIO(b'x' * 1000)
    assert getsize(stream) == 1000


def test_mssim_rejects_images_too_small_for_its_window():
    image = np.zeros((5, 5), dtype=np.uint8)
    try:
        compare_mssim(image, image)
    except ValueError as error:
        assert 'at least 11' in str(error)
    else:
        raise AssertionError('Expected small images to be rejected')


def test_filter_gaussian_valid_one_pixel_kernel_keeps_shape():
    image = np.zeros((3, 3), dtype=np.uint8)
    assert filter_gaussian(image, 1, 0, mode='valid').shape == image.shape
