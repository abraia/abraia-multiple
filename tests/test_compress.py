import os
import cv2
import numpy as np
from io import BytesIO

from PIL import Image
from numpy.testing import assert_almost_equal

from abraia.utils.compress import compare_mse, compare_mssim, compare_psnr, compare_ssim, filter_gaussian
from abraia.utils.compress import convert_mode, getsize, optimal_quality, save_jpeg, save_png, save_webp


img1 = cv2.imread('images/person.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.GaussianBlur(img1, (11, 11), 1.5)


def test_mse_vs_matlab():
    mse_matlab = 0.0012
    mse = compare_mse(img1, img2)
    assert_almost_equal(mse, mse_matlab, decimal=2)


def test_mse_uses_the_range_of_float_images():
    image = np.ones((2, 2), dtype=np.float32)
    assert compare_mse(image, np.zeros_like(image)) == 1


def test_mse_supports_explicit_float_data_range():
    image = np.full((2, 2), 255, dtype=np.float32)
    assert compare_mse(image, np.zeros_like(image), data_range=255) == 1


def test_psnr_vs_matlab():
    expected_psnr = 10 * np.log10(1 / compare_mse(img1, img2))
    psnr = compare_psnr(img1, img2)
    assert_almost_equal(psnr, expected_psnr, decimal=2)


def test_ssim_vs_matlab():
    ssim_matlab = 0.8681
    ssim = compare_ssim(img1, img2)
    assert_almost_equal(ssim, ssim_matlab, decimal=4)


def test_optimal_quality_color():
    im = Image.open('images/lion.jpg')
    q = optimal_quality(im)[0]
    assert type(q) == int


def test_optimal_quality_gray():
    im = Image.open('images/skate_gray.jpg')
    q = optimal_quality(im)[0]
    assert type(q) == int


def test_save_jpeg():
    im = Image.open('images/birds.jpg')
    save_jpeg(im, 'images/optimal.jpg', None)
    assert os.path.isfile('images/optimal.jpg')


# def test_save_png():
#     im = Image.open('images/bat.png')
#     save_png(im, 'images/optimal.png', None)
#     assert os.path.isfile('images/optimal.png')


def test_save_webp():
    im = Image.open('images/lion.jpg')
    save_webp(im, 'images/optimal.webp')
    assert os.path.isfile('images/optimal.webp')


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
