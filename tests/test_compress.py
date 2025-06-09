import os
import cv2

from PIL import Image
from numpy.testing import assert_almost_equal

from abraia.utils.compress import compare_mse, compare_psnr, compare_ssim
from abraia.utils.compress import optimal_quality, save_jpeg, save_png, save_webp


img1 = cv2.imread('images/person.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.GaussianBlur(img1, (11, 11), 1.5)


def test_mse_vs_matlab():
    mse_matlab = 0.0012
    mse = compare_mse(img1, img2)
    assert_almost_equal(mse, mse_matlab, decimal=2)


def test_psnr_vs_matlab():
    psnr_matlab = 77.4923
    psnr = compare_psnr(img1, img2)
    assert_almost_equal(psnr, psnr_matlab, decimal=2)


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
