import sys
import cv2
import numpy as np
from math import log
from io import BytesIO

from PIL import Image
from PIL import ImageCms
from PIL import ImageColor


def _assert_compatible(im1, im2):
    """Raise an error if the shape and dtype do not match."""
    if not im1.dtype == im2.dtype:
        raise ValueError('Input images must have the same dtype.')
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return


def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    if im1.dtype != np.float32:
        im1 = im1.astype(np.float32)
    if im2.dtype != np.float32:
        im2 = im2.astype(np.float32)
    return im1 / 255, im2 / 255


def compare_mse(img1, img2):
    """Compute the mean-squared error (MSE) between two images.

    Arguments:
        img1 (ndarray): Image of any dimensionality.
        img2 (ndarray): Image of any dimensionality.

    Returns:
        mse (float): The mean-squared error (MSE) metric.
    """
    _assert_compatible(img1, img2)
    img1, img2 = _as_floats(img1, img2)
    return np.mean(np.square(img1 - img2))


def compare_psnr(img1, img2):
    """Compute the peak signal to noise ratio (PSNR) comparing two images.

    Arguments:
        img1 (ndarray): Image of any dimensionality.
        img2 (ndarray): Image of any dimensionality.

    Returns:
        psnr (float): The peak signal to noise ratio (PSNR) metric.

    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _assert_compatible(img1, img2)
    img1, img2 = _as_floats(img1, img2)
    _mse = compare_mse(img1, img2)
    if _mse == 0:
        return 100
    else:
        return 10 * np.log10(1 / _mse)


def _preprocess_images(img1, img2):
    _assert_compatible(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
            img2 = img2[:, :, :3]
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return _as_floats(img1, img2)


def _downsample(img1, img2):
    img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return img1, img2


def _ssim_map(img1, img2, C1=6.5025, C2=58.5225):
    """Standard SSIM computation."""
    img1_img1 = cv2.pow(img1, 2)
    img2_img2 = cv2.pow(img2, 2)
    img1_img2 = cv2.multiply(img1, img2)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu11 = cv2.GaussianBlur(img1_img1, (11, 11), 1.5, dst=img1_img1)
    mu22 = cv2.GaussianBlur(img2_img2, (11, 11), 1.5, dst=img2_img2)
    mu12 = cv2.GaussianBlur(img1_img2, (11, 11), 1.5, dst=img1_img2)
    mu1_mu2 = cv2.multiply(mu1, mu2)
    mu1_mu1 = cv2.pow(mu1, 2, dst=mu1)
    mu2_mu2 = cv2.pow(mu2, 2, dst=mu2)
    b1 = cv2.addWeighted(mu11, 1, mu1_mu1, -1, 0, dst=mu11)
    b2 = cv2.addWeighted(mu22, 1, mu2_mu2, -1, 0, dst=mu22)
    A2 = cv2.addWeighted(mu12, 2, mu1_mu2, -2, C2, dst=mu12)
    A1 = cv2.addWeighted(mu1_mu2, 1, mu1_mu2, 1, C1, dst=mu1_mu2)
    B1 = cv2.addWeighted(mu1_mu1, 1, mu2_mu2, 1, C1, dst=mu1_mu1)
    B2 = cv2.addWeighted(b1, 1, b2, 1, C2, dst=b1)
    A = cv2.multiply(A1, A2, dst=A1)
    B = cv2.multiply(B1, B2, dst=B1)
    ssim_map = cv2.divide(A, B, dst=A)
    return ssim_map


def compare_ssim(img1, img2):
    """Compute the structural similarity index (SSIM) between two images, to
    address image quality comparison by taking texture into account.

    Arguments:
        img1 (ndarray): Image of any dimensionality.
        img2 (ndarray): Image of any dimensionality.

    Returns:
        ssim: (float): The structural similarity index (SSIM).

    .. [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image
        quality assessment: From error visibility to structural similarity,"
        IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612,
        Apr. 2004.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,

    .. [2] Zhou Wang; Bovik, A.C.; ,"Mean squared error: Love it or leave it?
        A new look at Signal Fidelity Measures," Signal Processing Magazine,
        IEEE, vol. 26, no. 1, pp. 98-117, Jan. 2009.

    .. [3] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
    """
    img1, img2 = _preprocess_images(img1, img2)
    C1, C2 = 0.01 * 0.01, 0.03 * 0.03
    S = _ssim_map(img1, img2, C1, C2)
    return S.mean()


def compare_mssim(img1, img2):
    """Compute the multi-scale structural similarity index (MS-SSIM) between
    two images.

    Arguments:
        img1 (ndarray): Image of any dimensionality.
        img2 (ndarray): Image of any dimensionality.

    Returns:
        msssim: (float): The multi-scale structural similarity index (MS-SSIM).

    .. [1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural
    similarity for image quality assessment," Invited Paper, IEEE Asilomar
    Conference on Signals, Systems and Computers, Nov. 2003
    """
    img1, img2 = _preprocess_images(img1, img2)
    C1, C2 = 0.01 * 0.01, 0.03 * 0.03
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    ssims = []
    for l in range(len(weights)):
        if l > 0:
            img1, img2 = _downsample(img1, img2)
        if min(img1.shape) < 11:
            break
        ssim_map = _ssim_map(img1, img2, C1, C2)
        ssims.append(cv2.mean(ssim_map)[0])
    return np.sum(np.array(ssims) * weights[:len(ssims)]) / np.sum(weights[:len(ssims)])


def filter_gaussian(img, size, sigma, mode='same'):
    """To avoid edge effects 'valid' mode will ignore filter radius strip
    around edges."""
    out = cv2.GaussianBlur(img, (size, size), sigma)
    if mode == 'valid':
        pad = (size-1) // 2
        return out[pad:-pad, pad:-pad]
    return out


def downsample(img):
    """Implements the MATLAB filter imfilter(im1, ker, 'symmetric', 'same')."""
    ker = np.ones((2, 2)) / 4
    filtered = cv2.filter2D(
        img, -1, ker, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
    return filtered[::2, ::2]


def adobe_to_srgb(im):
    """Return a new image that is sRGB when it has a ICC profile."""
    try:
        icc = im.info.get('icc_profile')
        if icc:
            srgb = ImageCms.createProfile('sRGB')
            im = ImageCms.profileToProfile(im, BytesIO(icc), srgb)
    except:
        print('PyCMSError: cannot build transform')
    return im


def alpha_to_color(im, background='white'):
    if im.mode == 'RGBA':
        color = ImageColor.getcolor(background, 'RGB')
        back = Image.new('RGB', im.size, color)
        back.paste(im, (0, 0), im)
        return back
    return im


def mode_to_color(im):
    if im.mode == 'L' or im.mode == 'P':
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')
    elif im.mode == 'LA':
        im = im.convert('RGBA')
    elif im.mode == 'RGBX':
        im = im.convert('RGB')
    return im


def convert_mode(im, mode, background='white'):
    im = adobe_to_srgb(im) # color management function
    if im.mode == mode:
        return im
    if mode == 'RGB':
        return alpha_to_color(im, background)
    else:
        im.convert(mode)
        return im


def to_array(im):
    im = mode_to_color(im)
    return np.asarray(im)


def resize_image(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[1] > 1200:
        scale = (0.5 * (img.shape[1] - 1200) + 1200) / img.shape[1]
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def luminance(img):
    limg = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]
    return np.mean(limg) / 255


def red_ratio(img):
    kernel = np.ones((8, 8), np.uint8)
    hsv = cv2.cvtColor(cv2.blur(img, (4, 4)), cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, (0, 240, 180), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 240, 180), (180, 255, 255))
    binary = cv2.bitwise_or(red1, red2)
    return np.mean(cv2.erode(binary, kernel)) / 255


def optimal_threshold(img):
    area = img.shape[0] * img.shape[1]
    # k = max(0.0021 * log(7.5e-6 * area), 0)
    # thr = k * (1 - luminance(img)) + 0.45 * k + 0.0003
    # thr = 0.4 * thr if red_ratio(img) > 0.004 else thr
    k = max(0.00089 * log(8.9e-6 * area), 0)
    thr = 10/3 * k * (1 - luminance(img)) + k + 0.0003
    return thr


def minimal_quality(img):
    area = img.shape[0] * img.shape[1]
    # qmin = 100 - 7.5 * log(0.00004 * area)
    qmin = min(100 - 6.5 * log(0.00006 * area), 95)
    return int(qmin)


def red_correction(img, thr, qmin):
    red = red_ratio(img)
    thr = 0.33 * thr if red > 0.007 else thr
    qmin = (100 + qmin) / 2 if red > 0.05 * 0.007 else qmin
    return thr, int(qmin)


def rgb2pca(img):
    X = img.reshape(-1, img.shape[-1])
    mean, eigenvectors = cv2.PCACompute(X, np.float32([]))
    return cv2.PCAProject(X, mean, eigenvectors).reshape(img.shape)


def saturation(img):
    kernel = np.ones((9, 9), np.uint8)
    hsv = cv2.cvtColor(cv2.blur(img, (4, 4)), cv2.COLOR_RGB2HSV)
    binary = cv2.inRange(hsv, (0, 200, 80), (255, 255, 255))
    return np.mean(cv2.erode(binary, kernel)) / 255


def salient_sat(img):
    imgpca = rgb2pca(cv2.blur(img, (4, 4)))
    L2_ab = np.sqrt(np.square(imgpca[:, :, 1]) + np.square(imgpca[:, :, 2]))
    return np.max(L2_ab)


def subsample(img):
    subsample = (saturation(img) < 0.0042) and (salient_sat(img) < 120)
    return subsample


def number_of_colors(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    hist = cv2.calcHist(
        [img], [0, 1, 2], None, [16, 64, 32], [0, 256, 0, 256, 0, 256])
    ncolors = np.sum(hist > 0.0002 * (img.shape[0] * img.shape[1]))
    return ncolors


def getsize(stream):
    if isinstance(stream, BytesIO):
        return sys.getsizeof(stream)
    return len(stream)


def stats(original, optimized):
    orisize = getsize(original) / 1024
    optsize = getsize(optimized) / 1024
    reduction = 100 * (1 - (optsize / orisize))
    return orisize, optsize, reduction


def encode_jpeg(im, quality, subsampling, progressive):
    output = BytesIO()
    im.save(output, 'JPEG', quality=quality, progressive=progressive, subsampling=subsampling, optimize=True)
    output.seek(0)
    return output


def optimal_quality(im, thr=0.003, qmin=60, qmax=95):
    img = to_array(im)
    rimg = resize_image(img)
    rim = Image.fromarray(rimg)
    progressive = True
    thr = optimal_threshold(img)
    qmin = minimal_quality(img)
    thr, qmin = red_correction(img, thr, qmin)
    gray1 = rimg if rimg.ndim == 2 else cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
    gray2 = np.zeros(gray1.shape, dtype=np.uint8)
    q0, q1 = 100, qmin
    d0, d1 = 0, 1
    q = q1
    for k in range(5):
        fileobj = encode_jpeg(rim, q, 0, False)
        progressive = getsize(fileobj) > 10240
        gray2[...] = cv2.cvtColor(to_array(Image.open(fileobj)), cv2.COLOR_RGB2GRAY)
        mssim = compare_mssim(gray1, gray2)
        d = 1 / mssim - 1
        if abs(thr - d) < (0.05 * thr):
            break
        if d > thr and d < d1:
            d1, q1 = d, q
        if d < thr and d > d0:
            d0, q0 = d, q
        q = int(q0 + (thr - d0) * (q1 - q0) / (d1 - d0 + 1e-10))
        if q1 == q or q1 == q0 or q > 100 or q < qmin:
            break
    q = np.clip(q, qmin, qmax)
    subsampling = 2 if subsample(img) else 0
    return int(q), progressive, subsampling
    

def save_jpeg(im, dest, quality=None, subsampling=2, progressive=False):
    im = alpha_to_color(mode_to_color(im))
    if quality is None:
        quality, progressive, subsampling = optimal_quality(im)
    im.save(dest, 'JPEG', quality=quality, progressive=progressive, subsampling=subsampling, optimize=True)
    return dest


def save_png(im, dest, quality=None):
    if quality is None or quality < 100:
        im = mode_to_color(im)
        im = im.quantize(colors=256, method=3)
    im.save(dest, format='PNG', optimize=True)
    return dest


def save_webp(im, dest, quality=75):
    lossless = True if quality == 100 else False
    im.save(dest, format='WEBP', lossless=lossless, quality=quality, optimize=True)
    return dest
