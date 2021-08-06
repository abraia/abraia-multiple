import os
import spectral
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.decomposition import PCA
from PIL import Image


cf = os.path.dirname(os.path.abspath(__file__))

try:
    BANDS_WLTH = np.array([463, 469, 478, 490, 502, 514, 525, 540, 553, 555, 565, 579, 593, 601, 623, 631])
    CMF = np.array(np.loadtxt(os.path.join(cf, 'cie-cmf_1nm.txt'), usecols=(0, 1, 2, 3)))
except:
    pass


def __spec_to_xyz(hsi, bands=BANDS_WLTH):
    """Convert HSI cube in the visible espectrum to XYZ image (CIE1931)"""
    size, nbands = hsi.shape[:2], hsi.shape[2]
    X, Y, Z = np.zeros(size), np.zeros(size), np.zeros(size)
    for i in range(nbands):
        band_cmf = np.array(CMF[np.where(CMF == BANDS_WLTH[i])[0]])
        band = hsi[:, :, i]
        X = X + band_cmf[0][1] * band
        Y = Y + band_cmf[0][2] * band
        Z = Z + band_cmf[0][3] * band
    return np.dstack([X, Y, Z])


def __xyz_to_sRGB(XYZ):
    """Convert XYZ (CIE1931) image to sRGB image"""
    X, Y, Z = XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2]
    # https://en.wikipedia.org/wiki/SRGB
    r = 3.24096994 * X - 1.53738318 * Y - 0.49861076 * Z
    g = -0.96924364 * X + 1.8759675 * Y + 0.04155506 * Z
    b = 0.5563008 * X - 0.20397696 * Y + 1.05697151 * Z
    #from skimage.color import rgb2xyz, xyz2rgb
    #rgb = xyz2rgb(XYZ)
    rgb = np.dstack((r, g, b))
    addwhite = np.amin(rgb)
    r = r - addwhite
    g = g - addwhite; b = b - addwhite
    rgb = np.dstack([r, g, b])
    # Gamma function (https://en.wikipedia.org/wiki/SRGB )
    R = np.maximum((1.055 * np.power(r, 0.41667)) - 0.055, 12.92 * r)
    G = np.maximum((1.055 * np.power(g, 0.41667)) - 0.055, 12.92 * g)
    B = np.maximum((1.055 * np.power(b, 0.41667)) - 0.055, 12.92 * b)
    return np.dstack([R, G, B])


def spec_to_rgb(cube):
    return __xyz_to_sRGB(__spec_to_xyz(cube))


def random(img, n_bands=6, indexes=False):
    """Returns a list of random bands"""
    bands = []
    indexes = []
    for i in range(n_bands):
        q = np.random.randint(img.shape[2])
        indexes.append(q)
        bands.append(img[:, :, q])
    if indexes:
        return bands, indexes
    return bands


def rgb(img, bands=None):
    """Returns the RGB image from the selected bands (R, G, B)"""
    return spectral.get_rgb(img, bands=bands)


def ndvi(img, red_band, nir_band):
    """Returns the NDVI image from the specified read and nir bands"""
    return spectral.ndvi(img, red_band, nir_band)


def resample(img, n_samples=32):
    """Resamples the number of spectral bands (n_samples)"""
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    r = resample(np.transpose(X), n_samples=n_samples)
    return np.transpose(r).reshape(h, w, n_samples)


def principal_components(img, n_components=3, spectrum=False):
    """Calculate principal components of the image"""
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    pca = PCA(n_components=n_components)
    bands = pca.fit_transform(X).reshape(h, w, n_components)
    if spectrum:
        bands, pca.components_
    return bands


def resize(img, size):
    """Resize the image to the given size (w, h)"""
    return np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))


def normalize(img):
    """Normalize the image to the range [0, 1]"""
    min, max = np.amin(img), np.amax(img)
    return (img - min) / (max - min)


def saliency(img):
    """Calculate saliency map of the image"""
    smaps = []
    for n in range(img.shape[2]):
        band = img[:, :, n]
        h, w = band.shape
        fft = np.fft.fft2(resize(band, (64, 64)))
        log_amplitude, phase = np.log(np.absolute(fft)), np.angle(fft)
        spectral_residual = log_amplitude - nd.uniform_filter(log_amplitude, size=3, mode='nearest')
        smap = np.absolute(np.fft.ifft2(np.exp(spectral_residual + 1.j * phase)))
        smap = nd.gaussian_filter(smap, sigma=3)
        smaps.append(normalize(resize(smap, (w, h))))
    return np.sum(np.dstack(smaps), axis=2)


def spectrum(img, point=None):
    """Get the spectrum at a given point (x, y)

    When a point is not specified the spectrum of the most salient point is returned.
    """
    if point is None:
        sal = saliency(img)
        idx = np.unravel_index(np.argmax(sal), sal.shape)
        point = (idx[1], idx[0])
    return img[point[1], point[0], :]
