import os
import wget
import numpy as np
import scipy.io as sio
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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
    from spectral import get_rgb
    return get_rgb(img, bands=bands)


def ndvi(img, red_band, nir_band):
    """Returns the NDVI image from the specified read and nir bands"""
    from spectral import ndvi
    return ndvi(img, red_band, nir_band)


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
    pca = PCA(n_components=n_components, whiten=True)
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


def load_dataset(dataset):
    """Load one of the available hyperspectral datasets (IP, PU, SA, KSC)."""
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    
    if dataset == 'IP':
        if not os.path.exists('datasets/Indian_pines_corrected.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                          'datasets/Indian_pines_corrected.mat')
        if not os.path.exists('datasets/Indian_pines_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
                          'datasets/Indian_pines_gt.mat')
        data_hsi = sio.loadmat(
            'datasets/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt_hsi = sio.loadmat('datasets/Indian_pines_gt.mat')['indian_pines_gt']

    if dataset == 'PU':
        if not os.path.exists('datasets/PaviaU.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                          'datasets/PaviaU.mat')
        if not os.path.exists('datasets/PaviaU_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                          'datasets/PaviaU_gt.mat')
        data_hsi = sio.loadmat('datasets/PaviaU.mat')['paviaU']
        gt_hsi = sio.loadmat('datasets/PaviaU_gt.mat')['paviaU_gt']

    if dataset == 'SA':
        if not os.path.exists('datasets/Salinas_corrected.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
                          'datasets/Salinas_corrected.mat')
        if not os.path.exists('datasets/Salinas_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
                          'datasets/Salinas_gt.mat')
        data_hsi = sio.loadmat('datasets/Salinas_corrected.mat')['salinas_corrected']
        gt_hsi = sio.loadmat('datasets/Salinas_gt.mat')['salinas_gt']

    if dataset == 'KSC':
        if not os.path.exists('datasets/KSC.mat'):
            wget.download('http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                          'datasets/KSC.mat')
        if not os.path.exists('datasets/KSC_gt.mat'):
            wget.download('http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat',
                          'datasets/KSC_gt.mat')
        data_hsi = sio.loadmat('datasets/KSC.mat')['KSC']
        gt_hsi = sio.loadmat('datasets/KSC_gt.mat')['KSC_gt']
    return data_hsi, gt_hsi


def split_train_test(X, y, train_ratio=0.7):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio)
    return X_train, X_test, y_train, y_test
