import os
import math
import numpy as np
from PIL import Image

from ..client import Abraia
from ..utils import temporal_src


def load_image(src):
    """Load an image from the given source"""
    return np.asarray(Image.open(src))


def save_image(src, img):
    """Save an image to the given source"""
    Image.fromarray(img).save(src)


def load_tiff(src):
    """Load a TIFF image from the given source"""
    import tifffile
    return tifffile.imread(src)


def save_tiff(src, img):
    """Save a TIFF image to the given source"""
    import tifffile
    tifffile.imwrite(src, img)


def decode_mosaic(img, size=(4, 4)):
    """Decode a mosaic image into its spectral bands"""
    r, c = size
    return np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])


def load_mat(src):
    """Load a MATLAB .mat file from the given source"""
    import scipy
    mat = scipy.io.loadmat(src)
    for key, value in mat.items():
        if type(value) == np.ndarray:
            return value
    return mat


def save_mat(src, img):
    """Save an image to a MATLAB .mat file at the given source"""
    import scipy
    scipy.io.savemat(src, {'raw': img})


def principal_components(img, n_components=3, spectrum=False):
    """Calculate principal components of the image"""
    from sklearn.decomposition import PCA
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    pca = PCA(n_components=n_components, whiten=True)
    bands = np.squeeze(pca.fit_transform(X).reshape(h, w, n_components))
    if spectrum:
        return bands, pca.components_
    return bands


def normalize(img):
    """Normalize the image to the range [0, 1]"""
    min, max = np.amin(img), np.amax(img)
    return (img - min) / (max - min)


def calculate_gray(img):
    """Calculate the grayscale image by PCA method"""
    pc_img = principal_components(img, n_components=1)
    return np.uint8(255 * normalize(np.squeeze(pc_img)))


def create_visible(src):
    img = decode_mosaic(load_tiff(src))
    gray = calculate_gray(img / 65535)
    save_image(f"{os.path.splitext(src)[0]}.png", gray)


def random(img, n_bands=6, indexes=False):
    """Return a list of random bands"""
    bands = []
    indexes = []
    for i in range(n_bands):
        q = np.random.randint(img.shape[2])
        indexes.append(q)
        bands.append(img[:, :, q])
    if indexes:
        return bands, indexes
    return bands


def resize(img, size):
    """Resize the image to the given size (w, h)"""
    return np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))


def resample(img, n_samples=32):
    """Resample the number of spectral bands (n_samples)"""
    from sklearn.utils import resample
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    r = resample(np.transpose(X), n_samples=n_samples)
    return np.transpose(r).reshape(h, w, n_samples)


def plot_image(img, title=''):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_images(imgs, titles=None, cmap='nipy_spectral'):
    import matplotlib.pyplot as plt
    plt.figure()
    k = len(imgs)
    r = int(math.sqrt(k))
    c = math.ceil(k / r)
    ax = plt.subplots(r, c)[1].reshape(-1)
    for i in range(k):
        if titles and len(titles) >= k:
            ax[i].title.set_text(titles[i])
        ax[i].imshow(imgs[i], cmap=cmap)
        ax[i].axis('off')
    plt.show()


try:
    import spectral
    spectral.settings.envi_support_nonlowercase_params = True
except ImportError:
    print('Install the spectral package to work with envi files')


def rgb(img, bands=None):
    """Return the RGB image from the selected bands (R, G, B)"""
    from spectral import get_rgb
    return get_rgb(img, bands=bands)


def ndvi(img, red_band, nir_band):
    """Return the NDVI image from the specified read and nir bands"""
    from spectral import ndvi
    return ndvi(img, red_band, nir_band)


class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_header(self, path):
        dest = self.download_file(path, cache=True)
        return spectral.io.envi.read_envi_header(dest)

    def load_metadata(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(path)

    def load_envi(self, path):
        dest = self.download_file(path, cache=True)
        raw = self.download_file(f"{path.split('.')[0]}.raw", cache=True)
        return np.array(spectral.io.envi.open(dest, raw)[:, :, :])

    def load_mosaic(self, path, size=(4, 4)):
        img = self.load_image(path)
        return decode_mosaic(img, size)

    def load_image(self, path, mosaic_size=None):
        if path.lower().endswith('.hdr'):
            img = self.load_envi(path)
        elif path.lower().endswith('.mat'):
            img = load_mat(self.download_file(path, cache=True))
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            img = load_tiff(self.download_file(path, cache=True))
        else:
            img = load_image(self.download_file(path, cache=True))
        if mosaic_size and len(img.shape) == 2:
            img = decode_mosaic(img, mosaic_size)
        return img

    def save_envi(self, path, img, metadata={}):
        src = temporal_src(path)
        spectral.io.envi.save_image(src, img, metadata=metadata, force=True)
        self.upload_file(f"{src.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload_file(src, path)

    def save_image(self, path, img, metadata={}):
        src = temporal_src(path)
        if path.lower().endswith('.hdr'):
            return self.save_envi(path, img, metadata)
        elif path.lower().endswith('.mat'):
            save_mat(path, img)
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            save_tiff(src, img)
        else:
            save_image(src, img)
        return self.upload_file(src, path)
    