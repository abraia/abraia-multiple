import os
import tifffile
import numpy as np
from PIL import Image

try:
    import spectral
    spectral.settings.envi_support_nonlowercase_params = True
except ImportError:
    print('Install the spectral package to work with envi files')

try:
    import scipy
except ImportError:
    print('Install the scipy package to work with matlab files')

from .client import Abraia
from .utils import temporal_src


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

    def load_mat(self, path):
        mat = scipy.io.loadmat(self.download_file(path, cache=True))
        for key, value in mat.items():
            if type(value) == np.ndarray:
                return value
        return mat

    def load_mosaic(self, path, size=(4, 4)):
        r, c = size
        img = self.load_image(path)
        return np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])

    def load_image(self, path, mosaic_size=None):
        if path.lower().endswith('.hdr'):
            img = self.load_envi(path)
        elif path.lower().endswith('.mat'):
            img = self.load_mat(path)
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            img = tifffile.imread(self.download_file(path, cache=True))
        else:
            img = np.asarray(Image.open(self.download_file(path, cache=True)))
        if mosaic_size and len(img.shape) == 2:
            r, c = mosaic_size
            img = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return img

    def save_envi(self, path, img, metadata={}):
        src = temporal_src(path)
        spectral.io.envi.save_image(src, img, metadata=metadata, force=True)
        self.upload_file(f"{src.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload_file(src, path)

    def save_mat(self, path, img):
        src = temporal_src(path)
        scipy.io.savemat(src, {'raw': img})
        return self.upload_file(src, path)

    def save_image(self, path, img, metadata={}):
        src = temporal_src(path)
        if path.lower().endswith('.hdr'):
            return self.save_envi(path, img, metadata)
        elif path.lower().endswith('.mat'):
            return self.save_mat(path, img)
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            tifffile.imwrite(src, img)
            return self.upload_file(src, path)
        else:
            Image.fromarray(img).save(src)
            return self.upload_file(src, path)
    