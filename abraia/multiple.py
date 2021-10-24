import os
import tempfile
import tifffile
import numpy as np

from .abraia import Abraia

# TODO: Add load and save csv, json? or pandas?


tempdir = tempfile.gettempdir()

class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_header(self, path):
        from spectral.io import envi
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        return envi.read_envi_header(dest)

    def load_metadata(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(path)

    def load_envi(self, path):
        from spectral.io import envi
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        raw = f"{dest.split('.')[0]}.raw"
        if not os.path.exists(raw):
            self.download_file(f"{path.split('.')[0]}.raw", raw)
        return np.array(envi.open(dest, raw)[:, :, :])

    def load_mat(self, path):
        from scipy.io import loadmat
        mat = loadmat(self.download_file(path))
        for key, value in mat.items():
            if type(value) == np.ndarray:
                return value
        return mat

    def load_tiff(self, path):
        return tifffile.imread(self.download_file(path))

    def load_mosaic(self, path, size=(4, 4)):
        r, c = size
        img = self.load_image(path)
        cube = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return cube

    def load_image(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_envi(path)
        elif path.lower().endswith('.mat'):
            return self.load_mat(path)
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            return self.load_tiff(path)
        return super(Multiple, self).load_image(path)

    def save_envi(self, path, img, metadata={}):
        from spectral.io import envi
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        envi.save_image(dest, img, metadata=metadata, force=True)
        self.upload_file(f"{dest.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload_file(dest, path)

    def save_mat(self, path, img):
        from scipy.io import savemat
        basename = os.path.basename(path)
        src = os.path.join(tempdir, basename)
        savemat(src, {'raw': img})
        return self.upload_file(src, path)
    
    def save_tiff(self, path, img):
        basename = os.path.basename(path)
        src = os.path.join(tempdir, basename)
        tifffile.imwrite(src, img)
        return self.upload_file(src, path)

    def save_image(self, path, img, metadata={}):
        if path.lower().endswith('.hdr'):
            return self.save_envi(path, img, metadata)
        elif path.lower().endswith('.mat'):
            return self.save_mat(path, img)
        elif path.lower().endswith('.tiff') or path.lower().endswith('.tif'):
            return self.save_tiff(path, img)
        return super(Multiple, self).save_image(path, img)
