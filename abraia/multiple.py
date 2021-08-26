import os
import tempfile
import numpy as np

from .abraia import Abraia


tempdir = tempfile.gettempdir()

class Multiple(Abraia):
    try:
        from spectral.io import envi
    except ImportError:
        print('Install spectral package to support envi files')
        
    try:
        from scipy.io import loadmat, savemat
    except ImportError:
        print('Install scipy package to support mat files')

    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_header(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        return self.envi.read_envi_header(dest)

    def load_metadata(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(path)

    def load_envi(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download_file(path, dest)
        raw = f"{dest.split('.')[0]}.raw"
        if not os.path.exists(raw):
            self.download_file(f"{path.split('.')[0]}.raw", raw)
        return np.array(self.envi.open(dest, raw)[:, :, :])

    def load_mat(self, path):
        mat = self.loadmat(self.download_file(path))
        for key, value in mat.items():
            if type(value) == np.ndarray:
                return value
        return mat

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
        return super(Multiple, self).load_image(path)

    def save_envi(self, path, img, metadata={}):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        self.envi.save_image(dest, img, metadata=metadata, force=True)
        self.upload_file(f"{dest.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload_file(dest, path)

    def save_mat(self, path, img):
        basename = os.path.basename(path)
        src = os.path.join(tempdir, basename)
        self.savemat(src, {'raw': img})
        return self.upload_file(src, path)

    def save_image(self, path, img, metadata={}):
        if path.lower().endswith('.hdr'):
            return self.save_envi(path, img, metadata)
        elif path.lower().endswith('.mat'):
            return self.save_mat(path, img)
        return super(Multiple, self).save_image(path, img)
