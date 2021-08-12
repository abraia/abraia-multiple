import os
import io
import tempfile
import mimetypes
import numpy as np

from PIL import Image
from abraia import Abraia

try:
    from spectral.io import envi
except ImportError:
    print('Install spectral module to support envi files')
    
try:
    from scipy.io import loadmat, savemat
except ImportError:
    print('Install scipy module to support mat files')


tempdir = tempfile.gettempdir()

class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_file(self, path):
        return self.download(path)

    def load_header(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download(path, dest)
        return envi.read_envi_header(dest)

    def load_metadata(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(path)

    def load_envi(self, path):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        if not os.path.exists(dest):
            self.download(path, dest)
        raw = f"{dest.split('.')[0]}.raw"
        if not os.path.exists(raw):
            self.download(f"{path.split('.')[0]}.raw", raw)
        return np.array(envi.open(dest, raw)[:, :, :])

    def load_mat(self, path):
        mat = loadmat(self.download(path))
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
        return np.asarray(Image.open(self.download(path)))

    def save_envi(self, path, img, metadata={}):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        envi.save_image(dest, img, metadata=metadata, force=True)
        self.upload(f"{dest.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload(dest, path)

    def save_mat(self, path, img):
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        savemat(dest, {'raw': img})
        return self.upload(dest, path)

    def save_image(self, path, img, metadata={}):
        if path.lower().endswith('.hdr'):
            return self.save_envi(path, img, metadata)
        elif path.lower().endswith('.mat'):
            return self.save_mat(path, img)
        # stream = io.BytesIO()
        # mime = mimetypes.guess_type(path)[0]
        # format = mime.split('/')[1]
        # Image.fromarray(img).save(stream, format)
        # print(mime, format)
        basename = os.path.basename(path)
        dest = os.path.join(tempdir, basename)
        Image.fromarray(img).save(dest)
        return self.upload(dest, path)

