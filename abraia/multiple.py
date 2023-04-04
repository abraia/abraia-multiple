import os
import random
import tempfile
import tifffile
import numpy as np
from PIL import Image

try:
    import spectral
    from spectral.io import envi
    spectral.settings.envi_support_nonlowercase_params = True
except ImportError:
    print('Install the spectral package to read envi files')

from .abraia import Abraia

tempdir = tempfile.gettempdir()


class Multiple(Abraia):
    def __init__(self, folder=''):
        super(Multiple, self).__init__()

    def load_file(self, path):
        dest = os.path.join(tempdir, path)
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            self.download_file(path, dest)
        return dest

    def load_header(self, path):
        dest = self.load_file(path)
        return envi.read_envi_header(dest)

    def load_metadata(self, path):
        if path.lower().endswith('.hdr'):
            return self.load_header(path)
        return super(Multiple, self).load_metadata(path)

    def load_envi(self, path):
        dest = self.load_file(path)
        raw = f"{dest.split('.')[0]}.raw"
        if not os.path.exists(raw):
            self.download_file(f"{path.split('.')[0]}.raw", raw)
        return np.array(envi.open(dest, raw)[:, :, :])

    def load_mat(self, path):
        from scipy.io import loadmat
        mat = loadmat(self.load_file(path))
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
            img = tifffile.imread(self.load_file(path))
        else:
            img = np.asarray(Image.open(self.load_file(path)))
        if mosaic_size and len(img.shape) == 2:
            r, c = mosaic_size
            img = np.dstack([img[(k % r)::r, (k // c)::c] for k in range(r * c)])
        return img

    def save_envi(self, path, img, metadata={}):
        src = os.path.join(tempdir, path)
        os.makedirs(os.path.dirname(src), exist_ok=True)
        envi.save_image(src, img, metadata=metadata, force=True)
        self.upload_file(f"{src.split('.')[0]}.img", f"{path.split('.')[0]}.raw")
        return self.upload_file(src, path)

    def save_mat(self, path, img):
        from scipy.io import savemat
        src = os.path.join(tempdir, path)
        os.makedirs(os.path.dirname(src), exist_ok=True)
        savemat(src, {'raw': img})
        return self.upload_file(src, path)

    def save_image(self, path, img, metadata={}):
        src = os.path.join(tempdir, path)
        os.makedirs(os.path.dirname(src), exist_ok=True)
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

    # TODO: Add load_csv, save csv, from to pandas

    def load_dataset(self, dataset, shuffle=True):
        paths, labels = [], []
        [files, folders] = self.list_files(f"{dataset}/")
        for folder in folders:
            files = self.list_files(folder['path'])[0]
            paths.extend([file['path'] for file in files])
            labels.extend(len(files) * [folder['name']])
        if shuffle:
            ids = list(range(len(paths)))
            random.shuffle(ids)
            paths = [paths[id] for id in ids]
            labels = [labels[id] for id in ids]
        return paths, labels

    def load_model(self, path, model):
        dest = os.path.join(tempdir, 'model.h5')
        self.download_file(path, dest)
        model.load(dest)

    def save_model(self, path, model):
        src = os.path.join(tempdir, 'model.h5')
        model.save(src)
        self.upload_file(src, path)
    