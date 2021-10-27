import os
import wget
import tempfile
import numpy as np
import scipy.io as sio
import scipy.ndimage as nd

from PIL import Image
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow import keras
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv3D, Flatten, Dense, Reshape, Dropout

from .plot import plot_image, plot_images, plot_train_history

tempdir = tempfile.gettempdir()


def download(url):
    basename = os.path.basename(url)
    dest = os.path.join(tempdir, basename)
    if not os.path.exists(dest):
        wget.download(url, dest)
    return dest


def load_dataset(dataset):
    """Load one of the available hyperspectral datasets (IP, PU, SA, KSC)."""
    if dataset == 'IP':
        data_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'))['indian_pines_corrected']
        gt_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'))['indian_pines_gt']
        class_names = ['', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                       'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                       'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                       'Stone Steel Towers']
        return data_hsi, gt_hsi, class_names
    if dataset == 'PU':
        data_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat'))['paviaU']
        gt_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'))['paviaU_gt']
        class_names = ['', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                       'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
        return data_hsi, gt_hsi, class_names
    if dataset == 'SA':
        data_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat'))['salinas_corrected']
        gt_hsi = sio.loadmat(download(
            'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'))['salinas_gt']
        class_names = ['', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                       'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                       'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                       'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
        return data_hsi, gt_hsi, class_names
    if dataset == 'KSC':
        data_hsi = sio.loadmat(download(
            'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat'))['KSC']
        gt_hsi = sio.loadmat(download(
            'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'))['KSC_gt']
        return data_hsi, gt_hsi


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


def split_train_test(X, y, train_ratio=0.7):
    """Split data for training and test"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, stratify=y)
    return X_train, X_test, y_train, y_test


def principal_components(img, n_components=3, spectrum=False):
    """Calculate principal components of the image"""
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    pca = PCA(n_components=n_components, whiten=True)
    bands = pca.fit_transform(X).reshape(h, w, n_components)
    if spectrum:
        bands, pca.components_
    return bands


def pad_with_zeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def create_patch(data, height_index, width_index, patch_size):
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    return data[height_slice, width_slice, :]


# TODO: Convert create patches to generator with batch_size parameter
def create_patches(X, patch_size):
    patches = []
    width, height = X.shape[1], X.shape[0]
    X = pad_with_zeros(X, patch_size // 2)
    for i in range(height):
        for j in range(width):
            image_patch = create_patch(X, i, j, patch_size)
            patches.append(image_patch.reshape(image_patch.shape + (1,)).astype('float32'))
    return np.array(patches)


def create_image_cubes(X, y, patch_size):
    width, height = X.shape[1], X.shape[0]
    patchesData = create_patches(X, patch_size)
    labels = []
    for i in range(height):
        for j in range(width):
            labels.append(y[i, j])
    patchesLabels = np.array(labels)
    return patchesData, patchesLabels


def generate_training_data(X, y, patch_size, train_ratio=0.7):
    X, y = create_image_cubes(X, y, patch_size)
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio)
    X_train = X_train.reshape(-1, patch_size, patch_size, X.shape[-1], 1)
    X_test = X_test.reshape(-1, patch_size, patch_size, X.shape[-1], 1)
    return X_train, X_test, y_train, y_test


def create_hsn_model(input_shape, n_classes):
    input_layer = Input((*input_shape, 1))
    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
    conv_layer3 = Reshape((conv_layer3.shape[1], conv_layer3.shape[2], conv_layer3.shape[3] * conv_layer3.shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(conv_layer3)
    flatten_layer = Flatten()(conv_layer4)
    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=n_classes, activation='softmax')(dense_layer2)
    # define and compile the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    adam = keras.optimizers.Adam(learning_rate=0.001, decay=1e-06)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_hsn_model(model, X, patch_size):
    width, height = X.shape[1], X.shape[0]
    X_pred = create_patches(X, patch_size)
    y_pred = np.argmax(model.predict(X_pred), axis=1)
    return y_pred.reshape(height, width).astype(int)


class HyperspectralModel:
    def __init__(self, name, *args):
        self.name = name
        if self.name == 'svm':
            self.model = SVC(C=150, kernel='rbf')
        elif self.name == 'hsn':
            self.input_shape, self.n_classes = args
            self.model = create_hsn_model(self.input_shape, self.n_classes) # Hybrid Spectral Net

    def train(self, X, y, train_ratio=0.7, epochs=50):
        if self.name == 'svm':
            X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, X.shape[-1]), y, train_size=train_ratio, stratify=y)
            self.model.fit(X_train, y_train)
            return y_test, self.model.predict(X_test)
        elif self.name == 'hsn':
            X = principal_components(X, n_components=self.input_shape[2])
            X_train, X_test, y_train, y_test = generate_training_data(X, y, self.input_shape[0], train_ratio)
            self.history = self.model.fit(x=X_train, y=np_utils.to_categorical(y_train), batch_size=256, epochs=epochs)
            return y_test, np.argmax(self.model.predict(X_test), axis=1)

    def predict(self, X):
        if self.name == 'svm':
            return self.model.predict(X.reshape(-1, X.shape[2])).reshape(X.shape[0], X.shape[1])
        elif self.name == 'hsn':
            X = principal_components(X, n_components=self.input_shape[2])
            return predict_hsn_model(self.model, X, self.input_shape[0])
    
    def plot_history():
        if self.history:
            plot_train_history(self.history)
    
    def save(self, filename='model.h5'):
        self.model.save(filename)

    def load(self, filename='model.h5'):
        self.model = load_model(filename)


def create_model(name, *args):
    """Create a new model: svm or hsn"""
    return HyperspectralModel(name, *args)
