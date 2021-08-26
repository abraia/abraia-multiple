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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow import keras
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Conv2D, Conv3D, Flatten, Dense, Reshape, Dropout

from .deep import plot_train_history


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
        class_names = ['', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                       'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                       'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                       'Stone Steel Towers']
        return data_hsi, gt_hsi, class_names
    if dataset == 'PU':
        if not os.path.exists('datasets/PaviaU.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                          'datasets/PaviaU.mat')
        if not os.path.exists('datasets/PaviaU_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                          'datasets/PaviaU_gt.mat')
        data_hsi = sio.loadmat('datasets/PaviaU.mat')['paviaU']
        gt_hsi = sio.loadmat('datasets/PaviaU_gt.mat')['paviaU_gt']
        class_names = ['', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                       'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
        return data_hsi, gt_hsi, class_names
    if dataset == 'SA':
        if not os.path.exists('datasets/Salinas_corrected.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
                          'datasets/Salinas_corrected.mat')
        if not os.path.exists('datasets/Salinas_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
                          'datasets/Salinas_gt.mat')
        data_hsi = sio.loadmat('datasets/Salinas_corrected.mat')['salinas_corrected']
        gt_hsi = sio.loadmat('datasets/Salinas_gt.mat')['salinas_gt']
        class_names = ['', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                       'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                       'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                       'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
        return data_hsi, gt_hsi, class_names
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


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX


def patch(data, height_index, width_index, patch_size):
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    return data[height_slice, width_slice, :]


def create_patches(X, patch_size):
    patches = []
    width, height = X.shape[1], X.shape[0]
    X = padWithZeros(X, patch_size // 2)
    for i in range(height):
        for j in range(width):
            image_patch = patch(X, i, j, patch_size)
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


def generate_training_data(X, y, window_size, K, train_ratio=0.7):
    X = principal_components(X, n_components=K)
    X, y = create_image_cubes(X, y, window_size)
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio)
    X_train = X_train.reshape(-1, window_size, window_size, K, 1)
    X_test = X_test.reshape(-1, window_size, window_size, K, 1)
    return X_train, X_test, y_train, y_test


def create_model(window_size, n_bands, output_units):
    ## input layer
    input_layer = Input((window_size, window_size, n_bands, 1))
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
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    # compiling the model
    adam = keras.optimizers.Adam(learning_rate=0.001, decay=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, batch_size=256, epochs=50):
    history = model.fit(x=X_train, y=np_utils.to_categorical(y_train), batch_size=batch_size, epochs=epochs)
    return history


def evaluate_model(model, X_test, y_test):
    return np.argmax(model.predict(X_test), axis=1)


def predict_model(model, X, patch_size, K):
    width, height = X.shape[1], X.shape[0]
    X = principal_components(X, n_components=K)
    X_pred = create_patches(X, patch_size)
    y_pred = np.argmax(model.predict(X_pred), axis=1)
    output = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            k = i * width + j
            output[i, j] = y_pred[k]
    return output.astype(int)
