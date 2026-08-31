import math
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from PIL import Image

from ..training.ops import train_test_split


from . import Multiple, random, principal_components, rgb, ndvi, resample

multiple = Multiple()


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred)))
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, target_names=None):
    cm = confusion_matrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    support = np.sum(cm, axis=1)
    
    report = "              precision    recall  f1-score   support\n\n"
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        name = target_names[i] if target_names and i < len(target_names) else str(i)
        report += f"{name:>12}       {p:.2f}      {r:.2f}      {f:.2f}      {s:>7}\n"
    
    report += f"\n    accuracy                           {accuracy_score(y_true, y_pred):.2f}      {np.sum(support):>7}\n"
    return report


def load_dataset(dataset, shuffle=False):
    """Load one of the available hyperspectral datasets (IP, PU, SA)."""
    paths, labels = [], []
    files, folders = multiple.list_files(f"{dataset}/")
    if dataset == 'IP':
        paths = [file['path'] for file in files]
        if 'Indian_pines_corrected.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat', f"{dataset}/Indian_pines_corrected.mat")
        if 'Indian_pines_gt.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat', f"{dataset}/Indian_pines_gt.mat")
        data_hsi = multiple.load_image(f"{dataset}/Indian_pines_corrected.mat")
        gt_hsi = multiple.load_image(f"{dataset}/Indian_pines_gt.mat")
        class_names = ['', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                       'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                       'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',
                       'Stone Steel Towers']
        return data_hsi, gt_hsi, class_names
    if dataset == 'PU':
        paths = [file['path'] for file in files]
        if 'PaviaU.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat', f"{dataset}/PaviaU.mat")
        if 'PaviaU_gt.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat', f"{dataset}/PaviaU_gt.mat")
        data_hsi = multiple.load_image(f"{dataset}/PaviaU.mat")
        gt_hsi = multiple.load_image(f"{dataset}/PaviaU_gt.mat")
        class_names = ['', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                       'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
        return data_hsi, gt_hsi, class_names
    if dataset == 'SA':
        paths = [file['path'] for file in files]
        if 'Salinas_corrected.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat', f"{dataset}/Salinas_corrected.mat")
        if 'Salinas_gt.mat' not in paths:
            multiple.upload_file('http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat', f"{dataset}/Salinas_gt.mat")
        data_hsi = multiple.load_image(f"{dataset}/Salinas_corrected.mat")
        gt_hsi = multiple.load_image(f"{dataset}/Salinas_gt.mat")
        class_names = ['', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                       'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
                       'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk',
                       'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
        return data_hsi, gt_hsi, class_names
    for folder in folders:
        files = multiple.list_files(folder['path'])[0]
        paths.extend([file['path'] for file in files])
        labels.extend(len(files) * [folder['name']])
    if shuffle:
        ids = list(range(len(paths)))
        # random.shuffle(ids)
        paths = [paths[id] for id in ids]
        labels = [labels[id] for id in ids]
    return paths, labels


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


def pad_with_zeros(X, margin=2):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)))


def create_patch(data, height_index, width_index, patch_size):
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    return data[height_slice, width_slice, :]


def create_patches(X, patch_size):
    patches = []
    height, width = X.shape[:2]
    X = pad_with_zeros(X, patch_size // 2)
    for i in range(height):
        for j in range(width):
            image_patch = create_patch(X, i, j, patch_size)
            patches.append(image_patch.reshape(image_patch.shape + (1,)).astype('float32'))
    return np.array(patches)


def create_image_cubes(X, y, patch_size):
    height, width = X.shape[:2]
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


def plot_image(img, title=''):
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
