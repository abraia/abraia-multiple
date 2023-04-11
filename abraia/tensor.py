from .multiple import Multiple, tempdir

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map

from tensorflow import keras
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input, Cropping2D , Conv2D, Conv3D, Flatten, Reshape
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.callbacks import EarlyStopping, ModelCheckpoint


multiple = Multiple()


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
        random.shuffle(ids)
        paths = [paths[id] for id in ids]
        labels = [labels[id] for id in ids]
    return paths, labels


def normalize(img):
    """Normalize the image to the range [0, 1]"""
    min, max = np.amin(img), np.amax(img)
    return (img - min) / (max - min)


def principal_components(img, n_components=3, spectrum=False):
    """Calculate principal components of the image"""
    h, w, d = img.shape
    X = img.reshape((h * w), d)
    pca = PCA(n_components=n_components, whiten=True)
    bands = pca.fit_transform(X).reshape(h, w, n_components)
    if spectrum:
        bands, pca.components_
    return bands


def load_projects():
    folders = multiple.list_files()[1]
    return [folder['name'] for folder in folders if folder['name'] != 'export']


def class_to_category(val, class_names):
    class_id = class_names.index(val)
    return to_categorical(class_id, num_classes=len(class_names))


def category_to_class(val, class_names):
    class_id = val.argmax()
    return class_names[class_id]


def pad_with_zeros(X, margin=2):
    return np.pad(X, ((margin, margin), (margin, margin), (0, 0)))


def create_patch(data, height_index, width_index, patch_size):
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    return data[height_slice, width_slice, :]


def patch_generator(X, y, patch_size, samples, class_names, batch_size=32):
    height, width, depth = X.shape
    X = pad_with_zeros(X, patch_size // 2)
    while True:
        batch_X, batch_Y = [], []
        idxs = random.sample(range(len(samples)), batch_size)
        for idx in idxs:
            i, j = idx // width, idx % width
            batch_X.append(create_patch(X, i, j, patch_size).astype('float32'))
            batch_Y.append(class_to_category(y[i, j], class_names))
        yield np.array(batch_X), np.array(batch_Y)


def split_train_test(paths, labels, train_ratio=0.7):
    return train_test_split(paths, labels, test_size=1-train_ratio)


def data_generator(paths, labels, load_image, class_names, batch_size=32):
    process_map(multiple.cache_file, paths, max_workers=5)
    while True:
        batch_X, batch_Y = [], []
        idxs = random.sample(range(len(paths)), batch_size)
        batch_X = [load_image(paths[idx]) for idx in idxs]
        batch_Y = [class_to_category(labels[idx], class_names) for idx in idxs]
        yield np.array(batch_X), np.array(batch_Y)


def create_inception3_model(n_classes):
    base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = Dropout(0.4)(GlobalAveragePooling2D(name='avg_pool')(base_model.output))
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_mobilenet2_model(n_classes, input_shape=(160, 160, 3)):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)
    base_model.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_hsn_model(n_classes, input_shape=(100, 100, 16)):
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_hsi_densenet(n_classes, input_shape=(100, 100, 16), crop=False):
    input_images = Input(shape=input_shape)
    if crop:
        # Crop hypercubes to a spatial window
        crop_images = Cropping2D(cropping=crop, input_shape=input_shape)(input_images)
        # Reduce spectral dimensionality to 3
        input_tensor = Conv2D(3, (1, 1))(crop_images)
        hsi_layers = 2
    else:
        # Reduce spectral dimensionality to 3
        input_tensor = Conv2D(3, (1, 1))(input_images)
        hsi_layers = 1
    base_model = DenseNet(include_top=False, weights=None, input_tensor=input_tensor)
    # Load weights pre-trained on imagenet      
    base_model_imagenet = DenseNet(include_top=False, weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3))
    for i, layer in enumerate(base_model_imagenet.layers):
        # Skip first layer with no weights
        if i == 0:
            continue
        base_model.layers[i+hsi_layers].set_weights(layer.get_weights())
    # Global spatial average pooling layer
    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    # or just flatten the layers
    # top_model = Flatten()(top_model)
    predictions = Dense(n_classes, activation='softmax')(top_model)
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    # set convolution block for reducing 13 to 3 layers trainable
    for layer in model.layers[:(1+hsi_layers)]:
        layer.trainable = True
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def preprocess_inception3_input(x):
    return keras.applications.inception_v3.preprocess_input(x)


def preprocess_mobilenet2_input(x):
    return keras.applications.mobilenet_v2.preprocess_input(x)


def predict_inception3_model(model, X):
    x = np.expand_dims(X, axis=0)
    x = keras.applications.inception_v3.preprocess_input(x)
    return model.predict(x)


def predict_mobilenet2_model(model, X):
    x = np.expand_dims(X, axis=0)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    return model.predict(x)


def create_model(name, n_classes, input_shape=(100, 100, 16), crop=False):
    """Create a new model: densenet, 3d_hsn, inception3, or mobilenet2"""
    if name == 'densenet':
        return create_hsi_densenet(n_classes, input_shape, crop)
    if name == '3d_hsn':
        return create_hsn_model(n_classes, input_shape)
    if name == 'inception3':
        return create_inception3_model(n_classes)
    if name == 'mobilenet2':
        return create_mobilenet2_model(n_classes)    


def train_model(model,  train_generator, test_generator, epochs=50, train_steps=16, test_steps=5):
    checkpoint_path = tempdir + '/model_{epoch:02d}.hdf5'
    checkpointer = ModelCheckpoint(checkpoint_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystopper = EarlyStopping(monitor='val_categorical_accuracy', patience=10, mode='max', restore_best_weights=True)
    return model.fit(train_generator, epochs=epochs, steps_per_epoch=train_steps, validation_data=test_generator, validation_steps=test_steps, callbacks=[checkpointer, earlystopper])
    # history = model.fit(train_generator, validation_data=test_generator, epochs=epochs, steps_per_epoch=train_steps, validation_steps=test_steps)
    # plot_train_history(self.history)    
    return history


def predict_model(model, paths, load_image, class_names):
    imgs = np.array([load_image(path) for path in paths])
    y_pred = model.predict(imgs)
    return [category_to_class(val, class_names) for val in y_pred]


def save_model(model, path):
    src = os.path.join(tempdir, path)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    model.save(src)
    multiple.upload_file(src, path)


def load_model(path):
    dest = multiple.cache_file(path)
    return keras.models.load_model(dest)


def save_classes(path, class_names):
    txt = '\n'.join(class_names)
    multiple.save_file(path, txt)


def load_classes(path):
    txt = multiple.load_file(path)
    return [line.strip() for line in txt.splitlines()]


def plot_image(img, title=''):
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = normalize(principal_components(img))
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def plot_images(imgs, titles=None, cmap='nipy_spectral'):
    plt.figure()
    k = len(imgs)
    r = int(math.sqrt(k))
    c = math.ceil(k / r)
    ax = plt.subplots(r, c)[1].reshape(-1)
    for i, img in enumerate(imgs):
        if titles and len(titles) >= k:
            ax[i].title.set_text(titles[i])
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = normalize(principal_components(img))
        ax[i].imshow(img, cmap=cmap)
        ax[i].axis('off')
    plt.show()


def plot_train_history(history):
    plt.ylim(0, 1.01)
    plt.grid()
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss','Test accuracy'], loc='upper right')
