from .multiple import Multiple, tempdir

import os
import wget
import random
import numpy as np

from random import choice, sample
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input, Cropping2D , Conv2D, Conv3D, Flatten, Reshape
from keras.applications.densenet import DenseNet201 as DenseNet

from .plot import plot_image, plot_images, plot_train_history


multiple = Multiple()


def download(url):
    basename = os.path.basename(url)
    dest = os.path.join(tempdir, basename)
    if not os.path.exists(dest):
        wget.download(url, dest)
    return dest


def load_projects():
    folders = multiple.list_files()[1]
    return [folder['name'] for folder in folders]


def class_to_category(val, class_names):
    class_id = class_names.index(val)
    return to_categorical(class_id, num_classes=len(class_names))


def category_to_class(val, class_names):
    class_id = val.argmax()
    return class_names[class_id]


def load_dataset(dataset, shuffle=False):
    paths, labels = [], []
    [files, folders] = multiple.list_files(f"{dataset}/")
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


def split_train_test(paths, labels, train_ratio=0.7):
    return train_test_split(paths, labels, test_size=1-train_ratio)


def data_generator(paths, labels, load_image, class_names, batch_size=32):
    process_map(multiple.load_file, paths, max_workers=5)
    while True:
        batch_X, batch_Y = [], []
        idxs = sample(range(len(paths)), batch_size)
        batch_X = [load_image(path) for path in paths[idxs]]
        batch_Y = [class_to_category(label, class_names) for label in labels[idxs]]
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


def predict_model(model, name, X):
    if name == 'inception3':
        return predict_inception3_model(model, X)
    if name == 'mobilenet2':
        return predict_mobilenet2_model(model, X)


def train_model(model,  train_generator, validation_generator, epochs=5):
    # return model.fit(train_generator, epochs=epochs, steps_per_epoch=320, validation_data=validation_generator, validation_steps=60)
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    # plot_train_history(self.history)    
    return history


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


def save_model(model, path):
    src = os.path.join(tempdir, path)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    model.save(src)
    multiple.upload_file(src, path)


def load_model(path):
    dest = multiple.load_file(path)
    return keras.models.load_model(dest)
