import os
import wget
import glob
import shutil
import zipfile
import tempfile
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.inception_v3 import preprocess_input

from .plot import plot_image, plot_images, plot_train_history

tempdir = tempfile.gettempdir()


def download(url):
    basename = os.path.basename(url)
    dest = os.path.join(tempdir, basename)
    if not os.path.exists(dest):
        wget.download(url, dest)
    return dest


def load_dataset(dataset='cats-and-dogs'):
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    if dataset == 'cats-and-dogs':
        zip_file = download('https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('datasets/')
        os.remove('datasets/PetImages/Cat/666.jpg')
        os.remove('datasets/PetImages/Dog/11702.jpg')
        cat_paths = glob.glob('datasets/PetImages/Cat/*.jpg')
        dog_paths = glob.glob('datasets/PetImages/Dog/*.jpg')
        class_names = ['Cat', 'Dog']
        return cat_paths, dog_paths, class_names


def split_train_test(cat_paths, dog_paths, train_ratio=0.7):
    cats_train, cats_test = train_test_split(cat_paths, test_size=1-train_ratio)
    dogs_train, dogs_test = train_test_split(dog_paths, test_size=1-train_ratio)
    folders = ['train', 'test', 'train/Cat', 'train/Dog', 'test/Cat', 'test/Dog']
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    for cat_train in cats_train:
        shutil.move(cat_train, 'train/Cat')
    for dog_train in dogs_train:
        shutil.move(dog_train, 'train/Dog')
    for cat_test in cats_test:
        shutil.move(cat_test, 'test/Cat')
    for dog_test in dogs_test:
        shutil.move(dog_test, 'test/Dog')
    return 'train', 'test'


def create_inception3_model(n_classes):
    base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = Dropout(0.4)(GlobalAveragePooling2D(name='avg_pool')(base_model.output))
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_inception3_model(model, X):
    x = np.expand_dims(X, axis=0)
    x = keras.applications.inception_v3.preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def create_mobilenet2_model(n_classes):
    base_model = keras.applications.MobileNetV2(input_shape=(160, 160, 3), weights='imagenet', include_top=False)
    base_model.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_mobilenet2_model(model, X):
    x = np.expand_dims(X, axis=0)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


class ClassificationModel:
    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes
        if self.name == 'inception3':
            self.model = create_inception3_model(self.n_classes)
        elif self.name == 'mobilenet2':
            self.model = create_mobilenet2_model(self.n_classes)

    def preprocess_input(self, x):
        if self.name == 'inception3':
            return keras.applications.inception_v3.preprocess_input(x)
        elif self.name == 'mobilenet2':
            return keras.applications.mobilenet_v2.preprocess_input(x)

    def train(self, train_generator, validation_generator, epochs=5):
        # return model.fit(train_generator, epochs=epochs, steps_per_epoch=320, validation_data=validation_generator, validation_steps=60)
        self.history = self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
        return self.history

    def predict(self, X):
        if self.name == 'inception3':
            return predict_inception3_model(self.model, X)
        elif self.name == 'mobilenet2':
            return predict_mobilenet2_model(self.model, X)

    def plot_history(self):
        if self.history:
            plot_train_history(self.history)
    
    def save(self, filename='model.h5'):
        self.model.save(filename)

    def load(self, filename='model.h5'):
        self.model = load_model(filename)


def create_model(name, n_classes):
    """Create a new model: inception3 or mobilenet2"""
    return ClassificationModel(name, n_classes)
