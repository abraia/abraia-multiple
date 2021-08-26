import os
import wget
import glob
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img


def load_dataset(dataset='cats-and-dogs'):
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    if dataset == 'cats-and-dogs':
        zip_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
        zip_file = 'datasets/kagglecatsanddogs_3367a.zip'
        if not os.path.exists('datasets/kagglecatsanddogs_3367a.zip'):
            wget.download(zip_url, zip_file)
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


def create_model(CLASSES=2):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dropout(0.4)(x)
    predictions = Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # transfer learning
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_generator, validation_generator, EPOCHS=5, STEPS_PER_EPOCH=320, VALIDATION_STEPS=64):
    history = model.fit(
        train_generator, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator, validation_steps=VALIDATION_STEPS)
    return history


def predict_model(model, img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_train_history(history):
    plt.ylim(0, 1.01)
    plt.grid()
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss','Test accuracy'], loc='upper right')
