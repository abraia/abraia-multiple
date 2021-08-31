[![Build Status](https://github.com/abraia/abraia-python/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-python/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
[![Coverage Status](https://coveralls.io/repos/github/abraia/abraia-python/badge.svg)](https://coveralls.io/github/abraia/abraia-python)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia python API and CLI

## Abraia Multiple package

The Abraia Multiple package provides and easy and practical way to analyze and classify images directly from your browser. You just need to click on the open in colab button to start with one of the available notebooks:

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-python/blob/master/notebooks/image-analysis.ipynb) Simple image analysis

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-python/blob/master/notebooks/image-classification.ipynb) Deep image classification

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-python/blob/master/notebooks/hyperspectral-analysis.ipynb) Hyperspectral image analysis

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-python/blob/master/notebooks/hyperspectral-classification.ipynb) Deep hyperspectral image classification

![classification](https://store.abraia.me/multiple/notebooks/classification.jpg)

The multiple module extends the Abraia package to provide support for HyperSpectral Image (HSI) analysis and classification.

> MULTIPLE is result and it is being developed by ABRAIA in the [Multiple project](https://multipleproject.eu/).

## Configuration

Installed the package, you have to configure your [ABRAIA KEY](https://abraia.me/console/settings) as environment variable:

```sh
export ABRAIA_KEY=api_key
```

On Windows you need to use `set` instead of `export`:

```sh
set ABRAIA_KEY=api_key
```

NOTE: To persist the configuration use your system options to set your ABRAIA_KEY environment variable and avoid to run the previous command every time you start a terminal/console session.

## Image analysis

Abraia provides a direct interface to directly load and save images. You can easily load the image data and the file metadata, or save a new image.

```python
from abraia import Abraia

abraia = Abraia()

img = abraia.load_image('test.jpg')
meta = abraia.load_metadata('test.jpg')
abraia.save_image('test.png', img)
```

You can directly visualize the image using Matplotlib.

```python
import matplotlib.pyplot as plt

plt.figure()
plt.title('Image')
plt.imshow(img)
plt.axis('off')
plt.show()
```

### List files

Return the list of `files` and `folders` on the specified cloud `folder`.

```python
folder = ''
files, folders = abraia.list_files(folder)
```

### Upload files

Upload a local (`src`) or a remote (`url`) file to the cloud.

```python
src = 'images/test.png'
path = 'test/test.png'
abraia.upload_file(src, path)
```

### Image detection

Detect labels, capture text, and detect faces in images (must be in JPEG format).

```python
labels = abraia.detect_labels(path)
lines = abraia.capture_text(path)
faces = abraia.detect_faces(path)
```

### Transform images

Transform and optimize images automatically choosing every compression parameter to provide the best result based on the perceived analysis of the original image.

```python
path = 'test/birds.jpg'
dest = 'birds_o.jpg'
params = {'width': 300, 'height': 300, 'mode': 'pad'}
abraia.transform_image(path, dest, params)
```

Parameter | Description
----------|------------
width | Image width (original width by default)
height | Image height (original height by default)
mode | Resize and crop mode: crop, face, thumb, resize (smart crop by default)
background | Change background color in padded mode (white by default)
action | Path to the action file to be used as template
format | Set the image format: jpeg, png, gif, webp (original format by default)
quality | Set the image quality (auto by default)


### Download files

Retrieve an stored file.

```python
path = 'test/birds.jpg'
dest = 'images/birds.jpg'
abraia.download_file(path, dest)
```

### Delete files

Delete a stored resource specified by its `path`.

```python
abraia.delete(path)
```

## HyperSpectral Image (HSI) analysis

MULTIPLE extends the ABRAIA Python API to provide seamless integration of multiple HyperSpectral Image (HSI) processing and analysis tools. This integrates state-of-the-art image manipulation libraries to provide ready to go scalable multispectral solutions.

> MULTIPLE is result and it is being developed by ABRAIA in the [Multiple project](https://multipleproject.eu/).

### Usage

```python
from abraia import Multiple

multiple = Multiple()
img = multiple.load_image('test.hdr')
meta = multiple.load_metadata('test.hdr')
multiple.save_image('test.hdr', img, metadata=meta)
```

### Upload and load HSI data

To start with, we may [upload some data](https://abraia.me/console/gallery) directly using the graphical interface, or using the multiple api:

```python
multiple.upload_file('PaviaU.mat')
```

Now, we can load the hyperspectral image data (HSI cube) directly from the cloud:

```python
img = multiple.load_image('PaviaU.mat')
```

### Basic HSI visualization

Hyperspectral images cannot be directly visualized, so we can get some random bands from our HSI cube, and visualize these bands as like any other monochannel image.

```python
from abraia import hsi

imgs, indexes = hsi.random(img)

plt.figure()
fig, ax = plt.subplots(2, 3)
ax = ax.reshape(-1)
for i, im in enumerate(imgs):
    ax[i].imshow(im, cmap='jet')
    ax[i].axis('off')
plt.show()
```

### Pseudocolor visualization

A common operation with spectral images is to reduce the dimensionality, applying principal components analysis (PCA). We can get the first three principal components into a three bands pseudoimage, and visualize this pseudoimage.

```python
pc_img = hsi.principal_components(img)

plt.figure()
plt.title('Principal components')
plt.imshow(pc_img)
plt.axis('off')
plt.show()
```

### Classification model

Two classification models are directly available for automatic identification on hysperspectral images. One is based on support vector machines ('svm') while the other is based on deep image classification ('hsn'). Both models are available under a simple interface like bellow:

```python
n_bands = 30
n_classes = 17
model = hsi.create_model('hsn', (25, 25, n_bands), n_classes)
model.train(X, y, train_ratio=0.3, epochs=5)
y_pred = model.predict(X)
``` 

## Abraia command line

The Abraia CLI tool provides a simple way to bulk resize, convert, and optimize your images and photos for web. Enabling the conversion from different input formats to get images in the right formats to be used in the web - JPEG, WebP, or PNG -. Moreover, it supports a number of transformations that can be applied to image batches. So you can easily convert your images to be directly published on the web.

### Installation

The Abraia CLI is a Python tool which can be installed on Windows, Mac, and Linux:

```sh
python -m pip install -U abraia
```

The first time you run Abraia CLI you need to configure your API key, just write the command bellow and paste your key.

```sh
abraia configure
```

### Resize images

To compress an image you just need to specify the input and output paths for the image:

```sh
abraia convert images/birds.jpg images/birds_o.jpg
```

![Image compressed from url](https://github.com/abraia/abraia-python/raw/master/images/birds_o.jpg)

To resize and optimize and image maintaining the aspect ratio is enough to specify the `width` or the `height` of the new image:

```sh
abraia convert --width 500 images/usain-bolt.jpeg images/usaint-bolt_500.jpeg
```

![Usain Bolt resized](https://github.com/abraia/abraia-python/raw/master/images/usaint-bolt_500.jpeg)

You can also automatically change the aspect ratio specifying both `width` and `height` parameters and setting the resize `mode` (pad, crop, thumb):

```sh
abraia convert --width 333 --height 333 --mode pad images/lion.jpg images/lion_333x333.jpg
abraia convert --width 333 --height 333 images/lion.jpg images/lion_333x333.jpg
```

![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333_pad.jpg)
![Image lion smart cropped](https://github.com/abraia/abraia-python/raw/master/images/lion_333x333.jpg)

So, you can automatically resize all the images in a specific folder preserving the aspect ration of each image just specifying the target `width` or `height`:

```sh
abraia convert --width 300 [path] [dest]
```

Or, automatically pad or crop all the images contained in the folder specifying both `width` and `height`:

```sh
abraia convert --width 300 --height 300 --mode crop [path] [dest]
```

![beauty casual resized](https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_333x500.jpg)
![beauty casual smart cropped](https://github.com/abraia/abraia-python/raw/master/images/beauty-casual_500x500.jpg)

### Convert images

The JPEG image format is still the most common format to publish photos on the web. However, converting images to WebP provides a significant improvement for web publishing.

To convert images to a web format (JPEG, PNG, WebP) or between these formats you just need to change the filename extension for the destination file:

```sh
abraia convert garlic.jpg garlic.webp
```

<figure>
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.jpg" alt="garlic jpeg">
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/garlic.webp" alt="garlic webp">
</figure>

In addition, you can also convert SVG and PSD files. For instance, converting a SVG to PNG is so simple as to type the command bellow:

```sh
abraia convert bat.svg bat.png
```

<figure>
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.svg" alt="bat svg">
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/bat.png" alt="bat png">
</figure>

> The SVG vector image is rendered in a Chrome instance to provide maximum fidelity, and preserving the transparent background.

Moreover, you can easily convert a PSD file (the layered image file used in Adobe Photoshop for saving data) flattening all the visible layers with a command like bellow:

```sh
abraia convert strawberry.psd strawberry.jpg
abraia convert strawberry.psd strawberry.png
```

<figure>
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.jpg" alt="white background strawberry">
    <img width="300px" src="https://github.com/abraia/abraia-python/raw/master/images/strawberry.png" alt="transparent strawberry">
</figure>

> When the PSD file is converted to JPEG a white background is added automatically, because the JPEG format does not support transparency. Instead, using the PNG or the WebP format you can preserve the transparent background.

Or, convert a batch of Photoshop files with a simple command. Just copy your PSD files to a folder, for instance the `photoshop` folder, and convert all the files in that folder.

```sh
abraia convert photoshop
```

You can also take web from the command line just specifying and url to get the capture.

```sh
abraia convert https://abraia.me screenshot.jpg
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
