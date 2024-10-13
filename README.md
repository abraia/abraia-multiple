[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia Python SDK image analysis toolbox

The Abraia Python SDK provides and easy and practical way to develop and deploy Machine Learning image applications on the edge. You can easily annotate and train your custom deep learning model with [DeepLab](https://abraia.me/deeplab/), and deploy the model with this Python SDK.

![people walking](https://github.com/abraia/abraia-multiple/raw/master/images/people-walking.gif)

## Installation

Abraia is a Python SDK and CLI which can be installed on Windows, Mac, and Linux:

```sh
python -m pip install -U abraia
```

To use the SDK you have to configure your [Id and Key](https://abraia.me/console/) as environment variables:

```sh
export ABRAIA_ID=user_id
export ABRAIA_KEY=user_key
```

On Windows you need to use `set` instead of `export`:

```sh
set ABRAIA_ID=user_id
set ABRAIA_KEY=user_key
```

## Load and run custom models

Annotate your images and train a state-of-the-art model for classification, detection, or segmentation using [DeepLab](https://abraia.me/deeplab/). You can directly load and run the model on the edge using the browser or this Python SDK.

### Object detection

Detect objects with a pre-trained YOLOv8 model on images, videos, or even camera streams.

```python
from abraia import detect

model_uri = f"https://api.abraia.me/files/multiple/models/yolov8n.onnx"

model = detect.load_model(model_uri)

im = detect.load_image('people-walking.png').convert('RGB')
results = model.run(im, confidence=0.5, iou_threshold=0.5)
im = detect.render_results(im, results)
im.show()
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.png)

To run a multi-object detector on video or directly on a camera stream, you just need to use the Video class to process every frame as is done for images.

```python
import numpy as np
from PIL import Image
from abraia import detect


model_uri = f"https://api.abraia.me/files/multiple/models/yolov8n.onnx"

model = detect.load_model(model_uri)

video = detect.Video('people-walking.mp4')
for frame in video:
    results = model.run(frame, confidence=0.5, iou_threshold=0.5)
    frame = detect.render_results(frame, results)
    video.show(frame)
```

### Face recognition

Identify people on images with face recognition as shown bellow. 

```python
import os
import numpy as np

from abraia.draw import load_image, save_image, render_results
from abraia.faces import Recognition


img = load_image('images/rolling-stones.jpg')
out = img.copy()

recognition = Recognition()
results = recognition.represent_faces(img)

for src in ['mick-jagger.jpg', 'keith-richards.jpg', 'ronnie-wood.jpg', 'charlie-watts.jpg']:
    img = load_image(f"images/{src}")
    rslt = recognition.represent_faces(img)[0]
    sims = [recognition.compute_similarity(rslt['embeddings'], result['embeddings']) for result in results]
    idx = np.argmax(sims)
    if sims[idx] > 0.45:
        results[idx]['label'] = os.path.splitext(src)[0]
        results[idx]['confidence'] = sims[idx]
    print(src, sims[idx], sims)

render_results(out, results)
save_image('images/rolling-stones-identified.jpg', out)
```

![rolling stones identified](https://github.com/abraia/abraia-multiple/raw/master/images/rolling-stones-identified.jpg)

### Blur license plates

Automatically blur car license plates in videos with just a few lines of code.

```python
from abraia import detect
from abraia import draw

model_uri = 'https://api.abraia.me/files/multiple/models/alpd-seg.onnx'
model = detect.load_model(model_uri)

src = 'images/cars.mp4'
video = detect.Video(src, output='images/blur.mp4')
for k, frame in enumerate(video):
    results = model.run(frame)
    for result in results:
        polygon = detect.approximate_polygon(result['polygon'])
        frame = draw.draw_blurred_polygon(frame, polygon)
    video.write(frame)
    video.show(frame)
```

![car license plate blurred](https://github.com/abraia/abraia-multiple/raw/master/images/blur.jpg)

## Image analysis toolbox

Abraia provides a direct interface to load and save images. You can easily load and show the image, load the file metadata, or save the image as a new one.

```python
from abraia import Abraia

abraia = Abraia()

im = abraia.load_image('usain.jpg')
abraia.save_image('usain.png', im)
im.show()
```

![plot image](https://github.com/abraia/abraia-multiple/raw/master/images/bolt.png)

Read the image metadata and save it as a JSON file.

```python
metadata = abraia.load_metadata('usain.jpg')
abraia.save_json('usain.json', metadata)
```

    {'FileType': 'JPEG',
    'MIMEType': 'image/jpeg',
    'JFIFVersion': 1.01,
    'ResolutionUnit': 'None',
    'XResolution': 1,
    'YResolution': 1,
    'Comment': 'CREATOR: gd-jpeg v1.0 (using IJG JPEG v62), quality = 80\n',
    'ImageWidth': 640,
    'ImageHeight': 426,
    'EncodingProcess': 'Baseline DCT, Huffman coding',
    'BitsPerSample': 8,
    'ColorComponents': 3,
    'YCbCrSubSampling': 'YCbCr4:2:0 (2 2)',
    'ImageSize': '640x426',
     'Megapixels': 0.273}

### Upload and list files

Upload a local `src` file to the cloud `path` and return the list of `files` and `folders` on the specified cloud `folder`.

```python
import pandas as pd

folder = 'test/'
abraia.upload_file('images/usain-bolt.jpeg', folder)
files, folders = abraia.list_files(folder)

pd.DataFrame(files)
```

![files](https://github.com/abraia/abraia-multiple/raw/master/images/files.png)

To list the root folder just omit the folder value.

### Download and remove files

You can download or remove an stored file just specifying its `path`.

```python
path = 'test/birds.jpg'
dest = 'images/birds.jpg'
abraia.download_file(path, dest)
abraia.remove_file(path)
```

## Command line interface

The Abraia CLI provides access to the Abraia Cloud Platform through the command line. It provides a simple way to manage your files and enables the resize and conversion of different image formats. It is an easy way to compress your images for web - JPEG, WebP, or PNG -, and get then ready to publish on the web. 

To compress an image you just need to specify the input and output paths for the image:

```sh
abraia convert images/birds.jpg images/birds_o.jpg
```

![Image compressed from url](https://github.com/abraia/abraia-multiple/raw/master/images/birds_o.jpg)

To resize and optimize and image maintaining the aspect ratio is enough to specify the `width` or the `height` of the new image:

```sh
abraia convert --width 500 images/usain-bolt.jpeg images/usaint-bolt_500.jpeg
```

![Usain Bolt resized](https://github.com/abraia/abraia-multiple/raw/master/images/usaint-bolt_500.jpeg)

You can also automatically change the aspect ratio specifying both `width` and `height` parameters and setting the resize `mode` (pad, crop, thumb):

```sh
abraia convert --width 333 --height 333 --mode pad images/lion.jpg images/lion_333x333.jpg
abraia convert --width 333 --height 333 images/lion.jpg images/lion_333x333.jpg
```

![Image lion smart cropped](https://github.com/abraia/abraia-multiple/raw/master/images/lion_333x333_pad.jpg)
![Image lion smart cropped](https://github.com/abraia/abraia-multiple/raw/master/images/lion_333x333.jpg)

So, you can automatically resize all the images in a specific folder preserving the aspect ration of each image just specifying the target `width` or `height`:

```sh
abraia convert --width 300 [path] [dest]
```

Or, automatically pad or crop all the images contained in the folder specifying both `width` and `height`:

```sh
abraia convert --width 300 --height 300 --mode crop [path] [dest]
```

## Hyperspectral image analysis toolbox

The Multiple class provides seamless integration of multispectral and hyperspectral images. ou just need to click on the open in Colab button to start with one of the available Abraia-Multiple notebooks:

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-analysis.ipynb) Hyperspectral image analysis

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-classification.ipynb) Hyperspectral image classification

The Multiple extension has being developed by [ABRAIA](https://abraia.me/about) in the [Multiple project](https://multipleproject.eu/) to extend the Abraia SDK and Cloud Platform providing support for straightforward HyperSpectral Image (HSI) analysis and classification.

![classification](https://github.com/abraia/abraia-multiple/raw/master/images/classification.png)

For instance, you can directly load and save ENVI files, and their metadata.

```python
from abraia import Multiple

multiple = Multiple()

img = multiple.load_image('test.hdr')
meta = multiple.load_metadata('test.hdr')
multiple.save_image('test.hdr', img, metadata=meta)
```

### Upload and load HSI data

To start with, we may [upload some data](https://abraia.me/deeplab/) directly using the graphical interface, or using the multiple api:

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
hsi.plot_images(imgs, cmap='jet')
```

### Pseudocolor visualization

A common operation with spectral images is to reduce the dimensionality, applying principal components analysis (PCA). We can get the first three principal components into a three bands pseudoimage, and visualize this pseudoimage.

```python
pc_img = hsi.principal_components(img)
hsi.plot_image(pc_img, 'Principal components')
```

### Classification model

Two classification models are directly available for automatic identification on hysperspectral images. One is based on support vector machines ('svm') while the other is based on deep image classification ('hsn'). Both models are available under a simple interface like bellow:

```python
n_bands, n_classes = 30, 17
model = hsi.create_model('hsn', (25, 25, n_bands), n_classes)
model.train(X, y, train_ratio=0.3, epochs=5)
y_pred = model.predict(X)
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
