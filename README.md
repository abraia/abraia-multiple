[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia Python SDK image analysis toolbox

The Abraia Python SDK provides and easy and practical way to develop and deploy Machine Learning image applications on the edge. You can easily annotate and train your custom deep learning model with [DeepLab](https://abraia.me/deeplab/), and deploy the model with this Python SDK.

![people walking](https://github.com/abraia/abraia-multiple/raw/master/images/people-walking.gif)

Just install the Abraia Python SDK and CLI on Windows, Mac, or Linux:

```sh
python -m pip install -U abraia
```

And start working with deep learning models ready to work on your local devices.

## Load and run custom models

Annotate your images and train a state-of-the-art model for classification, detection, or segmentation using [DeepLab](https://abraia.me/deeplab/). You can directly load and run the model on the edge using the browser or this Python SDK.

### Object detection

Detect objects with a pre-trained YOLOv8 model on images, videos, or even camera streams.

```python
from abraia import detect

model_uri = f"multiple/models/yolov8n.onnx"
model = detect.load_model(model_uri)

img = detect.load_image('people-walking.png')
results = model.run(img, conf_threshold=0.5, iou_threshold=0.5)
img = detect.render_results(img, results)
detect.show_image(img)
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.png)

To run a multi-object detector on video or directly on a camera stream, you just need to use the Video class to process every frame as is done for images.

```python
from abraia import detect

model_uri = f"multiple/models/yolov8n.onnx"
model = detect.load_model(model_uri)

video = detect.Video('people-walking.mp4')
for frame in video:
    results = model.run(frame, conf_threshold=0.5, iou_threshold=0.5)
    frame = detect.render_results(frame, results)
    video.show(frame)
```

### Face recognition

Identify people on images with face recognition as shown bellow. 

```python
import os

from abraia.faces import Recognition
from abraia.utils import load_image, save_image
from abraia.draw import render_results

img = load_image('images/rolling-stones.jpg')
out = img.copy()

recognition = Recognition()
results = recognition.represent_faces(img)

index = []
for src in ['mick-jagger.jpg', 'keith-richards.jpg', 'ronnie-wood.jpg', 'charlie-watts.jpg']:
    img = load_image(f"images/{src}")
    rslt = recognition.represent_faces(img)[0]
    index.append({'name': os.path.splitext(src)[0], 'embeddings': rslt['embeddings']})

result = recognition.identify_faces(results, index)
render_results(out, results)
save_image(out, 'images/rolling-stones-identified.jpg')
```

![rolling stones identified](https://github.com/abraia/abraia-multiple/raw/master/images/rolling-stones-identified.jpg)

### License plates blurring

Automatically blur car license plates in videos with just a few lines of code.

```python
import numpy as np

from abraia import detect
from abraia import draw

model_uri = 'multiple/models/alpd-seg.onnx'
model = detect.load_model(model_uri)

src = 'images/cars.mp4'
video = detect.Video(src, output='images/blur.mp4')
for k, frame in enumerate(video):
    results = model.run(frame, approx=0.02)
    mask = np.zeros(frame.shape[:2], np.uint8)
    [draw.draw_filled_polygon(mask, result['polygon'], 255) for result in results]
    frame = draw.draw_blurred_mask(frame, mask)
    video.write(frame)
    video.show(frame)
```

![car license plate blurred](https://github.com/abraia/abraia-multiple/raw/master/images/blur.jpg)

### License plates recognition

Automatically recognize car license plates in images and video streams.

```python
from abraia import draw
from abraia.alpr import ALPR
from abraia.utils import load_image, show_image

alpr = ALPR()

img = load_image('images/car.jpg')
results = alpr.detect(img)
results = alpr.recognize(img, results)
results = [result for result in results if len(result['lines'])]
for result in results:
    result['label'] = '\n'.join([line.get('text', '') for line in result['lines']])
    del result['confidence']
frame = draw.render_results(img, results)
show_image(img)
```

![car license plate recognition](https://github.com/abraia/abraia-multiple/raw/master/images/car-plate.jpg)

## Command line interface

The Abraia CLI provides access to the Abraia Cloud Platform through the command line. It makes simple to manage your files and enables bulk image editing capabilities. It provides and easy way to resize, convert, and compress your images - JPEG, WebP, or PNG -, and get them ready to publish on the web. Moreover, you can automatically remove the background, upscale, or anonymize your images in bulk.

To compress an image you just need to specify the input and output paths for the image:

```sh
abraia convert images/birds.jpg images/birds_o.jpg
```

![Image compressed from url](https://github.com/abraia/abraia-multiple/raw/master/images/birds_o.jpg)

To resize and optimize and image maintaining the aspect ratio is enough to specify the `width` or the `height` of the new image:

```sh
abraia convert --width 500 images/usain-bolt.jpeg images/usaint-bolt_500.jpeg
```

![Usain Bolt resized](https://github.com/abraia/abraia-multiple/raw/master/images/usain-bolt_500.jpeg)

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

### Remove background

Automatically remove images background and make them transparent in bulk.

```sh
abraia editing "*.jpg" --mode removebg
```

![bolt transparent background](https://github.com/abraia/abraia-multiple/raw/master/images/usain-bolt.png)

### Upscale images

Scale up and enhance images in bulk, doubling the size and preserving quality.

```sh
abraia editing "*.jpg" --mode upscale
```

![upscaled cat](https://github.com/abraia/abraia-multiple/raw/master/images/cat-upscaled.jpg)

### Anonymize images

Anonymize images in bulk, automatically blurring faces, car license plates, and removing metadata.

```sh
abraia editing "*.jpg" --mode anonymize
````

![people and car anonymized](https://github.com/abraia/abraia-multiple/raw/master/images/people-car-anonymized.jpg)

## Hyperspectral image analysis toolbox

The Multiple extension provides seamless integration of multispectral and hyperspectral images. It has being developed by [ABRAIA](https://abraia.me/about) in the [Multiple project](https://multipleproject.eu/) to extend the Abraia SDK and Cloud Platform providing support for straightforward HyperSpectral Image (HSI) analysis and classification.

Just click on one of the available Colab's notebooks to directly start testing the multispectral capabilities:

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-analysis.ipynb) Hyperspectral image analysis

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-classification.ipynb) Hyperspectral image classification

![classification](https://github.com/abraia/abraia-multiple/raw/master/images/classification.png)

Or install the multiple extension to use the Abraia-Multiple SDK:

```sh
python -m pip install -U "abraia[multiple]"
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

Then, you will be able to directly load and save ENVI files, and their metadata.

```python
from abraia.multiple import Multiple

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
from abraia.multiple import hsi

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
