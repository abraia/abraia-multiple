[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia SDK and CLI

The Abraia SDK and CLI is a Python package which provides a set of tools to develop and deploy advanced Machine Learning image applications on the edge. Moreover, with [Abraia DeepLab](https://abraia.me/deeplab/) you can easily annotate and train, your own versions of some of the best state of the art deep learning models, and get them ready to deploy with this Python SDK.

![people walking](https://github.com/abraia/abraia-multiple/raw/master/images/people-walking.gif)

Just install the Abraia SDK and CLI on Windows, Mac, or Linux:

```sh
python -m pip install -U abraia
```

And start working with deep learning models ready to work on your local devices.

## Load and run custom models

Annotate your images and train a state-of-the-art model for classification, detection, or segmentation using [DeepLab](https://abraia.me/deeplab/). You can directly load and run the model on the edge using the browser or this Python SDK.

### Object detection

Detect objects with a pre-trained YOLOv8 model on images, videos, or even camera streams.

```python
from abraia.inference import Model
from abraia.utils import load_image, show_image, render_results

model = Model("multiple/models/yolov8n.onnx")

img = load_image('images/people-walking.png')
results = model.run(img, conf_threshold=0.5, iou_threshold=0.5)
img = render_results(img, results)
show_image(img)
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.png)

To run a multi-object detector on video or directly on a camera stream, you just need to use the Video class to process every frame as is done for images.

```python
from abraia.inference import Model
from abraia.utils import Video, render_results

model = Model("multiple/models/yolov8n.onnx")

video = Video('images/people-walking.mp4')
for frame in video:
    results = model.run(frame, conf_threshold=0.5, iou_threshold=0.5)
    frame = render_results(frame, results)
    video.show(frame)
```

### Face recognition

Identify people on images with face recognition as shown bellow. 

```python
import os

from abraia.inference import FaceRecognizer
from abraia.utils import load_image, save_image, render_results

img = load_image('images/rolling-stones.jpg')
out = img.copy()

recognition = FaceRecognizer()
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

### License plates recognition

Automatically recognize car license plates in images and video streams.

```python
from abraia.inference import PlateRecognizer
from abraia.utils import load_image, show_image, render_results

alpr = PlateRecognizer()

img = load_image('images/car.jpg')
results = alpr.detect(img)
results = alpr.recognize(img, results)
results = [result for result in results if len(result['lines'])]
for result in results:
    result['label'] = '\n'.join([line.get('text', '') for line in result['lines']])
    del result['confidence']
frame = render_results(img, results)
show_image(img)
```

![car license plate recognition](https://github.com/abraia/abraia-multiple/raw/master/images/car-plate.jpg)

### Gender Age model

Model to predict gender and age. It can be useful to anonymize minors faces.

```python
from abraia.inference import FaceRecognizer, FaceAttribute
from abraia.utils import load_image, show_image, render_results

recognition = FaceRecognizer()
attribute = FaceAttribute()

img = load_image('images/image.jpg')
results = recognition.detect_faces(img)
faces = recognition.extract_faces(img, results)
for face, result in zip(faces, results):
    gender, age, score = attribute.predict(face)
    result['label'] = f"{gender} {age}"
    result['confidence'] = score
img = render_results(img, results)
show_image(img)
```

### Zero-shot classification

Use "clip" model for zero-shot classification.

```python
from abraia.utils import load_image
from abraia.inference.clip import Clip
from abraia.inference.ops import cosine_similarity, softmax

image = load_image("images/image.jpg")
texts = ["a photo of a man", "a photo of a woman"]

clip_model = Clip()
image_embeddings = clip_model.get_image_embeddings([image])[0]
text_embeddings = clip_model.get_text_embeddings(texts)

# To use the embeddings for zero-shot classification, you can use these two
# functions. Here we run on a single image, but any number is supported.
logits = [100 * cosine_similarity(image_embeddings[0], features) for features in text_embeddings]
for text, p in zip(texts, softmax(logits)):
    print(f"Probability that the image is '{text}': {p:.3f}")
```

### Blur license plate

Automatically blur car license plates.

```python
from abraia.utils import load_image, save_image
from abraia.inference import PlateDetector
from abraia.editing import build_mask
from abraia.utils.draw import draw_blurred_mask

src = 'images/car.jpg'
img = load_image(src)

detector = PlateDetector()
plates = detector.detect(img)
mask = build_mask(img, plates, [])
out = draw_blurred_mask(img, mask)

save_image(out, 'blur-car.jpg')
```

![blur car license plate](https://github.com/abraia/abraia-multiple/raw/master/images/blur-car.jpg)

### Semantic search

Search on images with embeddings.

```python
from tqdm import tqdm
from glob import glob
from abraia.utils import load_image
from abraia.inference.clip import Clip
from abraia.inference.ops import search_vector

clip_model = Clip()

image_paths = glob('images/*.jpg')
image_embeddings = [clip_model.get_image_embeddings([load_image(image_path)])[0] for image_path in tqdm(image_paths)]

text_query = "a man or a woman"
features = clip_model.get_text_embeddings([text_query])[0]

index, scores = search_vector(image_embeddings, features)
print(f"Similarity score is {scores[index]} for image {image_paths[index]}")
```

## Command line interface

The Abraia CLI provides access to the Abraia Cloud Platform through the command line. It makes simple to manage your files and enables bulk image editing capabilities. It provides and easy way to resize, convert, and compress your images - JPEG, WebP, or PNG -, and get them ready to publish on the web. Moreover, you can automatically remove the background, upscale, or anonymize your images in bulk.

### Remove unwanted objects

Remove unwanted objects in images and photos locally. Just click on the object to automatically select and delete it from the image. Finally, press "s" to save the output image.

```sh
abraia editing clean dog.jpg
```

![inpaint output](https://github.com/abraia/abraia-multiple/raw/master/images/inpaint-output.jpg)

### Remove background

Automatically remove images background and make them transparent in bulk.

```sh
abraia editing removebg "*.jpg"
```

![removebg output](https://github.com/abraia/abraia-multiple/raw/master/images/removebg-output.png)

### Blur background

Automatically blur the images background to focus attentioin on the main objects.

```sh
abraia editing blur "*.jpg"
```

![blur background output](https://github.com/abraia/abraia-multiple/raw/master/images/blur-background.jpg)

### Upscale images

Scale up and enhance images in bulk, doubling the size and preserving quality.

```sh
abraia editing upscale "*.jpg"
```

![upscaled cat](https://github.com/abraia/abraia-multiple/raw/master/images/cat-upscaled.jpg)

### Anonymize images

Automatically blur car license plates and faces to anonymize images in bulk.

```sh
abraia editing anonymize "*.jpg"
````

![people and car anonymized](https://github.com/abraia/abraia-multiple/raw/master/images/people-car-anonymized.jpg)

### Convert images

Compress images in bulk specifying the input glob pattern or folder:

```sh
abraia editing convert "images/bird*.jpg"
```

Automatically change the aspect ratio specifying both `width` and `height` parameters and setting the resize `mode` (pad, crop, thumb). Or simply resize images maintaining the aspect ratio just specifying the `width` or the `height` of the new image:

```sh
abraia editing convert images/birds.jpg --width 375 --height 375 --mode pad 
abraia editing convert images/birds.jpg --width 375 --height 375
abraia editing convert images/birds.jpg --width 750
```

![image padded](https://github.com/abraia/abraia-multiple/raw/master/images/birds_padded.jpg)
![image smart cropped](https://github.com/abraia/abraia-multiple/raw/master/images/birds_cropped.jpg)

So, you can automatically resize all the images in a specific folder preserving the aspect ration of each image just specifying the target `width` or `height`:

```sh
abraia editing convert [src] --width 300 
```

Or, automatically pad or crop all the images contained in the folder specifying both `width` and `height`:

```sh
abraia editing convert [src] --width 300 --height 300 --mode crop
```

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
