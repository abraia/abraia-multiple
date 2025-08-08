[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia Vision SDK

The [Abraia Vision](https://abraia.me/vision/) SDK is a Python package which provides a set of tools to develop and deploy advanced Machine Learning image applications on the edge. Moreover, with [Abraia DeepLab](https://abraia.me/deeplab/) you can easily annotate and train, your own versions of some of the best state of the art deep learning models, and get them ready to deploy with this Python SDK.

Just install the Abraia SDK and CLI on Windows, Mac, or Linux:

```sh
python -m pip install -U abraia
```

And start using deep learning models ready to work on your local devices.

## Deep learning custom models and applications

Consult your problem or directly try to annotate your images and train a state-of-the-art model for classification, detection, or segmentation using [DeepLab](https://abraia.me/deeplab/). You can directly load and run the model on the edge using the browser or this Python SDK.

### Object detection and tracking

Identify and track multiple objects with a custom detection model on videos and camera streams, enabling real-time counting applications. You just need to use
the Video class to process every frame as is done for images, and use the tracker to follow each object through 
every frame.

```python
from abraia.inference import Model, Tracker
from abraia.utils import Video, render_results

model = Model("multiple/models/yolov8n.onnx")

video = Video('images/people-walking.mp4')
tracker = Tracker(frame_rate=video.frame_rate)
for frame in video:
    results = model.run(frame, conf_threshold=0.5, iou_threshold=0.5)
    results = tracker.update(results)
    frame = render_results(frame, results)
    video.show(frame)
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.jpg)

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

results = recognition.identify_faces(results, index)
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
    del result['score']
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
    result['score'] = score
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

logits = [100 * cosine_similarity(image_embeddings[0], features) for features in text_embeddings]
for text, p in zip(texts, softmax(logits)):
    print(f"Probability that the image is '{text}': {p:.3f}")
```

### Blur license plate

Anonymize images automatically bluring car license plates.

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
image_index = [{'embeddings': clip_model.get_image_embeddings([load_image(image_path)])[0]} for image_path in tqdm(image_paths)]

text_query = "a man or a woman"
vector = clip_model.get_text_embeddings([text_query])[0]

index, scores = search_vector(vector, image_index)
print(f"Similarity score is {scores[index]} for image {image_paths[index]}")
```

## Hyperspectral image analysis toolbox

The Multiple extension provides seamless integration of multispectral and hyperspectral images, providing support for straightforward HyperSpectral Image (HSI) analysis and classification.

Just click on one of the available Colab's notebooks to directly start testing the multispectral capabilities:

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-analysis.ipynb) Hyperspectral image analysis

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral-classification.ipynb) Hyperspectral image classification

![classification](https://github.com/abraia/abraia-multiple/raw/master/images/classification.png)

To install the multiple extension use the command bellow:

```sh
python -m pip install -U "abraia[multiple]"
```

To use the SDK you have to configure your [Id and Key](https://abraia.me/editor/) as environment variables:

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
