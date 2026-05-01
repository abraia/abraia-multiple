[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)

# Abraia Vision SDK

The [Abraia Vision](https://abraia.me/vision/) SDK helps developers create, customize, and deploy edge-ready vision applications. It unifies image processing, model training, and inference so you can transform visual data into production-ready solutions, including real-time video analysis and object tracking.

Install the Abraia SDK and CLI on Windows, Mac, or Linux:

```sh
python -m pip install -U abraia
```

With [Abraia DeepLab](https://abraia.me/deeplab/), you can annotate images, train custom classification, detection, and segmentation models, and export them for use in this Python SDK.

### People monitoring

Abraia SDK provides a set of tools to monitor people flow and waiting time in public spaces or commercial areas. You can easily implement queue monitoring or flow counting applications using the specialized tools available in the `abraia.inference.tools` module.

```python
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionTimer
from abraia.utils import Video, render_results, render_counter, render_region

model = Model("multiple/models/yolov8n.onnx")
video = Video('images/people-walking.mp4')
tracker = Tracker(frame_rate=video.frame_rate)
line_counter = LineCounter([(0, 650), (1920, 650)])
region_timer = RegionTimer([(10, 600), (1690, 600), (1690, 700), (10, 700)])

for k, frame in enumerate(video):
    results = model.run(frame)
    results = [result for result in results if result['label'] == 'person']
    results = tracker.update(results)
    in_count, out_count = line_counter.update(results)
    in_objects, out_objects = region_timer.update(results, k / video.frame_rate)
    frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    frame = render_region(frame, region_timer.region, f"Count: {len(in_objects)}")
    frame = render_results(frame, in_objects)
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

index = []
for src in ['mick-jagger.jpg', 'keith-richards.jpg', 'ronnie-wood.jpg', 'charlie-watts.jpg']:
    img = load_image(f"images/{src}")
    rslt = recognition.identify_faces(img)[0]
    index.append({'name': os.path.splitext(src)[0], 'vector': rslt['vector']})

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

### Gender Age estimation

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
image_index = [{'vector': clip_model.get_image_embeddings([load_image(image_path)])[0]} for image_path in tqdm(image_paths)]

text_query = "full body person"
vector = clip_model.get_text_embeddings([text_query])[0]

idxs, scores = search_vector(vector, image_index)
print(f"Similarity score is {scores[0]} for image {image_paths[idxs[0]]}")
```

## Hyperspectral imaging

The Multiple extension provides seamless integration of multispectral and hyperspectral images, providing support for straightforward HyperSpectral Image (HSI) analysis and classification.

Just click on the available Colab's notebooks to directly start testing the multispectral capabilities:

* [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral.ipynb) Hyperspectral image analysis and classification

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

### Basic HSI visualization

Load an ENVI hyperspectral file, inspect its metadata, and save the image back with the same information.

```python
from abraia.multiple import Multiple

multiple = Multiple()

multiple.upload_file('test.hdr')
img = multiple.load_image('test.hdr')
meta = multiple.load_metadata('test.hdr')
multiple.save_image('test.hdr', img, metadata=meta)
```

Because hyperspectral data is multi-band, it cannot be displayed directly as a normal RGB image. You can extract a few individual bands and plot them as grayscale images instead.

```python
from abraia.multiple import hsi

imgs, indexes = hsi.random(img)
hsi.plot_images(imgs, cmap='jet')
```

Another option is to reduce dimensionality using PCA and create a 3-channel pseudo-RGB image from the first three principal components.

```python
pc_img = hsi.principal_components(img)
hsi.plot_image(pc_img, 'Principal components')
```

## License

This software is licensed under the MIT License. [View the license](LICENSE).
