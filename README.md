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

### Hyperspectral imaging

The `abraia.multiple` module simplifies working with multispectral and hyperspectral images, offering HSI analysis and classification workflows.

Hyperspectral data contains many spectral bands, so it cannot be shown directly as a standard RGB image. Instead, extract a few bands and plot them as grayscale images, or apply PCA to generate a 3-channel pseudo-RGB image from the first three principal components.

Use the available Colab notebook to start experimenting with the multispectral tools:

![classification](https://github.com/abraia/abraia-multiple/raw/master/images/classification.png)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abraia/abraia-multiple/blob/master/notebooks/hyperspectral.ipynb) Hyperspectral image analysis and classification

## License

This software is licensed under the MIT License. [View the license](LICENSE).
