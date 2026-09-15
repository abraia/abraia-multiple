[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Abraia Vision SDK

The **Abraia Vision SDK** is a high-performance, edge-ready Python library and toolkit for computer vision, image processing, model training, and advanced inference. It unifies state-of-the-art vision models (such as YOLO, SAM, CLIP, and custom recognition pipelines) into a seamless API for production-ready applications, real-time video analysis, object tracking, hyperspectral imaging, and edge hardware deployment.

---

## 📚 Table of Contents

- [Installation](#-installation)
- [Core Modules & Features](#-core-modules--features)
  - [1. Inference & Computer Vision](#1-inference--computer-vision)
  - [2. Image Editing & Enhancement](#2-image-editing--enhancement)
  - [3. Multispectral & Hyperspectral Imaging (HSI)](#3-multispectral--hyperspectral-imaging-hsi)
  - [4. Edge AI & Hardware Acceleration (Hailo)](#4-edge-ai--hardware-acceleration-hailo)
  - [5. Training & Dataset Operations](#5-training--dataset-operations)
  - [6. Utilities & Video Processing](#6-utilities--video-processing)
- [Examples & Usage Guides](#-examples--usage-guides)
  - [People Monitoring & Tracking](#people-monitoring--tracking)
  - [Face Recognition](#face-recognition)
  - [License Plate Recognition (ALPR)](#license-plate-recognition-alpr)
  - [Semantic Search with CLIP](#semantic-search-with-clip)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 📦 Installation

Install the Abraia SDK from PyPI:

```sh
pip install -U abraia
```

For training and development run the installation with optional extras (`dev`, `multiple`):

```sh
pip install -U abraia[dev,multiple]
```

For Vision Studio with ENVI support, install the Studio extra:

```sh
pip install -U abraia[studio]
```

For supervised HSI analysis or GIS helpers, add the
corresponding optional extras:

```sh
pip install -U abraia[multiple,analysis,gis]
```

---

## 🚀 Core Modules & Features

### 1. Inference & Computer Vision (`abraia.inference`)
- **Object Detection**: Fast ONNX/YOLO-based object detection (`abraia.inference.Model`).
- **Segmentation (SAM)**: Segment Anything Model integration for precise image masking (`abraia.inference.Sam`).
- **Object Tracking & People Flow**: Advanced multi-object tracking (`Tracker`), line crossing counters (`LineCounter`), and region duration timers (`RegionTimer`).
- **Face Recognition**: Identify and match faces in images and streams (`FaceRecognizer`).
- **License Plate Recognition (ALPR)**: Automatic license plate detection and text recognition (`PlateRecognizer`).
- **OCR**: Extract text from images (`Ocr`).
- **Semantic Search (CLIP)**: Vector embeddings and similarity search for text-to-image and image-to-image retrieval (`Clip`).

### 2. Image Editing & Enhancement (`abraia.editing`)
- **Upscaling**: Super-resolution image enhancement (`upscale`).
- **Smart Cropping**: Intelligent content-aware cropping (`smartcrop`).
- **Background Removal**: Foreground segmentation and background removal (`removebg`).
- **Inpainting**: Image restoration and object removal (`inpaint`).

### Object Processing Pipelines

Object detection, tracking, and counting workflows can be configured in a
JSON file and run from Python:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("pipeline.json").run()
```

The first pipeline format supports a source, detector, tracker, line counter,
region filter or timer, and display output. Stages run in the order listed.
The detector can be an ONNX model (the default for existing configurations) or
one of the built-in face and license-plate detectors:

```json
{
  "version": 1,
  "source": {"type": "video", "src": "people.mp4"},
  "model": {
    "task": "detection",
    "kind": "onnx",
    "uri": "multiple/models/yolov8n.onnx",
    "labels": ["person"]
  },
  "stages": [
    {"type": "tracker"},
    {"type": "line_counter", "line": [[0, 650], [1920, 650]]}
  ],
  "display": {"show": true, "dest": "output.avi"}
}
```

Built-in detectors do not require a model URI. For example:

```json
"model": {
  "task": "detection",
  "kind": "license_plate",
  "params": {"threshold": 0.85, "iou_threshold": 0.15}
}
```

The Studio model selector also includes the bundled object-detection and
instance-segmentation presets. Their model URIs are
`multiple/models/yolov8n.onnx` and `multiple/models/yolov8n-seg.onnx`.
OCR recognition is available as a built-in model:

```json
"model": {
  "task": "recognition",
  "kind": "ocr",
  "params": {"drop_score": 0.5}
}
```

Face recognition uses a JSON index containing `name` and `vector` entries:

```json
"model": {
  "task": "recognition",
  "kind": "face",
  "params": {"index": "faces.json", "threshold": 0.45}
}
```

### 3. Multispectral & Hyperspectral Imaging (`multiple`)
- Specialized tools for hyperspectral and multispectral image analysis, cube processing, spectral indices, radiometric calibration, scene manifests, and spectral signature extraction (`multiple.analysis`). Remote datasets support TIFF cubes, ENVI header/data pairs (`.hdr` with `.raw`, `.img`, or a declared companion file), and IMEC snapshot-mosaic scenes (`.raw` plus their calibration `.xml`). Studio can upload a folder containing the raw scenes and shared calibration file.
- The public `multiple` API is grouped into visualization, local I/O, shared band contracts, metadata, manifests, remote datasets and clients, radiometry, analysis, and registration modules. Analysis and GIS integrations expose optional dependency errors only when used.

### 4. Edge AI & Hardware Acceleration (`abraia.inference.hailo`)
- Optimized runtime support and toolboxes for Hailo NPU hardware acceleration (`abraia.inference.hailo`).

### 5. Training & Dataset Operations (`abraia.training`)
- Tools for training custom classification (`classify`) and detection (`detect`) models, along with dataset preprocessing utilities (`dataset`, `ops`).

### 6. Runtime & Video Processing (`abraia.runtime`)
- Robust video frame iteration and manipulation (`Video`).
- Annotation and rendering tools (`render_results`, `render_counter`, `render_region`).
- Compression and sketch generation utilities.

---

## 💡 Examples & Usage Guides

### People Monitoring & Tracking

Monitor people flow, count crossings, and track dwell times in public spaces or commercial areas:

```python
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionTimer
from abraia.runtime import Video
from abraia.utils import render_results, render_counter, render_region

model = Model("multiple/models/yolov8n.onnx")
video = Video('people-walking.mp4')
tracker = Tracker(frame_rate=video.frame_rate)
line_counter = LineCounter([(0, 650), (1920, 650)])
region_timer = RegionTimer([(10, 600), (1690, 600), (1690, 700), (10, 700)])

for k, frame in enumerate(video):
    results = model.run(frame, labels=['person'])
    results = tracker.update(results)
    in_count, out_count = line_counter.update(results)
    in_objects, out_objects = region_timer.update(results, k / video.frame_rate)
    frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    frame = render_region(frame, region_timer.region, f"Count: {len(in_objects)}")
    frame = render_results(frame, in_objects)
    video.show(frame)
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.jpg)

### Face Recognition

Identify and recognize people in images:

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

### License Plate Recognition (ALPR)

Automatically detect and recognize car license plates in images and video streams:

```python
from abraia.inference import PlateRecognizer
from abraia.utils import load_image, show_image, render_results

alpr = PlateRecognizer()

img = load_image('images/car.jpg')
results = alpr.recognize(img)
frame = render_results(img, results)
show_image(img)
```

![car license plate recognition](https://github.com/abraia/abraia-multiple/raw/master/images/car-plate.jpg)

### Semantic Search with CLIP

Search images using natural language text queries via CLIP embeddings:

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

---

## 🍓 Real-Time Edge Object Counter on Raspberry Pi with Hailo NPU

Deploy high-performance real-time object detection and counting on a Raspberry Pi equipped with a Hailo AI expansion board (such as Hailo-8 or Hailo-8L). This pipeline combines hardware-accelerated model inference (`abraia.inference.hailo`), multi-object tracking (`abraia.inference.Tracker`), line crossing counters (`LineCounter`), and region timers (`RegionTimer`), integrated with the asynchronous video processing pipeline (`VideoInput` & `VideoDisplay`).

### Implementation Guide

Create a script (e.g., `edge_counter.py`) ready for deployment on your Raspberry Pi:

```python
import threading
from abraia.inference.hailo.toolbox import ModelInference
from abraia.inference import Tracker
from abraia.inference.tools import LineCounter, RegionTimer
from abraia.runtime import VideoInput, VideoDisplay
from abraia.utils import render_results, render_counter, render_region
from abraia.inference.hailo.detect import run_inference_pipeline

# 1. Initialize threaded video input (e.g., Raspberry Pi Camera or RTSP stream)
stop_event = threading.Event()
input_data = VideoInput(input_src=0, resolution=(1920, 1080), stop_event=stop_event)
visualizer = VideoDisplay(source_fps=input_data.source_fps, stop_event=stop_event)

# 2. Load Hailo compiled model (.hef) optimized for edge NPU
model_inference = ModelInference(
    hef_path="yolov8n.hef",
    task="detect",
    labels=["person", "car"],
    batch_size=1,
    score_threshold=0.3
)

# 3. Setup Tracker & Analytics Tools (Line Counter & Region Timer)
tracker = Tracker(frame_rate=input_data.source_fps or 30.0)
line_counter = LineCounter([(100, 540), (1820, 540)])     # Crossing boundary line
region_timer = RegionTimer([(300, 200), (1620, 200), (1620, 900), (300, 900)]) # Zone of interest

# 4. Custom Inference & Analytics Result Handler
def edge_processing_handler(frame, detections, tracker=None, tracklet_history=None):
    if tracker:
        detections = tracker.update(detections)
    
    # Update line crossing and region analytics
    in_count, out_count = line_counter.update(detections)
    in_objects, out_objects = region_timer.update(detections, 1.0 / (input_data.source_fps or 30.0))
    
    # Render real-time visual overlays
    frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    frame = render_region(frame, region_timer.region, f"Zone Count: {len(in_objects)}")
    return render_results(frame, detections)

# 5. Run High-Performance Edge Pipeline
try:
    run_inference_pipeline(
        model_inference=model_inference,
        input_data=input_data,
        visualizer=visualizer,
        tracker=tracker
    )
finally:
    stop_event.set()
```

### Deployment on Raspberry Pi

Execute the script directly on the Raspberry Pi:

```sh
python3 edge_counter.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
