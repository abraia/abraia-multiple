[![Build Status](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml/badge.svg)](https://github.com/abraia/abraia-multiple/actions/workflows/build.yml)
[![Python Package](https://img.shields.io/pypi/v/abraia.svg)](https://pypi.org/project/abraia/)
![Package Downloads](https://img.shields.io/pypi/dm/abraia)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Abraia Vision SDK

The **Abraia Vision SDK** is a high-performance, edge-ready Python library and toolkit for computer vision, image processing, model training, and advanced inference. It unifies state-of-the-art vision models (such as YOLO, SAM, CLIP, and custom recognition pipelines) into a seamless API for production-ready applications, real-time video analysis, object tracking, and edge hardware deployment.

---

## 📚 Table of Contents

- [Installation](#-installation)
- [Command-Line Interface](#-command-line-interface)
- [Core Modules & Features](#-core-modules--features)
  - [1. Inference & Computer Vision](#1-inference--computer-vision)
  - [2. Image Editing & Enhancement](#2-image-editing--enhancement)
  - [Object Processing Pipelines](#object-processing-pipelines)
  - [3. Edge AI & Hardware Acceleration (Hailo)](#3-edge-ai--hardware-acceleration-hailo)
  - [4. Training & Dataset Operations](#4-training--dataset-operations)
  - [5. Runtime & Video Processing](#5-runtime--video-processing)
- [Examples & Usage Guides](#-examples--usage-guides)
  - [Pipeline-Based Monitoring, Face, and Plate Workflows](#pipeline-based-monitoring-face-and-plate-workflows)
  - [Semantic Search with CLIP](#semantic-search-with-clip)
  - [Real-Time Edge Object Counter](#real-time-edge-object-counter-on-raspberry-pi-with-hailo-npu)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 📦 Installation

Install the Abraia SDK from PyPI:

```sh
pip install -U abraia
```

For training and development, install the optional development extras:

```sh
pip install -U abraia[dev]
```

To export Ultralytics detection or segmentation models to Hailo HEF, install
the Hailo extra:

```sh
pip install -U abraia[hailo]
```

On a Linux x86_64 host, the compile command automatically downloads and
installs the matching DFC wheel from the global `multiple` namespace when the
required `hailo_sdk_client` package is missing or belongs to another compiler
generation. Hailo-8/8L use DFC 3.33.1; Hailo-10H/15H/15L use DFC 5.2.0.
Compilation on other hosts still requires a compatible external environment.

Hailo compilation runs on Linux x86_64. After preparing a dataset, compile a
trained checkpoint with:

```sh
abraia compile PROJECT --checkpoint path/to/best.pt --device hailo8l \
  --calibration-data PROJECT/data.yaml
```

The command uploads the HEF and the complete Ultralytics Hailo sidecar bundle
to the project. Hailo-8/8L use DFC 3.33.1, while Hailo-10H/15H/15L use DFC
5.2.0; select the target matching the deployment accelerator.

Grounding DINO requires the optional tokenizer dependency:

```sh
pip install -U abraia[grounding-dino]
```

Dataset curation with FastDup is optional:

```sh
pip install -U abraia[curation]
```

---

## 🖥️ Command-Line Interface

Installing the package also installs the `abraia` command. Use
`abraia --help` or `abraia COMMAND --help` for the complete option list.

### Authentication and file management

Configure the SDK with the Abraia API key. The command only prompts for the
API key and stores it for subsequent commands:

```sh
abraia configure
```

Manage remote files and inspect available datasets:

```sh
abraia files list [FOLDER]
abraia files upload LOCAL_PATH [REMOTE_FOLDER]
abraia files download REMOTE_PATH [LOCAL_FOLDER]
abraia files remove REMOTE_PATH
abraia files metadata [--remove] REMOTE_PATH
abraia list
```

`files upload` accepts a file or directory. Removing files asks for
confirmation before making changes.

### Dataset creation and annotation

Create or update a dataset from uploaded files, optionally adding images from
a search query and preprocessing them:

```sh
abraia create PROJECT [QUERY] [--anonymize] [--upscale PIXELS]
```

`--anonymize` blurs faces and license plates, while `--upscale` enlarges
images below the selected size threshold. Annotate a dataset with
Grounding DINO using a text label, optionally generating segmentation masks:

```sh
abraia annotate PROJECT LABEL [--segment]
```

### Training and Hailo compilation

Prepare and train a model on a dataset, then compile a trained Ultralytics
checkpoint to a Hailo HEF bundle:

```sh
abraia train PROJECT [EPOCHS]

abraia compile PROJECT --checkpoint path/to/best.pt \
  --device hailo8l --calibration-data PROJECT/data.yaml
```

Compilation supports `hailo8`, `hailo8l`, `hailo10h`, `hailo15h`, and
`hailo15l`. Optional compilation controls include `--fraction`, `--imgsz`,
`--conf`, `--iou`, and `--version`. On Linux x86_64, the matching Hailo DFC
wheel is installed automatically when it is required and available.

### Dataset curation and pruning

Analyze an image dataset for duplicate images, outliers, blurry images, and
invalid files with FastDup:

```sh
abraia curate PROJECT --work-dir ./fastdup-work
```

The command is a dry run by default and prints a JSON report. Apply duplicate,
blur, and invalid-image recommendations with `--apply`; include outlier
removals only after review with `--apply-outliers`. Use
`--duplicate-threshold`, `--blur-threshold`, or `--blur-percentile` to adjust
the analysis.

### Demos and inference

Run a built-in, pipeline-backed demo or a trained project. Built-in object
demos include `detect`, `segment`, `pose`, `people`, `queue`, `escalator`,
`tomato`, `apple`, `strawberry`, `grapes`, and `plates`; `faces` selects the
face-tracking demo:

```sh
abraia run demo DEMO_NAME [SOURCE] [--accelerator auto]
abraia run demo faces [SOURCE] [--accelerator auto]
abraia run PROJECT [CLASSES] [SOURCE] [--accelerator auto]
```

The `--accelerator` option applies to demos and can be `auto`, `onnx`, `cpu`,
`gpu`, or `hailo`. Custom trained-project inference currently uses the
standard ONNX model path. Use the special `search` class to search a project
with a text query:

```sh
abraia run PROJECT search "person with a red shirt"
```

---

## 🚀 Core Modules & Features

### 1. Inference & Computer Vision (`abraia.inference`)
- **Object Detection**: Fast ONNX/YOLO-based object detection (`abraia.inference.Model`).
- **Open-Vocabulary Detection**: Grounding DINO ONNX inference with text prompts (`abraia.inference.GroundingDINOModel`).
- **Segmentation (SAM)**: Segment Anything Model integration for precise image masking (`abraia.inference.SAM`).
- **Object Tracking & People Flow**: Advanced multi-object tracking (`Tracker`), line crossing counters (`LineCounter`), and region duration timers (`RegionTimer`).
- **Face Recognition**: Identify and match faces in images and streams (`FaceRecognizer`).
- **License Plate Recognition (ALPR)**: Automatic license plate detection and text recognition (`PlateRecognizer`).
- **OCR**: Extract text from images (`TextSystem`).
- **Semantic Search (CLIP)**: Vector embeddings and similarity search for text-to-image and image-to-image retrieval (`Clip`).

### 2. Image Editing & Enhancement (`abraia.editing`)
- **Upscaling**: Super-resolution image enhancement (`upscale_image`).
- **Smart Cropping**: Intelligent content-aware cropping (`smartcrop_image`).
- **Background Removal**: Foreground segmentation and background removal (`remove_background`).
- **Inpainting**: Image restoration and object removal (`inpaint_image`).

### Object Processing Pipelines

Object detection, tracking, and counting workflows can be configured in a
JSON file and run from Python:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("pipeline.json").run()
```

The version-one pipeline format supports a source, model, tracker, line
counter, region filter or timer, and display output. Stages run in the order
listed. The model can be an ONNX model, a Hailo model, or one of the built-in
face and license-plate detectors:

```json
{
  "version": 1,
  "source": {"type": "video", "src": "people.mp4"},
  "model": {
    "task": "detection",
    "kind": "yolov8",
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

For model-backed entries, `kind` identifies the model architecture or family
(`yolov8`, `yolov5`, or `yolo11`), while `task` identifies the operation
(`detection`, `segmentation`, `pose`, or `classification`). The runtime selects
ONNX or Hailo from the model URI (`.onnx` or `.hef`); Hailo is available for
the supported detection, segmentation, and pose targets.

Built-in detectors do not require a model URI. For example:

```json
"model": {
  "task": "detection",
  "kind": "license_plate",
  "params": {"threshold": 0.85, "iou_threshold": 0.15}
}
```

The runtime also includes Abraia-managed object-detection,
instance-segmentation, pose-estimation, and classification presets. Each task
has small (`n`), medium (`m`), and large (`l`) model URIs, for example
`multiple/models/yolov8n.onnx`, `multiple/models/yolov8m.onnx`, and
`multiple/models/yolov8l.onnx` for object detection.
Pipeline JSON can select a size without spelling out the URI:

```json
"model": {
  "task": "pose",
  "kind": "yolov8",
  "size": "small"
}
```

Grounding DINO detects labels supplied at inference time. The Abraia-managed
tiny ONNX model uses a fixed 800x800 input:

```json
{
  "task": "detection",
  "kind": "grounding_dino",
  "uri": "multiple/models/grounding_dino_tiny.onnx",
  "labels": ["person", "red car"],
  "conf_threshold": 0.35,
  "text_threshold": 0.25
}
```

The equivalent Python API accepts either `labels=[...]` or a free-form
`prompt="person. red car."`.

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

Hailo models use the asynchronous producer/consumer runtime internally. The
pipeline keeps completed frames ordered before applying tracking and counting
stages:

```json
"model": {
  "task": "detection",
  "kind": "yolov8",
  "uri": "multiple/models/yolov8n_hailo8.hef",
  "conf_threshold": 0.25,
  "params": {"batch_size": 1}
}
```

The Hailo platform runtime must be installed and a compatible device must be
available. The model URI may be a local `.hef` file, a native Ultralytics Hailo
export directory, or an uploaded Abraia `.hef` path. Native bundle metadata
supplies the task and class labels automatically; explicit `task` or `labels`
values still override it. Uploaded assets under `multiple/models/` are fetched
from the managed model cache. Demo pipelines derive the
architecture-specific HEF URI from the canonical ONNX URI; explicit local
paths remain supported. Hailo no longer resolves external Model Zoo downloads
or bare logical model names.

### 3. Edge AI & Hardware Acceleration (`abraia.inference.hailo`)
- Optimized runtime support and toolboxes for Hailo NPU hardware acceleration (`abraia.inference.hailo`).

### 4. Training & Dataset Operations (`abraia.training`)
- Tools for training custom classification, detection, and segmentation models, along with dataset preprocessing utilities (`dataset`, `ops`).
  Training supports small, medium, and large model sizes; detection and
  segmentation use YOLOv8n/m/l, while classification uses ResNet18/50/101.
  Ultralytics detection and segmentation models can be exported to Hailo HEF
  through `ModelTrainer.compile()` or the `abraia compile` command.
- Dataset curation and pruning are available through the optional FastDup
  integration documented in the CLI section above.

### 5. Runtime & Video Processing (`abraia.runtime`)
- Robust video frame iteration and manipulation (`Video`).
- Annotation and rendering tools (`render_results`, `render_counter`, `render_region`).
- Compression and sketch generation utilities.

---

## 💡 Examples & Usage Guides

### Pipeline-Based Monitoring, Face, and Plate Workflows

Object tracking, people flow, face recognition, and license-plate recognition
are configured as pipeline model and stage options in the
[pipeline section above](#object-processing-pipelines). Save the JSON
configuration as `pipeline.json` and run it with:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("pipeline.json").run()
```

#### People monitoring

For the built-in people-monitoring pipeline, the CLI provides the same
pipeline-backed workflow:

```sh
abraia run demo people people-walking.mp4 --accelerator auto
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.jpg)

#### Face recognition

Use `kind: "face"` in the pipeline model to select the built-in face
recognition workflow. This keeps the source, model, index, stages, display,
and accelerator selection in one configuration.

![rolling stones identified](https://github.com/abraia/abraia-multiple/raw/master/images/rolling-stones-identified.jpg)

#### License-plate recognition

Use `kind: "license_plate"` in the pipeline model to select the built-in
license-plate recognition workflow.

![car license plate recognition](https://github.com/abraia/abraia-multiple/raw/master/images/car-plate.jpg)

### Semantic Search with CLIP

Search images using natural language text queries via CLIP embeddings:

```python
from tqdm import tqdm
from glob import glob
from abraia.utils import load_image
from abraia.inference.models.clip import Clip
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

Deploy high-performance real-time object detection and counting on a Raspberry Pi equipped with a Hailo AI expansion board (such as Hailo-8 or Hailo-8L). This pipeline combines hardware-accelerated model inference (`abraia.inference.hailo`), multi-object tracking (`abraia.inference.Tracker`), line crossing counters (`LineCounter`), and region timers (`RegionTimer`) through the shared `abraia.runtime.Pipeline`.

### Implementation Guide

Save this configuration as `edge_counter.json` and adjust the source,
coordinates, and model URI for the target installation:

```json
{
  "version": 1,
  "source": {
    "type": "camera",
    "src": 0,
    "resolution": [1920, 1080],
    "fps": 30,
    "video_unpaced": false
  },
  "model": {
    "task": "detection",
    "kind": "yolov8",
    "uri": "multiple/models/yolov8n_hailo8.hef",
    "params": {
      "batch_size": 1,
      "score_threshold": 0.3
    }
  },
  "stages": [
    {"type": "tracker", "enabled": true},
    {"type": "line_counter", "line": [[100, 540], [1820, 540]]},
    {
      "type": "region_timer",
      "polygon": [[300, 200], [1620, 200], [1620, 900], [300, 900]]
    }
  ],
  "display": {"show": true}
}
```

Run the saved pipeline from Python:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("edge_counter.json").run()
```

### Deployment on Raspberry Pi

Execute the pipeline directly on the Raspberry Pi:

```sh
python3 -c 'from abraia.runtime import Pipeline; Pipeline.from_file("edge_counter.json").run()'
```

---

## 🛠️ Development & Testing

Create an isolated environment and install the development dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

Run the test suite with coverage:

```sh
pytest -v tests/ --cov=abraia
```

Run a focused test file while iterating:

```sh
pytest -q tests/test_inference.py
```

Build source and wheel distributions with:

```sh
python setup.py sdist bdist_wheel
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
