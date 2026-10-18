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
  - [Public Python APIs](#public-python-apis)
  - [Image Editing & Enhancement](#image-editing--enhancement)
  - [Inference & Computer Vision](#inference--computer-vision)
  - [Training & Dataset Operations](#training--dataset-operations)
  - [Edge AI & Hardware Acceleration](#edge-ai--hardware-acceleration)
  - [Object Processing Pipelines](#object-processing-pipelines)
- [Real-Time Edge Object Counter](#real-time-edge-object-counter-on-raspberry-pi-with-hailo-npu)
- [License](#-license)

---

## 📦 Installation

Install the Abraia SDK from PyPI:

```sh
pip install -U abraia
```

The default install uses the CPU build of ONNX Runtime. To run ONNX demos on
an NVIDIA GPU on Linux, replace it with the CUDA build in the same Python
environment:

```sh
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Check that ONNX Runtime sees CUDA before starting the demo:

```sh
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output should include `CUDAExecutionProvider`. Then run a demo with
`--accelerator gpu`, for example `abraia run demo detect --accelerator gpu`.
ONNX Runtime's GPU package must match the host's CUDA and cuDNN runtime; see
the [ONNX Runtime CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
if CUDA is still missing from the provider list.

For training and development, install the optional development extras:

```sh
pip install -U abraia[dev]
```

To train RTMDet detection or instance-segmentation models, install the optional
extra and MMDetection's compiled MMCV dependency:

```sh
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install mmdet
pip install -U abraia[rtmdet]
abraia train PROJECT --backend rtmdet
```

RTMDet training uses OpenMMLab's pretrained checkpoints by default.
The trained model is exported and saved as a versioned ONNX artifact for
Abraia's model catalog. ONNX export uses PyTorch and does not require MMDeploy;
Abraia decodes the exported raw RTMDet heads at inference time. RTMDet can also
be compiled to Hailo from its trained checkpoint; that separate path still
uses MMDeploy to create its temporary DFC graph. Unlike the Ultralytics YOLO
path, RTMDet compilation does not inject Hailo NMS; its detection HEF uses
RTMDet host-side decoding and NMS in Abraia's Hailo inference path.

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
abraia compile PROJECT --device hailo8l
```

The command uploads the HEF and the complete Ultralytics Hailo sidecar bundle
to the project. Hailo-8/8L use DFC 3.33.1, while Hailo-10H/15H/15L use DFC
5.2.0; select the target matching the deployment accelerator.

---

## 🖥️ Command-Line Interface

Installing the package also installs the `abraia` command. Use
`abraia --help` or `abraia COMMAND --help` for the complete option list.

### Python SDK

Use `Abraia` for remote files, metadata, and image operations. It reads the
`ABRAIA_KEY` environment variable or the credentials saved by `abraia
configure`; you can also pass the key directly:

```python
from abraia import Abraia, APIError

client = Abraia()  # or Abraia(abraia_key="...")

try:
    files, folders = client.list_files("images/")
    remote_path = client.upload_file("images/input.jpg", "examples/input.jpg")
    local_path = client.download_file(remote_path, "downloads/input.jpg")
    client.transform_image(
        remote_path,
        "downloads/input-small.jpg",
        params={"width": 640},
    )
except APIError as error:
    print(f"Abraia API error ({error.code}): {error}")
```

`list_files` returns `(files, folders)`, `upload_file` returns the remote path,
and `download_file` returns the local cached or destination path. The client
also provides `check_file`, `move_file`, `remove_file`, metadata operations,
and load/save methods for files, JSON, and images. Image transforms write to
the destination path.

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

Prepare and train a model on a dataset. Detection and segmentation can use
Ultralytics (the default) or RTMDet. Ultralytics models can be compiled to a
Hailo HEF bundle:

```sh
abraia train PROJECT [EPOCHS]
abraia train PROJECT [EPOCHS] --backend rtmdet

abraia compile PROJECT \
  --device hailo8l \
  --conf 0.25 \
  --iou 0.7
```

Compilation uses the checkpoint saved with the latest versioned training
model in the project, downloading it from cloud storage as needed. It supports
`hailo8`, `hailo8l`, `hailo10h`, `hailo15h`, and
`hailo15l`. RTMDet compilation exports a temporary ONNX graph from the trained
checkpoint, uses it for Hailo compilation, then removes it. The normal
versioned ONNX export remains available separately. The DFC may reject
MMDeploy operators, especially for instance segmentation. Detection HEFs use
host-side confidence filtering and NMS. Segmentation HEFs require `dets`,
`labels`, and `masks` outputs in the compiled graph.
Pass `--conf` and `--iou` to set confidence and NMS IoU thresholds; both
accept values from 0 to 1 and otherwise use the compiler defaults. RTMDet
compilation uses the input size embedded in its ONNX file and calibration
images from the project's training split. Ultralytics uses the project's
`data.yaml`. On Linux x86_64, the matching Hailo
DFC wheel is installed automatically when it is required and available.

### Dataset curation and pruning

Analyze an image dataset for exact and similar duplicates, outliers, blurry
images, exposure issues, and invalid files:

```sh
abraia curate PROJECT
```

The command is a dry run by default and prints a JSON report. Apply duplicate,
blur, and invalid-image recommendations with `--apply`; include outlier
removals only after review with `--apply-outliers`. Use `--yes` with `--apply`
to skip the deletion confirmation in automation.

### Demos and inference

Run a built-in, pipeline-backed demo or a trained project. Built-in object
demos include `detect`, `segment`, `pose`, `people`, `queue`, `escalator`,
`tomato`, `apple`, `strawberry`, `grapes`, and `plates`; `faces` selects the
face-tracking demo:

```sh
abraia run demo DEMO_NAME [SOURCE] [--accelerator auto]
abraia run demo faces [SOURCE] [--accelerator auto]
abraia run PROJECT [CLASSES] [SOURCE] [--accelerator auto]
abraia search PROJECT [QUERY]
```

The `--accelerator` option applies to demos and can be `auto`, `onnx`, `cpu`,
`gpu`, or `hailo`. For a project, `abraia run` executes a saved version-two
pipeline JSON when one is present in the project; otherwise it selects the
project's highest-scoring ONNX model and runs it as a one-model pipeline.
`search` accepts a text query or a local image path; when omitted, the query
defaults to `man with red shirt`.

---

## 🚀 Core Modules & Features

### Public Python APIs

The package keeps a small set of supported entry points at its public module
boundaries:

- `abraia` exports `Abraia`, `APIError`, and `config` for remote file and image
  operations.
- `abraia.editing` exports `anonymize_image`, `blur_background`,
  `clean_image`, `inpaint_image`, `remove_background`, `smartcrop_image`, and
  `upscale_image`.
- `abraia.inference` exports `load_model` and `register_model_factory` for
  configured model adapters.
- `abraia.runtime` exports `Pipeline`, the Python entry point for saved
  inference workflows.
- `abraia.training` exports `ModelTrainer` for custom model training.
- `abraia.dataset` exports `create_dataset`, `delete_dataset`,
  `list_datasets`, and `load_dataset`.

Use imports from these module boundaries in applications. Implementation
modules such as `abraia.runtime.video`, `abraia.inference.hailo`, and
`abraia.utils` are internal and do not define the supported package-level API.

### Image Editing & Enhancement

`abraia.editing` provides super-resolution upscaling, smart cropping,
background removal, inpainting, anonymization, blur, and image cleanup.

### Inference & Computer Vision

The `run` command executes saved pipelines and built-in model workflows.
Pipeline JSON selects detection, segmentation, classification, OCR, face,
plate, and open-vocabulary models, plus tracking and region operations. Use
`abraia.runtime.Pipeline` to run the same workflows from Python.

### Training & Dataset Operations

Train custom classification, detection, and segmentation models through
`abraia.training.ModelTrainer` or the `abraia train` command. Training supports
small, medium, and large model sizes; detection and segmentation can use
Ultralytics YOLOv8n/m/l or the optional RTMDet backend, while classification
uses ResNet18/50/101. Ultralytics detection and segmentation models can be
exported to Hailo HEF through `ModelTrainer.compile()` or `abraia compile`.
Dataset creation and deletion are available from `abraia.dataset`. Dataset
curation and pruning use built-in Pillow and NumPy image analysis.

### Edge AI & Hardware Acceleration

The optional Hailo runtime is used internally by configured pipelines.

#### Object Processing Pipelines

Object detection, tracking, and counting workflows can be configured in a
JSON file and run from Python:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("pipeline.json").run()
```

Pipelines use ordered version-two steps for inference. Step identifiers are
generated internally from their order and are omitted from saved documents. A
pipeline can
use up to two models, along with `filter`, `crop`, `tracker`, `line_counter`,
`region_filter`, and `region_timer` steps. These steps consume the preceding
step results automatically. The second model runs on regions from the
preceding results and attaches its output to the matching detections. For
example, a license-plate detector can pass each detected plate to OCR:

```json
{
  "version": 2,
  "source": {"type": "video", "src": "traffic.mp4"},
  "steps": [
    {
      "type": "model",
      "model": {"task": "detection", "kind": "license_plate"}
    },
    {
      "type": "model",
      "model": {"task": "recognition", "kind": "ocr"}
    }
  ],
  "display": {"show": true}
}
```

The OCR step crops each detected plate, recognizes its text, and adds the
recognition result to that detection. Run a version-two file the same way:

```python
from abraia.runtime import Pipeline

Pipeline.from_file("plate_ocr.json").run()
```

The pipeline v2 JSON Schema is included with the package at
`abraia/runtime/pipeline-v2.schema.json`.

##### People monitoring

Use this configuration for people tracking and line counting. The CLI provides
the same pipeline-backed workflow:

```sh
abraia run demo people people-walking.mp4 --accelerator auto
```

![people detected](https://github.com/abraia/abraia-multiple/raw/master/images/people-detected.jpg)

For model-backed entries, `kind` identifies the model architecture or family
(`yolov8`, `yolov5`, `yolo11`, or `rtmdet`), while `task` identifies the operation
(`detection`, `segmentation`, `pose`, or `classification`). The runtime selects
ONNX or Hailo from the model URI (`.onnx` or `.hef`). RTMDet supports detection
and segmentation, including host-side Hailo postprocessing.

Built-in detectors do not require a model URI. For example:

```json
"model": {
  "task": "detection",
  "kind": "license_plate",
  "params": {"threshold": 0.85, "iou_threshold": 0.15}
}
```

##### License-plate recognition

Use `kind: "license_plate"` to select the built-in license-plate recognition
workflow.

![car license plate recognition](https://github.com/abraia/abraia-multiple/raw/master/images/car-plate.jpg)

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

##### Face recognition

Use `kind: "face"` to select the built-in face-recognition workflow. The
pipeline keeps the source, model, index, steps, display, and accelerator
selection in one configuration.

![rolling stones identified](https://github.com/abraia/abraia-multiple/raw/master/images/rolling-stones-identified.jpg)

Hailo models use the asynchronous producer/consumer runtime internally. The
pipeline keeps completed frames ordered before applying tracking and counting
steps:

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

The built-in RTMDet ONNX models use the same managed model paths as other
Abraia inference models. Select a detection or segmentation size in a pipeline:

```json
"model": {
  "kind": "rtmdet",
  "task": "detection",
  "size": "small"
}
```

Detection models are `multiple/models/rtmdet_{small,medium,large}_v1.onnx`;
segmentation models use `multiple/models/rtmdet_ins_{small,medium,large}_v1.onnx`.
Custom RTMDet ONNX files can still be selected with an explicit `uri`.

RTMDet uses fixed model-specific preprocessing in Abraia. Its HEF bundles use
their `abraia.json` sidecar for labels and host-side NMS settings.

---

## 🍓 Real-Time Edge Object Counter on Raspberry Pi with Hailo NPU

Deploy high-performance real-time object detection and counting on a Raspberry Pi equipped with a Hailo AI expansion board (such as Hailo-8 or Hailo-8L). This pipeline combines hardware-accelerated model inference, tracking, line crossing counts, and region timers through the shared `abraia.runtime.Pipeline`.

### Implementation Guide

Save this configuration as `edge_counter.json` and adjust the source,
coordinates, and model URI for the target installation:

```json
{
  "version": 2,
  "source": {
    "type": "camera",
    "src": 0,
    "resolution": [
      1920,
      1080
    ],
    "fps": 30,
    "video_unpaced": false
  },
  "steps": [
    {
      "type": "model",
      "model": {
        "task": "detection",
        "kind": "yolov8",
        "uri": "multiple/models/yolov8n_hailo8.hef",
        "params": {
          "batch_size": 1,
          "score_threshold": 0.3
        }
      }
    },
    {
      "type": "tracker",
      "enabled": true
    },
    {
      "type": "line_counter",
      "line": [
        [
          100,
          540
        ],
        [
          1820,
          540
        ]
      ]
    },
    {
      "type": "region_timer",
      "polygon": [
        [
          300,
          200
        ],
        [
          1620,
          200
        ],
        [
          1620,
          900
        ],
        [
          300,
          900
        ]
      ]
    }
  ],
  "display": {
    "show": true
  }
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

## 📄 License

This project is licensed under the [MIT License](LICENSE).
