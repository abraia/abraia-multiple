"""Interactive demo runners backed by the shared pipeline catalog."""

import json
import os
import time
from copy import deepcopy

from requests import RequestException

from abraia.inference.models.faces import FaceRecognizer, FaceAttribute
from abraia.inference.models.sam import InteractiveSAM
from abraia.inference.search import ImageSearch
from abraia.inference.models.faces import find_pose
from abraia.runtime import Pipeline
from abraia.runtime.config import is_primary_model_step
from abraia.utils.draw import render_results, draw_overlay, draw_text_multiline
from abraia.runtime.video import Video
from abraia.media.images import load_image
from abraia.utils.remote import load_url, url_path
from abraia.inference.accelerators import (
    hailo_device_arch as _hailo_device_arch,
    hailo_model_available as _hailo_model_available,
    normalize_accelerator,
    onnx_providers,
    paired_hailo_uri,
)
from abraia.inference.model_config import (
    model_variants,
    normalize_task,
)


DEFAULT_DEMO = 'detect'
PI_CAMERA_DEMO = 'pi-camera'


DEMO_TASKS = {
    DEFAULT_DEMO: 'detection',
    'tomato': 'detection',
    'apple': 'segmentation',
    'strawberry': 'segmentation',
    'grapes': 'detection',
    'people': 'detection',
    'queue': 'detection',
    'escalator': 'detection',
    'plates': 'recognition',
    'segment': 'segmentation',
    'pose': 'pose',
    'yolo9': 'detection',
    'rfdetr': 'detection',
    'rfdetr-seg': 'segmentation',
    'dfine': 'detection',
    'dfine-seg': 'segmentation',
}


MODEL_VARIANT_KINDS = ('yolov8', 'yolo9', 'rfdetr', 'dfine')


# The current Hailo apple entry uses a generic COCO segmentation HEF. Keep it
# available for explicit ``accelerator='hailo'`` runs, but do not select it
# automatically as an apple-equivalent model.
HAILO_AUTO_EXCLUSIONS = frozenset({'apple'})
DEMO_PIPELINE_ROOT = 'multiple/demos'


def load_public_pipeline(demo):
    """Fetch a demo configuration without using an account or file cache."""
    path = f'{DEMO_PIPELINE_ROOT}/{demo}.json'
    try:
        stream = load_url(url_path(path), timeout=(5, 10))
    except RequestException:
        stream = None
    if stream is None:
        raise ValueError(f"Could not download public demo '{demo}' from {path}")
    with stream:
        try:
            configuration = json.load(stream)
        except ValueError as error:
            raise ValueError(f"Invalid JSON for public demo '{demo}' at {path}") from error
    if not isinstance(configuration, dict) or configuration.get('version') != 2:
        raise ValueError(f"Invalid version 2 pipeline for public demo '{demo}'")
    return configuration


def demo_model_options(demo):
    """List model variants compatible with a demo's model task."""
    task = DEMO_TASKS.get(demo)
    if task is None:
        return ()
    variants = model_variants(MODEL_VARIANT_KINDS)
    return tuple(
        name for name, model in variants.items()
        if model['task'] == task
    )


def _apply_demo_model(config, demo, model_name):
    """Replace a demo's primary model with a compatible built-in variant."""
    model_name = str(model_name).strip().lower()
    variants = model_variants(MODEL_VARIANT_KINDS)
    variant = variants.get(model_name)
    model = _primary_model_step(config)['model']
    task = normalize_task(model.get('task'))
    if variant is not None and variant['task'] == task:
        model.update(variant)
        return
    available = ', '.join(
        name for name, option in variants.items() if option['task'] == task
    )
    if variant is None:
        raise ValueError(
            f"Unknown model variant '{model_name}'. Compatible options for "
            f"'{demo}' are: {available}"
        )
    raise ValueError(
        f"Model '{model_name}' supports {variant['task']}, while demo "
        f"'{demo}' uses {task}. Compatible models: {available}"
    )


def _primary_model_step(config):
    """Return the frame model step in a version-two pipeline."""
    return next(
        step for step in config.get('steps', [])
        if is_primary_model_step(step)
    )


def _resolve_hailo_config(config, architecture):
    """Select an available architecture-specific HEF for a demo model."""
    selected = deepcopy(config)
    model = _primary_model_step(selected)['model']
    onnx_uri = model.get('uri')
    kind = str(model.get('kind', '')).strip().lower()
    task = normalize_task(model.get('task'))
    if kind not in ('yolov5', 'yolov8', 'yolo9', 'yolo11') or not task:
        return None

    hef_uri = paired_hailo_uri(
        onnx_uri,
        architecture,
    )
    if not hef_uri:
        return None
    model['task'] = task
    model['kind'] = kind
    model['uri'] = hef_uri
    return selected if _hailo_model_available(selected, architecture) else None


def run_camera_pipeline(
    src=None,
    accelerator='auto',
    output=None,
):
    """Publish the Raspberry Pi camera through a managed MediaMTX server."""
    src = 0 if src is None else src
    rtsp_url = os.environ.get('ABRAIA_RTSP_URL')
    if rtsp_url:
        selected = {
            'version': 2,
            'source': {
                'type': 'rpi_camera',
                'src': src,
                'resolution': [1280, 720],
                'fps': 30,
            },
            'steps': [],
            'display': {'show': False},
            'output': {'type': 'rtsp', 'url': rtsp_url, 'bitrate': 4_000_000},
        }
        if output is not None:
            selected['display']['dest'] = output
        with Pipeline.from_dict(selected, accelerator=accelerator) as pipeline:
            pipeline.run()
        return

    if output is not None:
        raise ValueError(
            "The automatic pi-camera server publishes a stream and does not "
            "write a local video file. Set ABRAIA_RTSP_URL to use an external "
            "RTSP server with --output."
        )

    from abraia.media.mediamtx import managed_pi_camera_server

    with managed_pi_camera_server(src) as stream_url:
        print(f"Raspberry Pi camera stream: {stream_url}")
        print("Press Ctrl+C to stop the stream.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def resolve_pipeline(demo=DEFAULT_DEMO, accelerator='auto', model=None):
    """Load a public demo and select its runtime model.

    ``hailo`` is treated as an accelerator selection.  CPU, GPU, and ONNX
    selections use the regular ONNX pipeline definition; ONNX Runtime chooses
    the available provider for that model.  ``auto`` prefers Hailo only when a
    compatible device and model are available, then falls back to ONNX.
    """
    accelerator = normalize_accelerator(accelerator)
    if demo not in DEMO_TASKS:
        raise ValueError(f"Unknown demo '{demo}'")
    config = load_public_pipeline(demo)
    if model is not None:
        _apply_demo_model(config, demo, model)

    if accelerator == 'hailo':
        architecture = _hailo_device_arch()
        if not architecture:
            raise RuntimeError(
                "Hailo accelerator requested, but no compatible Hailo device "
                "was detected"
            )
        resolved_hailo_config = _resolve_hailo_config(config, architecture)
        if resolved_hailo_config is None:
            raise RuntimeError(
                f"No Hailo model is available for the '{demo}' demo on "
                f"{architecture}"
            )
        return resolved_hailo_config

    if (
        accelerator == 'auto'
        and demo not in HAILO_AUTO_EXCLUSIONS
    ):
        architecture = _hailo_device_arch()
        resolved_hailo_config = (
            _resolve_hailo_config(config, architecture)
            if architecture
            else None
        )
        if resolved_hailo_config:
            return resolved_hailo_config

    return config


def track_faces(src=None, resolution=(1280, 720), accelerator='auto', output=None):
    """Track faces in a video stream from a file or webcam."""
    providers = onnx_providers(accelerator)
    recognition = FaceRecognizer(providers=providers)
    attribute = FaceAttribute(providers=providers)
    index = []
    src = src or 0
    with Video(src, resolution=resolution, dest=output) as video:
        for frame in video:
            results = recognition.detect_faces(frame)
            faces = recognition.extract_faces(frame, results)
            results = recognition.identify_faces(frame, results, index)
            for k, (face, result) in enumerate(zip(faces, results)):
                (h, w), (fh, fw) = frame.shape[:2], face.shape[:2]
                frame = draw_overlay(frame, face, [w - fw, k * fh, fw, fh])
                roll, yaw, pitch = find_pose(result['keypoints'])
                draw_text_multiline(frame, [f"Roll: {roll} degrees",
                                            f"Pitch: {pitch} degrees",
                                            f"Yaw: {yaw} degrees"], (10, 140), text_color=(255, 0, 0), text_scale=0.6)
                if result['label'] == 'unknown':
                    index.append({'name': f"face_{len(index)}", 'vector': result['vector']})
                gender, age, score = attribute.predict(face)
                print(f"{gender[0]} {age}, {score}")
                result['label'] = f"{gender[0]} {age} ({result['label']})"
            frame = render_results(frame, results)
            video.show(frame)


def search_images(project, query="man with red shirt"):
    """Search for images using a text query or an image path (with click object selection) via ImageSearch."""

    search = ImageSearch(project)
    print("Searching for images...")
    print("Query:", query)

    if os.path.exists(query):
        img = load_image(query)
        interactive_sam = InteractiveSAM(img)
        cropped, box = interactive_sam.select_object()
        if cropped is None:
            print("No object selected.")
            cropped = img
        print("Searching for similar images in project...")
        search.search_similar(cropped, max_results=5)
    else:
        search.search_text(query, max_results=5)
