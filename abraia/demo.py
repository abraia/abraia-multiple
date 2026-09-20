"""Interactive demo runners backed by the shared pipeline catalog."""

import os
from copy import deepcopy

from abraia.inference import FaceRecognizer, FaceAttribute
from abraia.inference.models.faces import find_pose
from abraia.runtime.stages import count_objects
from abraia.runtime import Pipeline
from abraia.utils.draw import render_results, draw_overlay, draw_text_multiline
from abraia.runtime import Video
from abraia.utils import download_url, load_image
from abraia.inference.accelerators import (
    hailo_device_arch as _hailo_device_arch,
    hailo_model_available as _hailo_model_available,
    normalize_accelerator,
    onnx_providers,
)


DEFAULT_PIPELINE = {
    'version': 1,
    'source': {'type': 'camera', 'src': 0},
    'model': {
        'task': 'detection',
        'kind': 'yolov8',
        'uri': 'multiple/models/yolov8n.onnx',
    },
    'stages': [{'type': 'tracker'}],
    'display': {'show': True},
}


PIPELINES = {
    'tomato': {
        'version': 1,
        'source': {'type': 'video', 'src': '10179855-hd_1280_720_30fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/tomato/yolov8n_v6.onnx',
            'labels': ['tomato'],
        },
        'stages': [
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[960, 0], [960, 720]]},
        ],
        'display': {'show': True},
    },
    'apple': {
        'version': 1,
        'source': {'type': 'video', 'src': '5479199-hd_1280_720_25fps.mp4'},
        'model': {
            'task': 'segmentation',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8n-seg.onnx',
            'labels': ['apple'],
        },
        'stages': [
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[960, 0], [960, 720]]},
        ],
        'display': {'show': True},
    },
    'strawberry': {
        'version': 1,
        'source': {'type': 'video', 'src': '9710983-hd_1920_1080_30fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/strawberry/yolov8n.onnx',
            'labels': ['strawberry'],
        },
        'stages': [{'type': 'tracker'}],
        'display': {'show': True},
    },
    'grapes': {
        'version': 1,
        'source': {'type': 'video', 'src': '5658544-hd_1366_720_24fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/grapes/yolov8n.onnx',
            'labels': ['grapes'],
        },
        'stages': [{'type': 'tracker'}],
        'display': {'show': True},
    },
    'people': {
        'version': 1,
        'source': {'type': 'video', 'src': '853889-hd_1920_1080_25fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8n.onnx',
            'labels': ['person'],
        },
        'stages': [
            {
                'type': 'region_filter',
                'polygon': [[0, 600], [1920, 600], [1920, 700], [0, 700]],
            },
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[0, 650], [1920, 650]]},
        ],
        'display': {'show': True},
    },
    'queue': {
        'version': 1,
        'source': {'type': 'video', 'src': '4775505-hd_1920_1080_30fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8n.onnx',
            'labels': ['person'],
        },
        'stages': [
            {'type': 'tracker'},
            {
                'type': 'region_timer',
                'polygon': [[10, 600], [1690, 600], [1690, 700], [10, 700]],
            },
        ],
        'display': {'show': True},
    },
    'escalator': {
        'version': 1,
        'source': {'type': 'video', 'src': '14393755-hd_1920_1080_30fps.mp4'},
        'model': {
            'task': 'detection',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8n.onnx',
            'labels': ['person'],
        },
        'stages': [
            {
                'type': 'region_filter',
                'polygon': [
                    [0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0],
                ],
            },
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[950, 670], [270, 895]]},
            {
                'type': 'region_timer',
                'polygon': [
                    [0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0],
                ],
            },
        ],
        'display': {'show': True},
    },
    'plates': {
        'version': 1,
        'source': {'type': 'video', 'src': 'cars.mp4'},
        'model': {
            'task': 'recognition',
            'kind': 'license_plate',
            'params': {'threshold': 0.85, 'iou_threshold': 0.15},
        },
        'stages': [],
        'display': {'show': True},
    },
    'detect': {
        **deepcopy(DEFAULT_PIPELINE),
    },
    'segment': {
        'version': 1,
        'source': {
            'type': 'video',
            'src': '853889-hd_1920_1080_25fps.mp4',
        },
        'model': {
            'task': 'segmentation',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8n-seg.onnx',
        },
        'stages': [{'type': 'tracker'}],
        'display': {'show': True},
    },
    'pose': {
        'version': 1,
        'source': {
            'type': 'camera',
            'src': 0,
            'resolution': [1280, 720],
            'fps': 30,
        },
        'model': {
            'task': 'pose',
            'kind': 'yolov8',
            'uri': 'multiple/models/yolov8m_pose.onnx',
        },
        'stages': [{'type': 'tracker'}],
        'display': {'show': True},
    },
}


PIPELINE_DEVICES = PIPELINES


# The current Hailo apple entry uses a generic COCO segmentation HEF. Keep it
# available for explicit ``accelerator='hailo'`` runs, but do not select it
# automatically as an apple-equivalent model.
HAILO_AUTO_EXCLUSIONS = frozenset({'apple'})
VIDEO_URL = 'https://api.abraia.me/files/multiple/videos/{}'


def _resolve_hailo_config(config, architecture):
    """Pair a canonical demo ONNX URI with an architecture-specific HEF."""
    selected = deepcopy(config)
    model = selected.get('model', {})
    from abraia.inference.accelerators import paired_hailo_uri
    from abraia.inference.hailo.models import model_type_from_onnx_uri
    from abraia.tasks import normalize_task

    onnx_uri = model.get('uri')
    kind = str(model.get('kind', '')).strip().lower()
    task = normalize_task(model.get('task'))
    if kind not in ('yolov5', 'yolov8', 'yolo11') or not task:
        return None
    hef_uri = paired_hailo_uri(
        onnx_uri,
        task,
        architecture,
    )
    if not hef_uri:
        return None
    model['task'] = task
    model['kind'] = kind
    model['uri'] = hef_uri
    params = model.setdefault('params', {})
    if not isinstance(params, dict):
        return None
    model_type = model_type_from_onnx_uri(onnx_uri)
    if model_type:
        params.setdefault('model_type', model_type)
    return selected


def _prepare_source(config, src=None, resolution=None):
    """Apply runtime source overrides without adding camera settings to video."""
    source = config['source']
    source['src'] = source['src'] if src is None else src
    if resolution is not None and source.get('type') == 'camera':
        source['resolution'] = list(resolution)
    return source['src']


def _ensure_video_available(src):
    """Download a catalog video when it is not present locally."""
    if isinstance(src, str) and not os.path.exists(src) and src.endswith('.mp4'):
        download_url(VIDEO_URL.format(src), src)


def resolve_pipeline(demo='detect', accelerator='auto'):
    """Resolve a logical demo to its unchanged runtime pipeline config.

    ``hailo`` is treated as an accelerator selection.  CPU, GPU, and ONNX
    selections use the regular ONNX pipeline definition; ONNX Runtime chooses
    the available provider for that model.  ``auto`` prefers Hailo only when a
    compatible device and model are available, then falls back to ONNX.
    """
    accelerator = normalize_accelerator(accelerator)

    config = PIPELINE_DEVICES.get(demo)
    if config is None:
        if accelerator == 'hailo':
            raise KeyError(f"Unknown Hailo pipeline demo: {demo}")
        return deepcopy(DEFAULT_PIPELINE)

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
                f"No Hailo model is configured for the '{demo}' demo on "
                f"{architecture}"
            )
        if not _hailo_model_available(resolved_hailo_config, architecture):
            raise RuntimeError(
                f"The Hailo model for the '{demo}' demo is not available "
                f"for {architecture}"
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
        if (
            resolved_hailo_config
            and _hailo_model_available(resolved_hailo_config, architecture)
        ):
            return resolved_hailo_config

    return deepcopy(config)


def monitor_objects(
    src=None,
    demo='detect',
    resolution=(1280, 720),
    accelerator='auto',
):
    """Monitor, count, or just detect objects in a video stream."""
    print(f"Available demos: {', '.join(PIPELINE_DEVICES.keys())}")
    selected = resolve_pipeline(demo, accelerator=accelerator)
    src = _prepare_source(selected, src, resolution)
    _ensure_video_available(src)

    def report(context, elapsed_ms):
        print(
            f"#{context.frame_index} {round(elapsed_ms, 1)}ms "
            f"{count_objects(context.results)}"
        )

    pipeline_options = {'on_frame': report}
    requested_accelerator = normalize_accelerator(accelerator)
    if (
        requested_accelerator != 'auto'
        and not str(selected.get('model', {}).get('uri', '')).lower().endswith('.hef')
    ):
        pipeline_options['accelerator'] = requested_accelerator
    pipeline = Pipeline.from_dict(selected, **pipeline_options)
    pipeline.run()

    line_counter = pipeline.components.get("line_counter")
    if line_counter:
        print(
            f"Final In: {line_counter.in_count}, "
            f"Final Out: {line_counter.out_count}"
        )


def track_faces(src=None, resolution=(1280, 720), accelerator='auto'):
    """Track faces in a video stream from a file or webcam."""
    providers = onnx_providers(accelerator)
    recognition = FaceRecognizer(providers=providers)
    attribute = FaceAttribute(providers=providers)
    index = []
    src = src or 0
    video = Video(src, resolution=resolution)
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
    from abraia.inference import ImageSearch, InteractiveSAM

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
