import os
import time
from copy import deepcopy

from abraia.inference import FaceRecognizer, FaceAttribute, PlateRecognizer
from abraia.inference.faces import find_pose
from abraia.inference.ops import count_objects
from abraia.utils import Pipeline
from abraia.utils.draw import render_results, draw_overlay, draw_text_multiline
from abraia.utils import Video, download_url, load_image


PIPELINES = {
    'tomato': {
        'version': 1,
        'source': {'type': 'video', 'src': '10179855-hd_1280_720_30fps.mp4'},
        'model': {
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
            'uri': 'multiple/grapes/yolov8n.onnx',
            'labels': ['grapes'],
        },
        'stages': [{'type': 'tracker'}],
        'display': {'show': True},
    },
    'people': {
        'version': 1,
        'source': {'type': 'video', 'src': '853889-hd_1920_1080_25fps.mp4'},
        'model': {'uri': 'multiple/models/yolov8n.onnx', 'labels': ['person']},
        'stages': [
            {'type': 'region_filter', 'polygon': [[0, 600], [1920, 600], [1920, 700], [0, 700]]},
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[0, 650], [1920, 650]]},
        ],
        'display': {'show': True},
    },
    'queue': {
        'version': 1,
        'source': {'type': 'video', 'src': '4775505-hd_1920_1080_30fps.mp4'},
        'model': {'uri': 'multiple/models/yolov8n.onnx', 'labels': ['person']},
        'stages': [
            {'type': 'tracker'},
            {'type': 'region_timer', 'polygon': [[10, 600], [1690, 600], [1690, 700], [10, 700]]},
        ],
        'display': {'show': True},
    },
    'escalator': {
        'version': 1,
        'source': {'type': 'video', 'src': '14393755-hd_1920_1080_30fps.mp4'},
        'model': {'uri': 'multiple/models/yolov8n.onnx', 'labels': ['person']},
        'stages': [
            {'type': 'region_filter', 'polygon': [[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]]},
            {'type': 'tracker'},
            {'type': 'line_counter', 'line': [[950, 670], [270, 895]]},
            {'type': 'region_timer', 'polygon': [[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]]},
        ],
        'display': {'show': True},
    },
}


DEFAULT_PIPELINE = {
    'version': 1,
    'source': {'type': 'video', 'src': 0},
    'model': {'uri': 'multiple/models/yolov8n.onnx'},
    'stages': [{'type': 'tracker'}],
    'display': {'show': True},
}


HAILO_DEMOS = {
    'tomato': {
        'hef_path': 'multiple/tomato/yolov8n.hef',
        'src': '10179855-hd_1280_720_30fps.mp4'
    },
    'apple': {
        'hef_path': 'yolov5m_seg_with_nms',
        'task': 'segment',
        'src': '5479199-hd_1280_720_25fps.mp4'
    },
    'segment': {
        'hef_path': 'yolov8n_seg',
        'task': 'segment',
        'model_type': 'v8',
        'src': '853889-hd_1920_1080_25fps.mp4'
    },
    'pose': {
        'hef_path': 'yolov8m_pose',
        'task': 'pose',
    }
}


def monitor_objects(src=None, demo='detect', resolution=(1280, 720)):
    """Monitor, count, or just detect objects in a video stream."""
    print(f"Available demos: {', '.join(PIPELINES.keys())}")
    selected = deepcopy(PIPELINES.get(demo, DEFAULT_PIPELINE))
    src = src if src is not None else selected['source']['src']
    if isinstance(src, str) and not os.path.exists(src) and src.endswith('.mp4'):
        download_url(f"https://api.abraia.me/files/multiple/videos/{src}", src)
    selected['source']['src'] = src
    selected['source']['resolution'] = list(resolution)

    def report(context, elapsed_ms):
        print(
            f"#{context.frame_index} {round(elapsed_ms, 1)}ms "
            f"{count_objects(context.results)}"
        )

    pipeline = Pipeline.from_dict(
        selected,
        on_frame=report,
    )
    pipeline.run()

    line_counter = pipeline.components.get("line_counter")
    if line_counter:
        print(
            f"Final In: {line_counter.in_count}, "
            f"Final Out: {line_counter.out_count}"
        )


def monitor_objects_hailo(src=None, demo='detect'):
    """Monitor, count, or just detect objects in a video stream using Hailo."""
    from abraia.hailo import detect
    print(f"Available Hailo demos: {', '.join(HAILO_DEMOS.keys())}")
    selected = HAILO_DEMOS.get(demo) or {}
    src = src or selected.get('src', 0)
    if isinstance(src, str) and not os.path.exists(src) and src.endswith('.mp4'):
        download_url(f"https://api.abraia.me/files/multiple/videos/{src}", src)
    options = selected.copy()
    options['input'] = src
    detect.main(**options)


def track_faces(src=None, resolution=(1280, 720)):
    """Track faces in a video stream from a file or webcam."""
    recognition = FaceRecognizer()
    attribute = FaceAttribute()
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


def detect_plates(src=None):
    """Detect license plates in a video and show OCR results."""
    src = src or 'cars.mp4'
    if isinstance(src, str) and not os.path.exists(src) and src.endswith('.mp4'):
        download_url(f"https://api.abraia.me/files/multiple/videos/{src}", src)
    video = Video(src)
    recognizer = PlateRecognizer()
    for frame in video:
        results = recognizer.recognize(frame)
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
