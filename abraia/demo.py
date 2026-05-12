import os
import time

from tqdm import tqdm
from glob import glob

from abraia.inference import Model, Tracker, FaceRecognizer, FaceAttribute, PlateRecognizer
from abraia.inference.faces import find_pose
from abraia.inference.clip import Clip
from abraia.inference.ops import count_objects, search_vector
from abraia.inference.tools import LineCounter, RegionFilter, RegionTimer
from abraia.utils.draw import render_results, render_counter, render_region, draw_overlay, draw_text_multiline
from abraia.utils import Video, download_url, load_image, show_image


DEMOS = {
    'apple': {
        'src': '5479199-hd_1920_1080_25fps.mp4',
        'label': 'apple'
    },
    'strawberry': {
        'model': 'multiple/strawberry/yolov8n.onnx',
        'src': '9710983-hd_1920_1080_30fps.mp4',
        'label': 'strawberry'
    },
    'grapes': {
        'model': 'multiple/grapes/yolov8n.onnx',
        'src': '5658544-hd_1366_720_24fps.mp4',
        'label': 'grapes'
    },
    'tomato': {
        'model': 'multiple/tomato/yolov8n_v6.onnx',
        'src': '10179855-hd_1920_1080_30fps.mp4',
        'label': 'tomato',
        'line': [(1440, 0), (1440, 1080)]
    },
    'people': {
        'src': '853889-hd_1920_1080_25fps.mp4',
        'label': 'person',
        'line': [(0, 650), (1920, 650)],
        'region': [(0, 600), (1920, 600), (1920, 700), (0, 700)]
    },
    'queue': {
        'src': '4775505-hd_1920_1080_30fps.mp4',
        'label': 'person',
        'timer': [(10, 600), (1690, 600), (1690, 700), (10, 700)]
    },
    'escalator': {
        'src': '14393755-hd_1920_1080_30fps.mp4',
        'label': 'person',
        'line': [(950, 670), (270, 895)],
        'region': [[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]],
        'timer': [[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]]
    }
}


def monitor_objects(src=None, demo='detect', resolution=(1280, 720)):
    """Monitor, count, or just detect objects in a video stream."""
    print(f"Available demos: {', '.join(DEMOS.keys())}")
    selected = DEMOS.get(demo) or {}
    src = src or selected.get('src', 0)
    if isinstance(src, str) and not os.path.exists(src) and src.endswith('.mp4'):
        download_url(f"https://api.abraia.me/files/multiple/videos/{src}", src)

    video = Video(src, resolution=resolution)
    tracker = Tracker(frame_rate=video.frame_rate)
    model = Model(selected.get('model', 'multiple/models/yolov8n.onnx'))
    
    line_counter = LineCounter(selected['line']) if selected.get('line') else None
    region_filter = RegionFilter(selected['region']) if selected.get('region') else None
    region_timer = RegionTimer(selected['timer']) if selected.get('timer') else None

    labels = [selected.get('label')] if selected.get('label') else None
    for k, frame in enumerate(video):
        frame_time = round(k / video.frame_rate, 2)
        t0 = time.time()
        results = model.run(frame, labels=labels)
        
        if region_filter:
            results, _ = region_filter.update(results)
        results = tracker.update(results)
        
        out = frame.copy()
        if line_counter:
            in_count, out_count = line_counter.update(results)
            out = render_counter(out, line_counter.line, f"In: {in_count} | Out: {out_count}" if demo in ['people', 'escalator'] else f"Count: {out_count}")
        
        if region_timer:
            in_objects, out_objects = region_timer.update(results, frame_time)
            out = render_region(out, region_timer.region, f"Count: {len(in_objects)}", color=(255, 255, 0))
            out = render_results(out, in_objects)
        else:
            out = render_results(out, results)
        
        print(f"#{k} {round((time.time() - t0) * 1000, 1)}ms {count_objects(results)}")
        video.show(out)

    if line_counter:
        print(f"Final In: {in_count}, Final Out: {out_count}")


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


def retrieve_images(query="man with red shirt", path='images/*.jpg'):
    """Search for images using a text query."""
    clip_model = Clip()
    print("Building image index...")
    image_paths = glob(path)
    image_index = [{'vector': clip_model.get_image_embeddings([load_image(image_path)])[0]} for image_path in tqdm(image_paths)]
    print("Searching for images...")
    print("Query:", query)
    vector = clip_model.get_text_embeddings([query])[0]
    idxs, scores = search_vector(vector, image_index, max_results=5)
    for idx in idxs:
        img = load_image(image_paths[idx])
        show_image(img)
