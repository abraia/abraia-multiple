from abraia.inference import Model, FaceRecognizer, FaceAttribute
from abraia.inference.faces import find_pose
from abraia.utils.draw import render_results, draw_overlay, draw_text
from abraia.utils import Video


def detect_objects(src=0, resolution=(1280, 720)):
    """Detect objects in a video stream from a file or webcam."""
    model = Model('multiple/models/yolov8n.onnx')
    video = Video(src, resolution=resolution)
    for frame in video:
        results = model.run(frame)
        frame = render_results(frame, results)
        video.show(frame)


def track_faces(src=0, resolution=(1280, 720)):
    """Track faces in a video stream from a file or webcam."""
    recognition = FaceRecognizer()
    attribute = FaceAttribute()

    index = []
    video = Video(src, resolution=resolution)
    for frame in video:
        results = recognition.detect_faces(frame)
        faces = recognition.extract_faces(frame, results)
        results = recognition.identify_faces(frame, results, index)
        for k, (face, result) in enumerate(zip(faces, results)):
            (h, w), (fh, fw) = frame.shape[:2], face.shape[:2]
            frame = draw_overlay(frame, face, [w - fw, k * fh, fw, fh])
            roll, yaw, pitch = find_pose(result['keypoints'])
            draw_text(frame, f"Roll: {roll} degrees", (10, 40))
            draw_text(frame, f"Pitch: {pitch} degrees", (10, 70))
            draw_text(frame, f"Yaw: {yaw} degrees", (10, 100))
            if result['label'] == 'unknown':
                index.append({'name': f"face_{len(index)}", 'vector': result['vector']})
            gender, age, score = attribute.predict(face)
            print(f"{gender[0]} {age}, {score}")
            result['label'] = f"{gender[0]} {age} ({result['label']})"
        frame = render_results(frame, results)
        video.show(frame)
