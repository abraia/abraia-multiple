from abraia.inference import FaceRecognizer, FaceAttribute
from abraia.inference.faces import find_pose
from abraia.utils.draw import render_results, draw_overlay, draw_text
from abraia.utils import Video

recognition = FaceRecognizer()
attribute = FaceAttribute()

index = []
video = Video(0)
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
