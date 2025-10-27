from abraia.inference.faces import FaceRecognizer, find_pose
from abraia.utils import Video, save_image
from abraia.utils.draw import render_results, draw_overlay, draw_text


recognition = FaceRecognizer()

index = []
def index_face(img,  result):
    face = recognition.extract_faces(img, [result])[0]
    # save_image(face, f"face_{len(index)}.png")
    index.append({'name': f"face_{len(index)}", 'vector': result['vector']})

video = Video(0)
for frame in video:
    results = recognition.detect_faces(frame)
    faces = recognition.extract_faces(frame, results)
    for k, face in enumerate(faces):
        (h, w), (fh, fw) = frame.shape[:2], face.shape[:2]
        frame = draw_overlay(frame, face, [w - fw, k * fh, fw, fh])
    results = recognition.represent_faces(frame, results)
    results = recognition.identify_faces(results, index)
    for result in results:
        roll, yaw, pitch = find_pose(result['keypoints'])
        draw_text(frame, f"Roll: {roll} degrees", (10, 40))
        draw_text(frame, f"Yaw: {yaw} degrees", (10, 70))
        draw_text(frame, f"Pitch: {pitch} degrees", (10, 100))
        if result['label'] == 'unknown':
            index_face(frame, result)
    frame = render_results(frame, results)
    video.show(frame)
