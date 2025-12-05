import time

from abraia.utils import Video, show_image, save_image
from abraia.inference import Model, Tracker
from abraia.utils.draw import render_results, draw_overlay_mask, get_color, hex_to_rgb
from abraia.inference.ops import count_objects
from abraia.editing.sam import SAM


model_uri = 'multiple/models/yolov8n.onnx'
model = Model(model_uri)

# src = 'rtsp://192.168.1.41:8554/mystream'
src = '5479199-hd_1920_1080_25fps.mp4'
video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    # TODO: Add filter classes to model before nms
    results = [result for result in results if result['label'] == 'apple']
    results = tracker.update(results)
    out = render_results(frame.copy(), results)
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    # video.write(frame)
    video.show(out)

# Automatically segment apples from the previously detected boxes
import json
sam = SAM()
sam.encode(frame)
out = frame.copy()
for result in results:
    track_id = result.get('track_id')
    x, y, w, h = result['box']
    color = get_color(track_id) if track_id is not None else result.get('color', get_color(0))
    mask = sam.predict(frame, prompt=json.dumps([{"type": "rectangle", "data": [x, y, x+w, y+h]}]))
    out = draw_overlay_mask(out, mask, color=hex_to_rgb(color), opacity=0.7)
out = render_results(out, results)
# save_image(out, 'apple_segmentation.jpg')
show_image(out)

