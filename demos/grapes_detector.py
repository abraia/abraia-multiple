import time

from abraia.utils import Video, show_image, load_image, show_image, save_image
from abraia.inference import Model, Tracker
from abraia.utils.draw import render_results, draw_overlay_mask, get_color, hex_to_rgb
from abraia.inference.ops import count_objects


model_uri = 'multiple/grapes/yolov8n.onnx'
model = Model(model_uri)

src = '5658544-hd_1366_720_24fps.mp4'
video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    results = tracker.update(results)
    out = render_results(frame.copy(), results)
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    # video.write(frame)
    video.show(out)

img = load_image('f8a666d53c50706b.jpg')
results = model.run(img)
out = render_results(img, results)
show_image(out)

import json
from abraia.editing import SAM

sam = SAM()
sam.encode(img)
out = img.copy()
for result in results:
    track_id = result.get('track_id')
    x, y, w, h = result['box']
    color = get_color(track_id) if track_id is not None else result.get('color', get_color(0))
    mask = sam.predict(img, prompt=json.dumps([{"type": "rectangle", "data": [x, y, x+w, y+h]}]))
    out = draw_overlay_mask(out, mask, color=hex_to_rgb(color), opacity=0.7)
out = render_results(out, results)
show_image(out)

