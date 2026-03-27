import time

from abraia.utils import Video, show_image, save_image
from abraia.inference import Model, Tracker
from abraia.utils.draw import render_results, draw_overlay_mask, get_color, hex_to_rgb
from abraia.inference.ops import count_objects


model_uri = 'multiple/strawberry/yolov8n.onnx'
model = Model(model_uri)

src = '9710983-hd_1920_1080_30fps.mp4'
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


from abraia.inference.detect import segment_objects

out = frame.copy()
results = segment_objects(frame, results)
print(results)
out = render_results(out, results)
show_image(out)
