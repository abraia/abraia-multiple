import time

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.utils.draw import render_results, render_counter, render_region
from abraia.inference.ops import count_objects


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
    frame = render_results(frame, results)
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    # video.write(frame)
    video.show(frame)
