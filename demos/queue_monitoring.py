import time

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionFilter, RegionTimer
from abraia.utils.draw import render_results, render_counter, render_region
from abraia.inference.ops import count_objects


model_uri = 'multiple/models/yolov8n.onnx'
model = Model(model_uri)

src = '4775505-hd_1920_1080_30fps.mp4'
video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)
region_timer = RegionTimer([(10, 600), (1690, 600), (1690, 700), (10, 700)])
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    results = [result for result in results if result['label'] == 'person']
    results = tracker.update(results)
    in_objects, out_objects = region_timer.update(results, frame_time)
    frame = render_results(frame, in_objects)
    frame = render_region(frame, region_timer.region, f"Count: {len(in_objects)}")
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    # video.write(frame)
    video.show(frame)
