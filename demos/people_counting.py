import time

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionFilter
from abraia.utils.draw import render_results, render_counter, render_region
from abraia.inference.ops import count_objects


model_uri = 'multiple/models/yolov8n.onnx'
model = Model(model_uri)

src = '853889-hd_1920_1080_25fps.mp4'
video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)
line_counter = LineCounter([(0, 650), (1920, 650)])
region_filter = RegionFilter([(0, 600), (1920, 600), (1920, 700), (0, 700)])
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    results = [result for result in results if result['label'] == 'person']
    results, _ = region_filter.update(results)
    results = tracker.update(results)
    # frame = render_region(frame, region_filter.region)
    frame = render_results(frame, results)
    in_count, out_count = line_counter.update(results)
    frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    video.show(frame)
print(in_count, out_count)
