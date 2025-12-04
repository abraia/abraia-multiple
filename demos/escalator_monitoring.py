import time

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionFilter, RegionTimer
from abraia.utils.draw import render_results, render_counter, render_region
from abraia.inference.ops import count_objects


model_uri = 'multiple/models/yolov8n.onnx'
model = Model(model_uri)

src = '14393755-hd_1920_1080_30fps.mp4'
video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)
line_counter = LineCounter([(950, 670), (270, 895)])
region_filter = RegionFilter([[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]])
region_timer = RegionTimer([[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]])
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    # TODO: Add filter classes to model before nms
    results = [result for result in results if result['label'] == 'person']
    results, _ = region_filter.update(results)
    results = tracker.update(results)
    in_count, out_count = line_counter.update(results)
    frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    in_objects, out_objects = region_timer.update(results, frame_time)
    for obj in in_objects:
        obj['label'] = obj['label'].replace('waiting', 'time')
    frame = render_region(frame, region_timer.region, f"Count: {len(in_objects)}", color=(255, 255, 0))
    frame = render_results(frame, in_objects)
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    video.show(frame)
print(in_count, out_count)
