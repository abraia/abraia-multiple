import time

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter
from abraia.utils.draw import render_results, render_counter
from abraia.inference.ops import count_objects


model_uri = 'multiple/tomato/yolov8n_v6.onnx'
model = Model(model_uri)

src = '10179855-hd_1920_1080_30fps.mp4'
video = Video(src) #, dest='tomato-counter.mp4')
tracker = Tracker(frame_rate=video.frame_rate)
line_counter = LineCounter([(270, 895), (950, 670)])
line_counter = LineCounter([(1440, 0), (1440, 1080)])
for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    results = tracker.update(results)
    frame = render_results(frame, results)
    in_count, out_count = line_counter.update(results)
    frame = render_counter(frame, line_counter.line, f"Count: {out_count}")
    t1 = time.time()
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    # video.write(frame)
    video.show(frame)
print(in_count, out_count)
