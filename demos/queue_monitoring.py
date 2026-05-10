import time
import psutil
import argparse

from abraia.utils import Video
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter, RegionFilter, RegionTimer
from abraia.utils.draw import render_results, render_counter, render_region, draw_text
from abraia.inference.ops import count_objects


parser = argparse.ArgumentParser(description='Abraia SDK monitoring demo')
parser.add_argument('--mode', type=str, choices=['people', 'queue', 'escalator'], help='Monitoring mode: people, queue, or escalator')
args = parser.parse_args()

if args.mode is None:
    parser.print_help()
    exit()

model_uri = 'multiple/models/yolov8n.onnx'
model = Model(model_uri)

if args.mode == 'people':
    src = '853889-hd_1920_1080_25fps.mp4'
    line_counter = LineCounter([(0, 650), (1920, 650)])
    region_filter = RegionFilter([(0, 600), (1920, 600), (1920, 700), (0, 700)])
    region_timer = None
elif args.mode == 'queue':
    src = '4775505-hd_1920_1080_30fps.mp4'
    line_counter = None
    region_filter = None
    region_timer = RegionTimer([(10, 600), (1690, 600), (1690, 700), (10, 700)])
else:
    src = '14393755-hd_1920_1080_30fps.mp4'
    line_counter = LineCounter([(950, 670), (270, 895)])
    region_filter = RegionFilter([[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]])
    region_timer = RegionTimer([[0, 245], [350, 1080], [1200, 1080], [530, 0], [0, 0]])

video = Video(src)
tracker = Tracker(frame_rate=video.frame_rate)

for k, frame in enumerate(video):
    frame_time = round(k / video.frame_rate, 2)
    t0 = time.time()
    results = model.run(frame)
    results = [result for result in results if result['label'] == 'person']
    
    if region_filter:
        results, _ = region_filter.update(results)
    
    results = tracker.update(results)
    
    if line_counter:
        in_count, out_count = line_counter.update(results)
        frame = render_counter(frame, line_counter.line, f"In: {in_count} | Out: {out_count}")
    
    if region_timer:
        in_objects, out_objects = region_timer.update(results, frame_time)
        frame = render_region(frame, region_timer.region, f"Count: {len(in_objects)}", color=(255, 255, 0))
        frame = render_results(frame, in_objects)
    else:
        frame = render_results(frame, results)
    
    t1 = time.time()
    draw_text(frame, f"FPS: {round(1 / (t1 - t0), 1)}", (10, 40))
    draw_text(frame, f"CPU: {psutil.cpu_percent()}%", (10, 70))
    draw_text(frame, f"RAM: {round(psutil.virtual_memory().used / (1024**3), 2)} GB", (10, 100))
    print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
    video.show(frame)

if line_counter:
    print(f"Final In: {in_count}, Final Out: {out_count}")
