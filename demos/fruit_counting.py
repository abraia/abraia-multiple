import os
import time
import psutil
import argparse

from abraia.utils import Video, download_url, show_image
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter
from abraia.utils.draw import render_results, render_counter, draw_text
from abraia.inference.ops import count_objects


FRUITS = {
    'apple': {
        'model': 'multiple/models/yolov8n.onnx',
        'src': '5479199-hd_1920_1080_25fps.mp4'
    },
    'strawberry': {
        'model': 'multiple/strawberry/yolov8n.onnx',
        'src': '9710983-hd_1920_1080_30fps.mp4'
    },
    'grapes': {
        'model': 'multiple/grapes/yolov8n.onnx',
        'src': '5658544-hd_1366_720_24fps.mp4'
    },
    'tomato': {
        'model': 'multiple/tomato/yolov8n_v6.onnx',
        'src': '10179855-hd_1920_1080_30fps.mp4',
        'line': [(1440, 0), (1440, 1080)]
    }
}


parser = argparse.ArgumentParser(description='Abraia SDK fruit counting demo')
parser.add_argument('--fruit', type=str, choices=list(FRUITS.keys()), help=f"Fruit to detect: {', '.join(FRUITS.keys())}")
args = parser.parse_args()

if args.fruit is None:
    parser.print_help()
    exit()

def main():
    config = FRUITS[args.fruit]
    if not os.path.exists(config['src']):
        download_url(f"https://github.com/abraia/abraia-multiple/blob/master/demos/{config['src']}", config['src'])
    model = Model(config['model'])
    video = Video(config['src'])
    tracker = Tracker(frame_rate=video.frame_rate)
    line_counter = LineCounter(config['line']) if 'line' in config else None
    print(f"Detecting {args.fruit} using {config['model']} on {config['src']}...")
    for k, frame in enumerate(video):
        t0 = time.time()
        results = model.run(frame)
        results = [result for result in results if result['label'] == args.fruit]
        results = tracker.update(results)
        out = render_results(frame.copy(), results)
        if line_counter:
            in_count, out_count = line_counter.update(results)
            out = render_counter(out, line_counter.line, f"Count: {out_count}")
        t1 = time.time()
        draw_text(out, f"FPS: {round(1 / (t1 - t0), 1)}", (10, 40))
        draw_text(out, f"CPU: {psutil.cpu_percent()}%", (10, 70))
        draw_text(out, f"RAM: {round(psutil.virtual_memory().used / (1024**3), 2)} GB", (10, 100))
        print(f"#{k}: {round((t1 - t0) * 1000, 1)} ms {count_objects(results)}")
        video.show(out)

    # Automatically segment the fruit from the previously detected boxes in the last frame
    from abraia.inference.detect import segment_objects

    out = frame.copy()
    results = segment_objects(frame, results)
    out = render_results(out, results)
    show_image(out)


if __name__ == '__main__':
    main()
