import time
import click

from abraia.utils import Video, show_image
from abraia.inference import Model, Tracker
from abraia.inference.tools import LineCounter
from abraia.utils.draw import render_results, render_counter
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


@click.command()
@click.option('--fruit', type=click.Choice(FRUITS.keys()), default='apple', help='Fruit to detect.')
def main(fruit):
    config = FRUITS[fruit]
    model = Model(config['model'])
    video = Video(config['src'])
    tracker = Tracker(frame_rate=video.frame_rate)
    line_counter = LineCounter(config['line']) if 'line' in config else None
    print(f"Detecting {fruit} using {config['model']} on {config['src']}...")
    for k, frame in enumerate(video):
        frame_time = round(k / video.frame_rate, 2)
        t0 = time.time()
        results = model.run(frame)
        results = [result for result in results if result['label'] == fruit]
        results = tracker.update(results)
        out = render_results(frame.copy(), results)
        if line_counter:
            in_count, out_count = line_counter.update(results)
            out = render_counter(out, line_counter.line, f"Count: {out_count}")
        t1 = time.time()
        print(f"#{k} [{frame_time}s] {count_objects(results)} {round((t1 - t0) * 1000, 1)}ms")
        video.show(out)

    # Automatically segment the fruit from the previously detected boxes in the last frame
    from abraia.inference.detect import segment_objects

    out = frame.copy()
    results = segment_objects(frame, results)
    out = render_results(out, results)
    show_image(out)


if __name__ == '__main__':
    main()
