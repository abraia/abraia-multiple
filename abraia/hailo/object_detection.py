import queue
import threading
import collections
import numpy as np

from functools import partial
from types import SimpleNamespace
from pathlib import Path

from .toolbox import (
    VideoInput,
    VideoVisualizer,
    get_labels,
    resolve_hef_path,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS,
    HailoInfer,
)
import logging
logger = logging.getLogger(__name__)

from ..inference.tracker import TrackletHistory, Tracker
from ..utils.draw import render_results

APP_NAME = Path(__file__).stem

DEFAULT_OPTIONS = {
    "input": 0,
    "hef_path": "yolov8m.hef",
    "batch_size": 1,
    "score_threshold": 0.25,
    "frame_rate": None,
    "track": True,
    "labels": None,
    "draw_trail": False,
    "camera_resolution": None,
    "output_dir": None,
    "save_output": False,
}

CONFIG_DATA = {
    "tracker": {
        "track_thresh": 0.1,
        "track_buffer": 30,
        "match_thresh": 0.9,
        "aspect_ratio_thresh": 2.0,
    }
}


class ModelInference(HailoInfer):
    def __init__(self, hef_path: str, batch_size: int = 1, labels: list = None, score_threshold: float = 0.25):
        super().__init__(hef_path, batch_size)
        self.score_threshold = score_threshold
        self.labels = labels

    def _inference_callback(self, completion_info, bindings_list: list, input_batch: list, output_queue: queue.Queue) -> None:
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                        for name in bindings._output_names
                    }
                
                infer_results = result if isinstance(result, list) else [result]
                
                image = input_batch[i]
                img_height, img_width = image.shape[:2]
                size = max(img_height, img_width)
                padding_length = int(abs(img_height - img_width) / 2)

                detections = []
                for class_id, detection in enumerate(infer_results):
                    for det in detection:
                        bbox, score = det[:4], det[4]
                        if score >= self.score_threshold:
                            # Denormalize and remove padding
                            box = [int(x * size) for x in bbox]
                            for j in range(4):
                                if j % 2 == 0:  # x-coordinates
                                    if img_height != size:
                                        box[j] -= padding_length
                                else:  # y-coordinates
                                    if img_width != size:
                                        box[j] -= padding_length
                            # Swap to [xmin, ymin, xmax, ymax]
                            xmin, ymin, xmax, ymax = box[1], box[0], box[3], box[2]
                            detections.append({
                                'label': self.labels[class_id] if self.labels else str(class_id),
                                'score': float(score),
                                'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                                'class_id': class_id
                            })

                output_queue.put((input_batch[i], detections))

    def infer(self, input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
        """
        Main inference loop that pulls data from the input queue, runs asynchronous
        inference, and pushes results to the output queue.

        Each item in the input queue is expected to be a tuple:
            (input_batch, preprocessed_batch)
            - input_batch: Original frames (used for visualization or tracking)
            - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

        Args:
            input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
            output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.
            stop_event (threading.Event): Event to signal stopping the inference loop.
        """
        pending_jobs = collections.deque()

        while True:
            next_batch = input_queue.get()
            if not next_batch:
                break

            if stop_event.is_set():
                continue

            input_batch, preprocessed_batch = next_batch

            inference_callback_fn = partial(
                self._inference_callback,
                input_batch=input_batch,
                output_queue=output_queue
            )

            while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
                pending_jobs.popleft().wait(10000)

            job = self.run(preprocessed_batch, inference_callback_fn)
            pending_jobs.append(job)

        self.close()
        output_queue.put(None)


def inference_result_handler(original_frame, detections, *args, tracker=None, tracklet_history=None, **kwargs):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        original_frame (np.ndarray): Original image frame.
        detections (list): Extracted detections.
        tracker (Tracker, optional): Tracker instance.
        tracklet_history (TrackletHistory, optional): Tracklet history instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    if tracker:
        detections = tracker.update(detections)
        if tracklet_history:
            tracklet_history.update(detections)
    return render_results(original_frame, detections)


def run_inference_pipeline(
    model_inference: ModelInference,
    input_data: VideoInput,
    visualizer: VideoVisualizer,
    tracker: Tracker = None,
    tracklet_history: TrackletHistory = None,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.
    """
    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    height, width, _ = model_inference.get_input_shape()

    try:
        preprocess_thread = threading.Thread(
            target=input_data.preprocess,
            args=(input_queue, width, height),
            name="preprocess-thread",
        )

        infer_thread = threading.Thread(
            target=model_inference.infer,
            args=(input_queue, output_queue, input_data.stop_event),
            name="infer-thread",
        )

        preprocess_thread.start()
        infer_thread.start()

        visualizer.visualize(
            output_queue,
            inference_result_handler,
            is_capture=input_data.has_capture,
            tracker=tracker,
            tracklet_history=tracklet_history,
        )
    finally:
        input_data.stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    logger.info(visualizer.frame_rate_summary())
    logger.info("Processing completed successfully.")


def main(**kwargs) -> None:
    """
    Main entry point for the object detection application.

    Args:
        **kwargs: Programmatic arguments to override defaults.

    Example:
        from abraia.hailo import object_detection
        object_detection.main(input='video.mp4', track=True)
    """
    options = DEFAULT_OPTIONS.copy()
    options.update(kwargs)
    args = SimpleNamespace(**options)
    
    logging.basicConfig(level=logging.INFO)
    hef_path = resolve_hef_path(args.hef_path, APP_NAME)
    labels = get_labels(args.labels)

    stop_event = threading.Event()

    input_data = VideoInput(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        stop_event=stop_event
    )

    visualizer = VideoVisualizer(
        output_dir=args.output_dir,
        save_output=args.save_output,
        source_fps=input_data.source_fps,
        frame_rate=args.frame_rate,
        stop_event=stop_event
    )

    model_inference = ModelInference(
        hef_path,
        labels=labels,
        batch_size=input_data.batch_size,
        score_threshold=args.score_threshold
    )

    tracker = None
    tracklet_history = None
    if args.track:
        tracker_config = CONFIG_DATA.get("tracker", {})
        tracker = Tracker(
            track_thresh=tracker_config.get('track_thresh', 0.1),
            track_buffer=tracker_config.get('track_buffer', 30),
            match_thresh=tracker_config.get('match_thresh', 0.9),
            frame_rate=args.frame_rate or 30
        )
        if args.draw_trail:
            tracklet_history = TrackletHistory()

    run_inference_pipeline(
        model_inference=model_inference,
        input_data=input_data,
        visualizer=visualizer,
        tracker=tracker,
        tracklet_history=tracklet_history,
    )


if __name__ == "__main__":
    main()
