import queue
import threading

import logging
logger = logging.getLogger(__name__)

from types import SimpleNamespace

from .toolbox import (
    VideoInput,
    VideoVisualizer,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    ModelInference,
)

from ..inference.tracker import TrackletHistory, Tracker
from ..utils.draw import render_results


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
    "save_output": None,
}

CONFIG_DATA = {
    "tracker": {
        "track_thresh": 0.1,
        "track_buffer": 30,
        "match_thresh": 0.9,
        "aspect_ratio_thresh": 2.0,
    }
}


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

    stop_event = threading.Event()

    input_data = VideoInput(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        stop_event=stop_event
    )

    visualizer = VideoVisualizer(
        save_output=args.save_output,
        source_fps=input_data.source_fps,
        frame_rate=args.frame_rate,
        stop_event=stop_event
    )

    model_inference = ModelInference(
        args.hef_path, task='detect', 
        labels=args.labels, 
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
