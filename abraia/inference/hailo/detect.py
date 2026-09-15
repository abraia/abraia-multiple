import os
import queue
import threading
import logging
from types import SimpleNamespace

from abraia.utils import download_file, load_json

from .toolbox import (
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    HAILO_AVAILABLE,
    ModelInference,
    get_labels,
    default_preprocess
)
from ...runtime.stream import VideoInput, VideoDisplay

from ..tracker import TrackletHistory, Tracker
from ...utils.draw import render_results

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "input": 0,
    "hef_path": None,
    "task": "detect",
    "labels": None,
    "batch_size": 1,
    "score_threshold": 0.25,
    "model_type": None,
    "track": True,
    "draw_trail": False,
    "frame_rate": None,
    "camera_resolution": None,
    "video_unpaced": False,
    "dest": None
}

CONFIG_DATA = {
    "tracker": {
        "track_thresh": 0.1,
        "track_buffer": 30,
        "match_thresh": 0.9
    }
}


def resolve_model_type(model_type, hef_path, task):
    """Resolve the postprocessor architecture for a Hailo model."""
    if model_type:
        return model_type
    if task == "segment" and "yolov8" in str(hef_path).lower():
        return "v8"
    return "v5"


def inference_result_handler(original_frame, detections, tracker=None, tracklet_history=None):
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
    visualizer: VideoDisplay,
    tracker: Tracker = None,
    tracklet_history: TrackletHistory = None,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.
    """
    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    height, width, _ = model_inference.get_input_shape()

    worker_errors = queue.Queue()

    def run_worker(worker, *args):
        try:
            worker(*args)
        except Exception as exc:
            worker_errors.put(exc)
            input_data.stop_event.set()

    preprocess_thread = None
    inference_thread = None
    try:
        preprocess_thread = threading.Thread(
            target=run_worker,
            args=(input_data.preprocess, input_queue, lambda frame: default_preprocess(frame, width, height)),
            name="preprocess-thread",
        )

        inference_thread = threading.Thread(
            target=run_worker,
            args=(model_inference.infer, input_queue, output_queue, input_data.stop_event),
            name="inference-thread",
        )

        preprocess_thread.start()
        inference_thread.start()

        visualizer.visualize(
            output_queue,
            inference_result_handler,
            is_capture=input_data.has_capture,
            tracker=tracker,
            tracklet_history=tracklet_history,
        )
    finally:
        input_data.stop_event.set()
        visualizer.stop_event.set()
        if preprocess_thread is not None:
            preprocess_thread.join()
        if inference_thread is not None:
            inference_thread.join()
        input_data.close()
        visualizer.close()
        model_inference.close()

    if not worker_errors.empty():
        raise worker_errors.get()

    logger.info(visualizer.frame_rate_summary())
    logger.info("Processing completed successfully.")
    if visualizer.dest:
        logger.info(f"Saved outputs to '{visualizer.dest}'.")


def _resolve_model_inputs(model_uri, task, labels_path):
    """Resolve a local or remote HEF and its optional metadata sidecar."""
    if not model_uri:
        raise ValueError("hef_path is required")

    model_uri = os.fspath(model_uri)
    labels = get_labels(labels_path)
    config = None

    if os.path.exists(model_uri):
        hef_path = model_uri
        config_uri = os.path.splitext(model_uri)[0] + ".json"
        if os.path.exists(config_uri):
            config = load_json(config_uri)
    elif model_uri.lower().endswith(".hef") and os.path.dirname(model_uri):
        hef_path = download_file(model_uri)
        config_uri = os.path.splitext(model_uri)[0] + ".json"
        try:
            config = load_json(download_file(config_uri))
        except Exception as exc:
            logger.info("No usable Hailo metadata sidecar for %s: %s", model_uri, exc)
    else:
        # Bare names such as ``yolov8n`` are resolved by ModelInference's
        # Hailo resource catalog; they are not API file paths.
        hef_path = model_uri

    if isinstance(config, dict):
        task = config.get("task", task)
        labels = config.get("classes", labels)
    return hef_path, task, labels


def main(**kwargs) -> None:
    """
    Main entry point for the object detection application.

    Args:
        **kwargs: Programmatic arguments to override defaults.

    Example:
        from abraia.inference.hailo import detect as object_detection
        object_detection.main(input='video.mp4', track=True)
    """
    options = DEFAULT_OPTIONS.copy()
    options.update(kwargs)
    args = SimpleNamespace(**options)
    logging.basicConfig(level=logging.INFO)

    if not HAILO_AVAILABLE:
        raise RuntimeError(
            "Hailo support is unavailable; install the hailo_platform package "
            "on a Hailo-enabled host"
        )

    hef_path, task, labels = _resolve_model_inputs(
        args.hef_path, args.task, args.labels
    )

    model_type = resolve_model_type(args.model_type, hef_path, task)

    stop_event = threading.Event()

    input_data = VideoInput(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        video_unpaced=args.video_unpaced,
        stop_event=stop_event
    )

    visualizer = VideoDisplay(
        dest=args.dest,
        source_fps=input_data.source_fps,
        stop_event=stop_event
    )

    model_inference = None
    try:
        model_inference = ModelInference(
            hef_path, task, labels,
            batch_size=input_data.batch_size,
            score_threshold=args.score_threshold,
            model_type=model_type
        )

        tracker = None
        tracklet_history = None
        if args.track:
            tracker_config = CONFIG_DATA.get("tracker", {})
            tracker = Tracker(
                track_thresh=tracker_config.get('track_thresh', 0.1),
                track_buffer=tracker_config.get('track_buffer', 30),
                match_thresh=tracker_config.get('match_thresh', 0.9),
                frame_rate=input_data.source_fps or 30.0
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
    finally:
        input_data.close()
        visualizer.close()
        if model_inference is not None:
            model_inference.close()


if __name__ == "__main__":
    main()
