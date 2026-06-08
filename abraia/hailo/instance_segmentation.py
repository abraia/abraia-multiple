import queue
import threading

import logging
logger = logging.getLogger(__name__)

from types import SimpleNamespace

from ..inference.tracker import Tracker
from .toolbox import (
    VideoInput,
    VideoVisualizer,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    ModelInference
)

from ..utils.draw import render_results

DEFAULT_OPTIONS = {
    "input": "yolov8n_seg.hef",
    "hef_path": None,
    "batch_size": 1,
    "frame_rate": None,
    "model_type": "v8",
    "track": False,
    "labels": None,
    "camera_resolution": None,
    "video_unpaced": False,
    "save_output": None,
}

CONFIG_DATA = {
    "v5": {
        "arch": "yolov5_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "sizes": [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]
        },
        "input_shape": [640, 640],
        "mask_channels": 32,
        "score_threshold": 0.001,
        "nms_iou_thresh": 0.6,
        "classes": 80,
        "layers": [
            [1, 160, 160, "mask_channels"],
            [1, 20, 20, "detection_channels"],
            [1, 40, 40, "detection_channels"],
            [1, 80, 80, "detection_channels"]
        ]
    },
    "v8": {
        "arch": "yolov8_seg",
        "anchors": {
            "strides": [8, 16, 32],
            "regression_length": 15
        },
        "input_shape": [640, 640],
        "mask_channels": 32,
        "score_threshold": 0.001,
        "nms_iou_thresh": 0.7,
        "meta_arch": "yolov8_seg_postprocess",
        "classes": 80,
        "layers": [
            [1, 20, 20, "detection_output_channels"],
            [1, 20, 20, "classes"],
            [1, 20, 20, "mask_channels"],
            [1, 40, 40, "detection_output_channels"],
            [1, 40, 40, "classes"],
            [1, 40, 40, "mask_channels"],
            [1, 80, 80, "detection_output_channels"],
            [1, 80, 80, "classes"],
            [1, 80, 80, "mask_channels"],
            [1, 160, 160, "mask_channels"]
        ]
    },
    "visualization_params": {
        "score_thres": 0.25,
        "mask_thresh": 0.45,
        "mask_alpha": 0.4,
        "max_boxes_to_draw": 50,
        "tracker": {
            "track_thresh": 0.01,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "aspect_ratio_thresh": 2.0,
        }
    }
}


def inference_result_handler(frame, detections, *args, tracker=None, **kwargs):
    """
    This function performs post-processing on the raw model output to extract
    detection results (bounding boxes, masks, classes, scores), applies tracking
    using Tracker, and renders the visualized results (boxes, masks, IDs)
    on top of the original input frame.

    Args:
        frame: The original input image or video frame (as a NumPy array).
        detections: The extracted detections or raw output tensors from the model inference.
        tracker: An instance of Tracker used for object tracking across frames.
    
    Returns:
        np.ndarray: The frame with visualized detection, segmentation, and tracking overlays.
    """
    if tracker:
        detections = track_detections(detections, tracker)
    return render_results(frame, detections)


def track_detections(detections: list, tracker: Tracker) -> list:
    """
    Perform tracking on the detections.

    Args:
        detections (list): List of detection dictionaries.
        tracker (Tracker): Tracker instance.

    Returns:
        list: List of tracked objects (dictionaries with 'label', 'score', 'box', 'mask', and 'track_id').
    """
    return [d for d in tracker.update(detections) if 'track_id' in d]


def run_inference_pipeline(
    model_inference: ModelInference,
    input_data: VideoInput,
    visualizer: VideoVisualizer,
    tracker: Tracker = None,
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
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
            tracker=tracker
        )
    finally:
        input_data.stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    logger.info(visualizer.frame_rate_summary())

    logger.info("Processing completed successfully.")

    if visualizer.save_output:
        logger.info(f"Saved outputs to '{visualizer.save_output}'.")


def main(**kwargs) -> None:
    """
    Main entry point for the instance segmentation application.

    Args:
        **kwargs: Programmatic arguments to override defaults.
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
        video_unpaced=args.video_unpaced,
        stop_event=stop_event,
    )

    visualizer = VideoVisualizer(
        save_output=args.save_output,
        source_fps=input_data.source_fps,
        stop_event=stop_event,
    )
    
    model_inference = ModelInference(
        args.hef_path,
        task='segment',
        batch_size=input_data.batch_size,
        labels=args.labels,
        config_data=CONFIG_DATA,
        model_type=args.model_type
    )

    tracker = None
    if args.track:
        tracker_config = CONFIG_DATA.get("tracker", {})
        tracker = Tracker(
            track_thresh=tracker_config.get("track_thresh", 0.25),
            track_buffer=tracker_config.get("track_buffer", 30),
            match_thresh=tracker_config.get("match_thresh", 0.8),
            frame_rate=input_data.source_fps or 30.0
        )

    run_inference_pipeline(
        model_inference=model_inference,
        input_data=input_data,
        visualizer=visualizer,
        tracker=tracker,
    )


if __name__ == "__main__":
    main()
