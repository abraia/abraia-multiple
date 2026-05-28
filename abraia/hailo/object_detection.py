import queue
import threading
import collections
import numpy as np

from functools import partial
from types import SimpleNamespace
from pathlib import Path

from .core import (
    handle_and_resolve_args,
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS,
)
import logging
logger = logging.getLogger(__name__)

from .hailo_inference import HailoInfer
from .toolbox import (
    VisualizationSettings,
    init_input_source,
    get_labels,
    preprocess,
    visualize,
    FrameRateTracker
)
from .tracker.byte_tracker import BYTETracker
from .tracker.matching import find_best_matching_detection_index

APP_NAME = Path(__file__).stem

# Dictionary to store a limited history of tracklet coordinates.
# The keys will be the track IDs.
tracklet_history = {}
# Maximum number of past frames to display
trail_length = 30 

DEFAULT_OPTIONS = {
    "input": "rpi",
    "hef_path": "yolov8m.hef",
    "batch_size": 1,
    "frame_rate": None,
    "track": True,
    "labels": None,
    "draw_trail": False,
    "camera_resolution": None,
    "video_unpaced": False,
    "output_resolution": None,
    "output_dir": None,
    "save_output": False,
}

CONFIG_DATA = {
    "visualization_params": {
        "score_thres": 0.25,
        "max_boxes_to_draw": 500,
        "tracker": {
            "track_thresh": 0.1,
            "track_buffer": 30,
            "match_thresh": 0.9,
            "aspect_ratio_thresh": 2.0,
            "min_box_area": 500,
            "mot20": False
        }
    }
}


def inference_result_handler(original_frame, infer_results, labels, score_threshold, max_boxes, tracker=None, draw_trail=False):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        original_frame (np.ndarray): Original image frame.
        infer_results (list): Raw output from the model.
        labels (list): List of class labels.
        score_threshold (float): Minimum confidence score to consider a detection.
        max_boxes (int): Maximum number of detections to keep.
        tracker (BYTETracker, optional): ByteTrack tracker instance.
        draw_trail (bool): Whether to draw tracking trails.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    infer_results = infer_results if isinstance(infer_results, list) else [infer_results]
    detections = extract_detections(original_frame, infer_results, score_threshold, max_boxes)
    return draw_detections(detections, original_frame, labels, tracker=tracker, draw_trail=draw_trail)


from ..utils.draw import (
    draw_rectangle, draw_text, calculate_optimal_thickness, calculate_optimal_text_scale,
    draw_line, draw_point, get_color, hex_to_rgb
)


def extract_detections(image: np.ndarray, detections: list, score_threshold: float, max_boxes: int) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        score_threshold (float): Minimum confidence score to consider a detection.
        max_boxes (int): Maximum number of detections to keep.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []
    print(f"Raw detections: {detections}")

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                # Denormalize and remove padding
                box = [int(x * size) for x in bbox]
                for i in range(4):
                    if i % 2 == 0:  # x-coordinates
                        if img_height != size:
                            box[i] -= padding_length
                    else:  # y-coordinates
                        if img_width != size:
                            box[i] -= padding_length
                # Swap to [ymin, xmin, ymax, xmax]
                denorm_bbox = [box[1], box[0], box[3], box[2]]
                all_detections.append((score, class_id, denorm_bbox))

    # Sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    # Take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None, draw_trail=False) -> np.ndarray:
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """
    thickness = calculate_optimal_thickness(img_out.shape[:2])
    text_scale = calculate_optimal_text_scale(img_out.shape[:2])

    # Extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [xmin,ymin,xmaxm, ymax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        # Convert detection format to [xmin,ymin,xmaxm ymax,score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  # [x, y, w, h]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        # Skip tracking if no detections passed
        if not dets_for_tracker:
            return img_out

        # Run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))

        # Draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  # Unique tracker ID
            x1, y1, x2, y2 = track.tlbr  # Bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            color = hex_to_rgb(get_color(track_id))  # Color based on class
            
            # Draw bounding box
            draw_rectangle(img_out, [xmin, ymin, xmax - xmin, ymax - ymin], color, thickness)

            # Draw label
            top_text = f"{labels[classes[best_idx]]}: {track.score * 100.0:.1f}%"
            draw_text(img_out, top_text, (xmin, ymin), background_color=color, text_scale=text_scale)
            draw_text(img_out, f"ID {track_id}", (xmax - 50, ymax), background_color=color, text_scale=text_scale)

            # Get the centroid of the current bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centroid = (center_x, center_y)
            
            # Initialize or update the tracklet history
            if track_id not in tracklet_history:
                tracklet_history[track_id] = collections.deque(maxlen=trail_length)
            tracklet_history[track_id].append(centroid)

            if draw_trail:
                for i in range(1, len(tracklet_history[track_id])):
                    # Get the center point for the current and previous frames
                    point_a = tracklet_history[track_id][i-1]
                    point_b = tracklet_history[track_id][i]

                    # Draw a line between the points and draw the points as circles
                    draw_line(img_out, (point_a, point_b), color, thickness=3)
                    draw_point(img_out, point_b, color, thickness=20)
    else:
        # No tracking — draw raw model detections
        for idx in range(num_detections):
            color = hex_to_rgb(get_color(classes[idx]))  # Color based on class
            xmin, ymin, xmax, ymax = map(int, boxes[idx])
            draw_rectangle(img_out, [xmin, ymin, xmax - xmin, ymax - ymin], color, thickness)
            top_text = f"{labels[classes[idx]]}: {scores[idx] * 100.0:.1f}%"
            draw_text(img_out, top_text, (xmin, ymin), background_color=color, text_scale=text_scale)

    return img_out


def run_inference_pipeline(
    net,
    labels,
    input_context,
    visualization_settings: VisualizationSettings,
    enable_tracking: bool = False,
    draw_trail: bool = False,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.
    """
    labels = get_labels(labels)
    config_data = CONFIG_DATA

    stop_event = threading.Event()
    fps_tracker = FrameRateTracker()
    tracker = None

    visualization_params = config_data.get("visualization_params", {})
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    if enable_tracking:
        tracker_config = visualization_params.get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    post_process_callback_fn = partial(
        inference_result_handler,
        labels=labels,
        score_threshold=score_threshold,
        max_boxes=max_boxes,
        tracker=tracker,
        draw_trail=draw_trail,
    )

    hailo_inference = HailoInfer(net, input_context.batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(
            input_context,
            input_queue,
            width,
            height,
            None,  # Use default preprocess from toolbox
            stop_event,
        ),
        name="preprocess-thread",
    )

    infer_thread = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue, stop_event),
        name="infer-thread",
    )

    preprocess_thread.start()
    infer_thread.start()

    fps_tracker.start()

    try:
        visualize(
            input_context,
            visualization_settings,
            output_queue,
            post_process_callback_fn,
            fps_tracker,
            stop_event,
        )
    finally:
        stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    logger.info(fps_tracker.frame_rate_summary())
    logger.info("Processing completed successfully.")

    if visualization_settings.save_stream_output or input_context.has_images:
        logger.info(f"Saved outputs to '{visualization_settings.output_dir}'.")


def infer(hailo_inference, input_queue, output_queue, stop_event):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    # Limit number of concurrent async inferences
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        if stop_event.is_set():
            continue  # Skip processing if stop signal is set

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )


        while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
            pending_jobs.popleft().wait(10000)

        # Run async inference
        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    # Release resources and context
    hailo_inference.close()
    output_queue.put(None)


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))


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
    handle_and_resolve_args(args, APP_NAME)

    input_context = init_input_source(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        video_unpaced=args.video_unpaced,
    )

    visualization_settings = VisualizationSettings(
        output_dir=args.output_dir,
        save_stream_output=args.save_output,
        output_resolution=args.output_resolution,
    )

    run_inference_pipeline(
        net=args.hef_path,
        labels=args.labels,
        input_context=input_context,
        visualization_settings=visualization_settings,
        enable_tracking=args.track,
        draw_trail=args.draw_trail,
    )


if __name__ == "__main__":
    main()
