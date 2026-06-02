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

from .tracker.byte_tracker import BYTETracker
from .tracker.matching import find_best_matching_detection_index

from ..utils.draw import (
    draw_rectangle, draw_text, calculate_optimal_thickness, calculate_optimal_text_scale,
    draw_line, draw_point, get_color, hex_to_rgb
)

APP_NAME = Path(__file__).stem

# Dictionary to store a limited history of tracklet coordinates.
# The keys will be the track IDs.
tracklet_history = {}
# Maximum number of past frames to display
trail_length = 30 

DEFAULT_OPTIONS = {
    "input": 0,
    "hef_path": "yolov8m.hef",
    "batch_size": 1,
    "frame_rate": None,
    "track": True,
    "labels": None,
    "draw_trail": False,
    "camera_resolution": None,
    "video_unpaced": False,
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
    detections = extract_detections(original_frame, infer_results, labels, score_threshold, max_boxes)
    if tracker:
        detections = track_detections(detections, tracker)
        update_trails(detections)
    return draw_detections(original_frame, detections, draw_trail=draw_trail)


def extract_detections(image: np.ndarray, detections: list, labels: list, score_threshold: float, max_boxes: int) -> list:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        labels (list): List of class labels.
        score_threshold (float): Minimum confidence score to consider a detection.
        max_boxes (int): Maximum number of detections to keep.

    Returns:
        list: Filtered detection results containing dictionaries with 'label', 'score', and 'box'.
    """

    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

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
                # Swap to [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box[1], box[0], box[3], box[2]
                all_detections.append({
                    'label': labels[class_id] if labels else str(class_id),
                    'score': float(score),
                    'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'class_id': class_id
                })

    # Sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x['score'])

    return all_detections[:max_boxes]


def track_detections(detections: list, tracker: BYTETracker) -> list:
    """
    Perform tracking on the detections.

    Args:
        detections (list): List of detection dictionaries.
        tracker (BYTETracker): ByteTrack tracker instance.

    Returns:
        list: List of tracked objects (dictionaries with 'label', 'score', 'box', and 'track_id').
    """
    dets_for_tracker = []
    for det in detections:
        x, y, w, h = det['box']
        dets_for_tracker.append([x, y, x + w, y + h, det['score']])

    if not dets_for_tracker:
        return []

    online_targets = tracker.update(np.array(dets_for_tracker))
    tracked_detections = []

    for track in online_targets:
        x1, y1, x2, y2 = track.tlbr
        xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
        
        # Use the format for boxes when matching
        det_boxes = [[d['box'][0], d['box'][1], d['box'][0]+d['box'][2], d['box'][1]+d['box'][3]] for d in detections]
        best_idx = find_best_matching_detection_index(track.tlbr, det_boxes)
        
        if best_idx is not None:
            tracked_detections.append({
                'label': detections[best_idx]['label'],
                'score': float(track.score),
                'box': [xmin, ymin, xmax - xmin, ymax - ymin],
                'track_id': track.track_id
            })
    
    return tracked_detections


def update_trails(detections: list) -> None:
    """
    Update the tracklet history for the detections.

    Args:
        detections (list): List of detection dictionaries.
    """
    for det in detections:
        track_id = det.get('track_id')
        if track_id is not None:
            x, y, w, h = det['box']
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            centroid = (center_x, center_y)
            
            if track_id not in tracklet_history:
                tracklet_history[track_id] = collections.deque(maxlen=trail_length)
            tracklet_history[track_id].append(centroid)
            det['trail'] = list(tracklet_history[track_id])


def draw_detections(image: np.ndarray, detections: list, draw_trail=False) -> np.ndarray:
    """
    Draw detections or tracking results on the image.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): List of detection dictionaries.
        draw_trail (bool): Whether to draw tracking trails.

    Returns:
        np.ndarray: Annotated image.
    """
    img_out = image.copy()
    thickness = calculate_optimal_thickness(img_out.shape[:2])
    text_scale = calculate_optimal_text_scale(img_out.shape[:2])

    for det in detections:
        x, y, w, h = det['box']
        track_id = det.get('track_id')
        color = hex_to_rgb(get_color(track_id if track_id is not None else det.get('class_id', 0)))
        top_text = f"{det['label']} {round(det['score'], 2)}"
        if track_id is not None:
            top_text = f"[{track_id}] {top_text}"

        draw_rectangle(img_out, [x, y, w, h], color, thickness)
        draw_text(img_out, top_text, (x, y), background_color=color, text_scale=text_scale)

        if draw_trail and 'trail' in det:
            trail = det['trail']
            for i in range(1, len(trail)):
                point_a, point_b = trail[i-1], trail[i]
                draw_line(img_out, (point_a, point_b), color, thickness=3)
                draw_point(img_out, point_b, color, thickness=20)

    return img_out


def inference(hailo_inference, input_queue, output_queue):
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
                    name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))


def run_inference_pipeline(
    net,
    labels,
    input_data: VideoInput,
    visualizer: VideoVisualizer,
    enable_tracking: bool = False,
    draw_trail: bool = False,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.
    """
    visualization_params = CONFIG_DATA.get("visualization_params", {})
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    tracker = None
    if enable_tracking:
        tracker_config = visualization_params.get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    hailo_inference = HailoInfer(net, input_data.batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        inference_result_handler,
        labels=labels,
        score_threshold=score_threshold,
        max_boxes=max_boxes,
        tracker=tracker,
        draw_trail=draw_trail,
    )

    try:
        preprocess_thread = threading.Thread(
            target=input_data.preprocess,
            args=(input_queue, width, height),
            name="preprocess-thread",
        )

        infer_thread = threading.Thread(
            target=inference,
            args=(hailo_inference, input_queue, output_queue),
            name="infer-thread",
        )

        visualizer.visualize(
            output_queue,
            post_process_callback_fn,
            is_capture=input_data.has_capture
        )

        preprocess_thread.start()
        infer_thread.start()
    finally:
        input_data.stop_event.set()
        preprocess_thread.join()
        infer_thread.join()

    logger.info(visualizer.frame_rate_summary())

    if visualizer.save_output or input_data.has_images:
        logger.info(f"Saved outputs to '{visualizer.output_dir}'.")


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
        video_unpaced=args.video_unpaced,
        stop_event=stop_event,
    )

    visualizer = VideoVisualizer(
        output_dir=args.output_dir,
        save_output=args.save_output,
        width=input_data.width,
        height=input_data.height,
        source_fps=input_data.source_fps,
        stop_event=stop_event,
    )

    run_inference_pipeline(
        net=hef_path,
        labels=labels,
        input_data=input_data,
        visualizer=visualizer,
        enable_tracking=args.track,
        draw_trail=args.draw_trail,
    )


if __name__ == "__main__":
    main()
