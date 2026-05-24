from __future__ import annotations
#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp
from queue import Queue
from functools import partial
import numpy as np
import threading
from pathlib import Path
from .pose_estimation_utils import PoseEstPostProcessing
import collections
from types import SimpleNamespace



from .hailo_logger import get_logger, init_logging
from .hailo_inference import HailoInfer
from .core import handle_and_resolve_args
from .toolbox import (
    InputContext,
    VisualizationSettings,
    init_input_source,
    preprocess,
    visualize,
    FrameRateTracker
)
from .defines import (
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS
)

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)

DEFAULT_OPTIONS = {
    "input": None,
    "hef_path": None,
    "list_models": False,
    "batch_size": 1,
    "show_fps": False,
    "frame_rate": None,
    "class_num": 1,
    "camera_resolution": None,
    "video_unpaced": False,
    "output_resolution": None,
    "output_dir": None,
    "save_output": False,
}

def inference_callback(
        completion_info,
        bindings_list: list,
        input_batch: list,
        output_queue: mp.Queue
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


def run_inference_pipeline(
    net,
    class_num,
    input_context: InputContext,
    visualization_settings: VisualizationSettings,
    show_fps: bool = False,
) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.

    Args:
        net (str): Path to the HEF model file.
        class_num (int): Number of output classes expected by the model.
        input_context (InputContext): Context containing input source details.
        visualization_settings (VisualizationSettings): Settings for visualization.
        show_fps (bool): If True, display real-time FPS on the output.

    Returns:
        None
    """

    input_queue = Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = Queue(MAX_OUTPUT_QUEUE_SIZE)


    pose_post_processing = PoseEstPostProcessing(
        max_detections=300,
        score_threshold=0.001,
        nms_iou_thresh=0.7,
        regression_length=15,
        strides=[8, 16, 32]
    )

    stop_event = threading.Event()
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    hailo_inference = HailoInfer(net, input_context.batch_size, output_type="FLOAT32")
    height, width, _ = hailo_inference.get_input_shape()

    post_process_callback_fn = partial(
        pose_post_processing.inference_result_handler,
        model_height=height,
        model_width=width,
        class_num = class_num
    )

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(
            input_context,
            input_queue,
            width,
            height,
            None,
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

    if show_fps:
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

    if show_fps:
        logger.info(fps_tracker.frame_rate_summary())

    logger.success("Processing completed successfully.")

    if visualization_settings.save_stream_output or input_context.has_images:
        logger.info(f"Saved outputs to '{visualization_settings.output_dir}'.")




def main(**kwargs) -> None:
    """
    Main entry point for the pose estimation application.

    Args:
        **kwargs: Programmatic arguments to override defaults.
    """
    options = DEFAULT_OPTIONS.copy()
    options.update(kwargs)
    args = SimpleNamespace(**options)

    init_logging()
    handle_and_resolve_args(args, APP_NAME)

    input_context = InputContext(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
        video_unpaced=args.video_unpaced,
    )

    input_context = init_input_source(input_context)

    visualization_settings = VisualizationSettings(
        output_dir=args.output_dir,
        save_stream_output=args.save_output,
        output_resolution=args.output_resolution,
    )

    run_inference_pipeline(
        net=args.hef_path,
        class_num=args.class_num,
        input_context=input_context,
        visualization_settings=visualization_settings,
        show_fps=args.show_fps,
    )


if __name__ == "__main__":
    main()
