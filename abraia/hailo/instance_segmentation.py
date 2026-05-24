from __future__ import annotations
#!/usr/bin/env python3
import os
import sys
import warnings

def apply_runtime_compatibility() -> None:
    """Apply minimal runtime compatibility fixes before third-party imports."""
    _suppress_known_future_warnings()
    _ensure_stdlib_distutils_for_older_python()


def _suppress_known_future_warnings() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\.bool.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\.bytes.*")


def _ensure_stdlib_distutils_for_older_python() -> None:
    if sys.version_info >= (3, 12):
        return

    if os.environ.get("SETUPTOOLS_USE_DISTUTILS") == "stdlib":
        return

    env = os.environ.copy()
    env["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

    try:
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
    except Exception as exc:
        raise RuntimeError(
            "Failed to restart process with SETUPTOOLS_USE_DISTUTILS=stdlib"
        ) from exc


apply_runtime_compatibility()

import queue
import threading
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import numpy as np
import collections
from .instance_segmentation_post_process import inference_result_handler


from .tracker.byte_tracker import BYTETracker
from .hailo_inference import HailoInfer
from .toolbox import (
    InputContext,
    VisualizationSettings,
    init_input_source,
    load_json_file,
    get_labels,
    visualize,
    preprocess,
    FrameRateTracker
)
from .defines import (
    MAX_INPUT_QUEUE_SIZE,
    MAX_OUTPUT_QUEUE_SIZE,
    MAX_ASYNC_INFER_JOBS
)
from .core import handle_and_resolve_args
from .hailo_logger import get_logger, init_logging


APP_NAME = Path(__file__).stem
logger = get_logger(__name__)

DEFAULT_OPTIONS = {
    "input": None,
    "hef_path": None,
    "batch_size": 1,
    "show_fps": False,
    "frame_rate": None,
    "model_type": "v5",
    "track": False,
    "labels": None,
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


def run_inference_pipeline(
    net,
    labels,
    model_type,
    input_context: InputContext,
    visualization_settings: VisualizationSettings,
    enable_tracking=False,
    show_fps=False,
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    app_dir = Path(__file__).resolve().parent
    config_path = app_dir / "instance_segmentation_config.json"
    config_data = load_json_file(str(config_path))
    labels = get_labels(labels)

    stop_event = threading.Event()
    tracker = None
    fps_tracker = None

    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking:
        # Load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    hailo_inference = HailoInfer(
        net,
        input_context.batch_size,
        output_type="FLOAT32")

    post_process_callback_fn = partial(
        inference_result_handler,
        tracker=tracker,
        config_data=config_data,
        model_type=model_type,
        labels=labels,
        nms_postprocess_enabled=hailo_inference.is_nms_postprocess_enabled()
    )

    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(
            input_context,
            input_queue,
            width,
            height,
            None,       # Use default preprocess from toolbox
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



def main(**kwargs) -> None:
    """
    Main entry point for the instance segmentation application.

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
        labels=args.labels,
        model_type=args.model_type,
        input_context=input_context,
        visualization_settings=visualization_settings,
        enable_tracking=args.track,
        show_fps=args.show_fps,
    )


if __name__ == "__main__":
    main()
