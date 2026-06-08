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

from ..utils.draw import render_results


DEFAULT_OPTIONS = {
    "input": None,
    "hef_path": None,
    "batch_size": 1,
    "frame_rate": None,
    "track": True,
    "labels": None,
    "camera_resolution": None,
    "video_unpaced": False,
    "save_output": None,
}


def run_inference_pipeline(model_inference: ModelInference, input_data: VideoInput, visualizer: VideoVisualizer) -> None:
    """
    Initialize queues, inference instance, and run the pipeline.

    Args:
        model_inference (ModelInference): Model inference instance.
        input_data (VideoInput): Input data source.
        visualizer (VideoVisualizer): Visualizer for output.

    Returns:
        None
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
            render_results,
            is_capture=input_data.has_capture,
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
    Main entry point for the pose estimation application.

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
        frame_rate=args.frame_rate,
        stop_event=stop_event,
    )

    model_inference = ModelInference(args.hef_path, task='pose', batch_size=input_data.batch_size)

    run_inference_pipeline(
        model_inference=model_inference,
        input_data=input_data,
        visualizer=visualizer,
    )


if __name__ == "__main__":
    main()
