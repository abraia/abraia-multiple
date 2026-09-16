import os
import logging
from types import SimpleNamespace

from abraia.utils import download_file, load_json

from ...tasks import normalize_config_task, normalize_task
from .models import get_labels
from .toolbox import HAILO_AVAILABLE
from ...runtime import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS = {
    "input": 0,
    "hef_path": None,
    "task": "detection",
    "labels": None,
    "batch_size": 1,
    "score_threshold": 0.25,
    "model_type": None,
    "track": True,
    "frame_rate": None,
    "camera_resolution": None,
    "video_unpaced": False,
    "dest": None
}

def resolve_model_type(model_type, hef_path, task):
    """Resolve the postprocessor architecture for a Hailo model."""
    if model_type:
        return model_type
    normalized_task = normalize_task(task)
    if normalized_task not in ("detection", "segmentation", "pose"):
        raise ValueError(
            f"Unsupported Hailo task: {normalized_task}. "
            "Use detection, segmentation, or pose"
        )
    if normalized_task == "pose":
        return "v8"
    if normalized_task == "segmentation" and "yolov8" in str(hef_path).lower():
        return "v8"
    return "v5"


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
        task = normalize_config_task(config.get("task"), default=task)
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
    pipeline_task = normalize_task(task)

    config = {
        "version": 1,
        "source": {
            "src": args.input,
            "resolution": args.camera_resolution or (1280, 720),
            "fps": args.frame_rate or 30,
            "video_unpaced": args.video_unpaced,
        },
        "model": {
            "task": pipeline_task,
            "kind": "hailo",
            "uri": hef_path,
            "params": {
                "labels": labels,
                "batch_size": args.batch_size,
                "score_threshold": args.score_threshold,
                "model_type": model_type,
            },
        },
        "stages": [
            {
                "type": "tracker",
                "enabled": args.track,
                "track_thresh": 0.1,
                "track_buffer": 30,
                "match_thresh": 0.9,
            }
        ],
        "display": {"dest": args.dest, "show": True},
    }
    Pipeline.from_dict(config).run()


if __name__ == "__main__":
    main()
