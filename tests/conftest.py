import pytest


@pytest.fixture
def public_demo_configs():
    """Small representative demo documents for mocked cloud responses."""
    models = {
        "detect": ("yolov8", "detection", "multiple/models/yolov8n.onnx"),
        "tomato": ("yolov8", "detection", "multiple/tomato/yolov8n.onnx"),
        "apple": ("yolov8", "segmentation", "multiple/models/yolov8n-seg.onnx"),
        "people": ("yolov8", "detection", "multiple/models/yolov8n.onnx"),
        "segment": ("yolov8", "segmentation", "multiple/models/yolov8n-seg.onnx"),
        "pose": ("yolov8", "pose", "multiple/models/yolov8m_pose.onnx"),
        "depth": ("yolo26_depth", "depth", "multiple/models/yolo26n-depth.onnx"),
        "yolo9": ("yolo9", "detection", "multiple/models/yolo9t.onnx"),
    }
    return {
        name: {
            "version": 2,
            "source": (
                {"type": "camera", "src": 0} if name in ("detect", "pose", "depth")
                else {"type": "video", "src": f"multiple/videos/{name}.mp4"}
            ),
            "steps": [{
                "type": "model",
                "model": {"kind": kind, "task": task, "size": "small", "uri": uri},
            }],
        }
        for name, (kind, task, uri) in models.items()
    }
