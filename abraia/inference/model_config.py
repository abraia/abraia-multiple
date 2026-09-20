"""Built-in model kinds, defaults, and runtime option definitions."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from ..tasks import HAILO_TASKS, PIPELINE_TASKS, normalize_task


MODEL_ARCHITECTURES = frozenset({"yolov5", "yolov8", "yolo11"})
GROUNDING_DINO_MODEL_KINDS = frozenset({"grounding_dino"})
RESNET_MODEL_KINDS = frozenset({"resnet"})
MODEL_RUN_OPTIONS = (
    "labels",
    "conf_threshold",
    "iou_threshold",
    "approx",
    "top_k",
    "score_threshold",
    "prompt",
    "text_prompt",
    "box_threshold",
    "text_threshold",
)


MODEL_SIZE_URIS = {
    "yolov8": {
        "detection": {
            "small": "multiple/models/yolov8n.onnx",
            "medium": "multiple/models/yolov8m.onnx",
            "large": "multiple/models/yolov8l.onnx",
        },
        "segmentation": {
            "small": "multiple/models/yolov8n-seg.onnx",
            "medium": "multiple/models/yolov8m-seg.onnx",
            "large": "multiple/models/yolov8l-seg.onnx",
        },
        "pose": {
            "small": "multiple/models/yolov8n_pose.onnx",
            "medium": "multiple/models/yolov8m_pose.onnx",
            "large": "multiple/models/yolov8l_pose.onnx",
        },
    },
    "resnet": {
        "classification": {
            "small": "multiple/models/resnet18.onnx",
            "medium": "multiple/models/resnet50.onnx",
            "large": "multiple/models/resnet101.onnx",
        },
    },
}

DEFAULT_MODEL_URIS = {
    kind: {
        task: sizes["small"]
        for task, sizes in tasks.items()
    }
    for kind, tasks in MODEL_SIZE_URIS.items()
}

# Public pipeline catalog shared by runtime validation and Studio's editor.
# Keep this next to the model backend definitions so adding a model cannot
# silently leave the configuration editor and runtime out of sync.
PIPELINE_MODEL_KINDS = (
    "yolov5",
    "yolov8",
    "yolo11",
    "resnet",
    "grounding_dino",
    "face",
    "license_plate",
    "ocr",
)
PIPELINE_MODEL_TASKS = PIPELINE_TASKS
PIPELINE_HAILO_TASKS = HAILO_TASKS
MODEL_TASKS_BY_KIND = {
    **{
        kind: frozenset({"detection", "segmentation", "pose"})
        for kind in MODEL_ARCHITECTURES
    },
    "resnet": frozenset({"classification"}),
    "grounding_dino": frozenset({"detection"}),
    "face": frozenset({"detection", "recognition"}),
    "license_plate": frozenset({"detection", "recognition"}),
    "ocr": frozenset({"recognition"}),
}
URI_MODEL_KINDS = frozenset((*MODEL_ARCHITECTURES, "resnet", "grounding_dino"))
GROUNDING_DINO_DEFAULT_URI = "multiple/models/grounding_dino_tiny.onnx"
MODEL_DEFAULT_THRESHOLDS = {
    ("yolov8", "detection"): (0.25, 0.45),
    ("yolov8", "segmentation"): (0.25, 0.7),
    ("yolov8", "pose"): (0.25, 0.7),
    ("resnet", "classification"): (0.25, 0.0),
    ("grounding_dino", "detection"): (0.25, 0.45),
    ("face", "detection"): (0.75, 0.5),
    ("face", "recognition"): (0.45, 0.5),
    ("license_plate", "detection"): (0.5, 0.1),
    ("license_plate", "recognition"): (0.85, 0.15),
    ("ocr", "recognition"): (0.5, 0.0),
}


@dataclass(frozen=True)
class ModelSpec:
    """Normalized model configuration shared by Studio and runtime code.

    The public pipeline document remains a dictionary, but all consumers use
    this value object for canonical kind/task names, model defaults, backend
    detection, and compatibility checks.
    """

    kind: str
    task: str
    uri: str = ""
    size: str = "small"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModelSpec":
        """Normalize a model section from a pipeline document."""
        if not isinstance(config, Mapping):
            raise ValueError("Pipeline model must be an object")

        kind = str(config.get("kind", "")).strip().lower()
        task = normalize_task(
            config.get("task"),
            default="classification" if kind == "resnet" else "detection",
        )
        raw_uri = config.get("uri", "")
        if raw_uri is None:
            uri = ""
        else:
            try:
                uri = os.fspath(raw_uri).strip()
            except TypeError as error:
                raise ValueError("Pipeline model 'uri' must be a path or string") from error

        raw_params = config.get("params", {}) or {}
        if not isinstance(raw_params, Mapping):
            raise ValueError("Pipeline model 'params' must be an object")

        return cls(
            kind=kind,
            task=task,
            uri=uri,
            size=str(config.get("size", "small") or "small").strip().lower(),
            params=MappingProxyType(dict(raw_params)),
        )

    @property
    def backend(self) -> str:
        """Return the backend implied by the model artifact suffix."""
        if self.uri.split("?", 1)[0].lower().endswith(".hef"):
            return "hailo"
        return "onnx"

    @property
    def allowed_tasks(self):
        """Return canonical tasks supported by this model kind."""
        return MODEL_TASKS_BY_KIND.get(self.kind, frozenset())

    @property
    def uses_uri(self) -> bool:
        """Return whether the model is selected through a model URI."""
        return self.kind in URI_MODEL_KINDS

    @property
    def default_uri(self) -> str:
        """Return the built-in URI for this kind/task/size combination."""
        if self.kind == "grounding_dino" and self.task == "detection":
            return GROUNDING_DINO_DEFAULT_URI
        return model_default_uri(self.kind, self.task, self.size)

    @property
    def resolved_uri(self) -> str:
        """Return the configured URI or its built-in default."""
        return self.uri or self.default_uri

    def pipeline_errors(self) -> Tuple[str, ...]:
        """Return user-facing validation errors for pipeline editors."""
        if self.kind not in PIPELINE_MODEL_KINDS:
            return ("Choose a supported detector kind.",)
        if self.task not in PIPELINE_MODEL_TASKS:
            return ("Choose a supported detector task.",)
        if self.task not in self.allowed_tasks:
            if self.kind in MODEL_ARCHITECTURES:
                return (
                    f"{self.kind} models support detection, segmentation, or pose.",
                )
            if self.kind == "resnet":
                return ("Classification models require the classification task.",)
            if self.kind == "grounding_dino":
                return ("Grounding DINO models require the detection task.",)
            if self.kind == "ocr":
                return ("OCR models only support the recognition task.",)
            return (f"{self.kind} models do not support the {self.task} task.",)
        if self.kind in ("resnet", *MODEL_ARCHITECTURES) and not self.uri:
            return ("Enter a model URI or local model path.",)
        if (
            self.kind == "face"
            and self.task == "recognition"
            and not self.params.get("index")
        ):
            return ("Face recognition requires an index JSON file.",)
        return ()

    def require_runtime_valid(self) -> None:
        """Validate configuration using runtime-facing error messages."""
        if not self.kind:
            raise ValueError("Pipeline model must define a model 'kind'")
        if self.kind not in PIPELINE_MODEL_KINDS:
            available = ", ".join(PIPELINE_MODEL_KINDS)
            raise ValueError(
                f"Unknown detector kind '{self.kind}'. Available detectors: {available}"
            )
        if self.backend == "hailo":
            if self.kind not in MODEL_ARCHITECTURES:
                raise ValueError("Hailo models require a supported model architecture kind")
            if self.task not in HAILO_TASKS:
                raise ValueError(f"Unsupported Hailo pipeline task: {self.task}")
        elif self.task not in self.allowed_tasks:
            if self.kind == "resnet":
                raise ValueError("ResNet models require the classification task")
            if self.kind == "grounding_dino":
                raise ValueError("Grounding DINO models require the detection task")
            if self.kind == "ocr":
                raise ValueError("OCR models only support the recognition task")
            raise ValueError(f"Unsupported pipeline model task: {self.task}")

        if self.kind == "resnet" and not self.uri:
            raise ValueError("A ResNet classifier requires a model 'uri'")
        if self.kind in MODEL_ARCHITECTURES and not self.uri:
            if self.kind not in MODEL_SIZE_URIS:
                raise ValueError(f"A {self.kind} model requires a model 'uri'")
            available_sizes = MODEL_SIZE_URIS[self.kind].get(self.task, {})
            if self.size not in available_sizes:
                options = ", ".join(available_sizes)
                raise ValueError(
                    f"Unsupported model size '{self.size}'. Use: {options}"
                )

    def params_copy(self) -> Dict[str, Any]:
        """Return a mutable copy of backend parameters."""
        return dict(self.params)


def model_defaults(kind, task):
    """Return editor defaults for a model kind/task pair."""
    spec = ModelSpec.from_config({"kind": kind, "task": task})
    lookup_kind = spec.kind
    if lookup_kind in MODEL_ARCHITECTURES and (
        lookup_kind,
        spec.task,
    ) not in MODEL_DEFAULT_THRESHOLDS:
        lookup_kind = "yolov8"
    return MODEL_DEFAULT_THRESHOLDS.get((lookup_kind, spec.task), (0.25, 0.45))


def model_control_visibility(kind, task):
    """Return which pipeline-editor controls apply to a model."""
    spec = ModelSpec.from_config({"kind": kind, "task": task})
    is_model_uri = spec.uses_uri
    return {
        "uri": is_model_uri,
        "index": spec.kind == "face" and spec.task == "recognition",
        "labels": is_model_uri,
        "confidence": True,
        "iou": (
            not (spec.kind == "face" and spec.task == "recognition")
            and spec.kind not in ("ocr", "resnet")
        ),
        "approx": spec.task == "segmentation" and spec.kind in MODEL_ARCHITECTURES,
    }


def _sized_model_options(kind, label, task, uri_kind):
    size_labels = {"small": "Small", "medium": "Medium", "large": "Large"}
    return tuple(
        (
            f"{kind}_{task}_{size}",
            f"{label} ({size_labels[size]})",
            kind,
            task,
            uri,
        )
        for size, uri in MODEL_SIZE_URIS[kind][uri_kind].items()
    )


PIPELINE_MODEL_OPTIONS = (
    *_sized_model_options("yolov8", "Object detection", "detection", "detection"),
    *_sized_model_options("yolov8", "Instance segmentation", "segmentation", "segmentation"),
    *_sized_model_options("yolov8", "Pose estimation", "pose", "pose"),
    *_sized_model_options(
        "resnet", "ResNet classification", "classification", "classification"
    ),
    ("face_detection", "Face detection", "face", "detection", ""),
    ("face_recognition", "Face recognition", "face", "recognition", ""),
    ("license_plate_detection", "License plate detection", "license_plate", "detection", ""),
    ("license_plate_recognition", "License plate recognition", "license_plate", "recognition", ""),
    ("ocr_recognition", "OCR recognition", "ocr", "recognition", ""),
)


def is_resnet_model_uri(uri):
    """Return whether a URI names a model produced by classification training."""
    if not uri:
        return False
    stem = Path(os.fspath(uri)).stem.lower()
    return stem in {"resnet18", "resnet50", "resnet101"}


def model_default_uri(kind, task, size="small"):
    """Return the built-in URI for an architecture/task/size combination."""
    return MODEL_SIZE_URIS.get(kind, {}).get(task, {}).get(size, "")


__all__ = [
    "DEFAULT_MODEL_URIS",
    "GROUNDING_DINO_MODEL_KINDS",
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "MODEL_ARCHITECTURES",
    "MODEL_DEFAULT_THRESHOLDS",
    "MODEL_TASKS_BY_KIND",
    "PIPELINE_HAILO_TASKS",
    "PIPELINE_MODEL_KINDS",
    "PIPELINE_MODEL_OPTIONS",
    "PIPELINE_MODEL_TASKS",
    "RESNET_MODEL_KINDS",
    "ModelSpec",
    "is_resnet_model_uri",
    "model_default_uri",
    "model_control_visibility",
    "model_defaults",
]
