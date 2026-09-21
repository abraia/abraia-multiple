"""Built-in model kinds, defaults, and runtime option definitions."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from ..tasks import HAILO_TASKS, PIPELINE_TASKS, normalize_task


@dataclass(frozen=True)
class ModelDescriptor:
    """Static capabilities shared by editors, validation, and factories."""

    tasks: FrozenSet[str]
    uri_based: bool = False
    explicit_uri: bool = False
    runtime_options: bool = False


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
MODEL_DESCRIPTORS = MappingProxyType({
    **{
        kind: ModelDescriptor(
            frozenset({"detection", "segmentation", "pose"}),
            uri_based=True,
            runtime_options=True,
        )
        for kind in MODEL_ARCHITECTURES
    },
    "resnet": ModelDescriptor(
        frozenset({"classification"}),
        uri_based=True,
        explicit_uri=True,
        runtime_options=True,
    ),
    "grounding_dino": ModelDescriptor(
        frozenset({"detection"}), uri_based=True, runtime_options=True
    ),
    "face": ModelDescriptor(frozenset({"detection", "recognition"})),
    "license_plate": ModelDescriptor(frozenset({"detection", "recognition"})),
    "ocr": ModelDescriptor(frozenset({"recognition"})),
})
MODEL_TASKS_BY_KIND = {
    kind: descriptor.tasks for kind, descriptor in MODEL_DESCRIPTORS.items()
}
URI_MODEL_KINDS = frozenset(
    kind for kind, descriptor in MODEL_DESCRIPTORS.items() if descriptor.uri_based
)
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
        descriptor = MODEL_DESCRIPTORS.get(self.kind)
        return descriptor.tasks if descriptor else frozenset()

    @property
    def uses_uri(self) -> bool:
        """Return whether the model is selected through a model URI."""
        descriptor = MODEL_DESCRIPTORS.get(self.kind)
        return bool(descriptor and descriptor.uri_based)

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
        return self._validation_messages(runtime=False)

    def _validation_messages(self, runtime: bool) -> Tuple[str, ...]:
        """Return validation messages for either editor or runtime callers."""
        if runtime and not self.kind:
            return ("Pipeline model must define a model 'kind'",)
        if self.kind not in PIPELINE_MODEL_KINDS:
            if runtime:
                available = ", ".join(PIPELINE_MODEL_KINDS)
                return (
                    f"Unknown detector kind '{self.kind}'. Available detectors: {available}",
                )
            return ("Choose a supported detector kind.",)
        if not runtime and self.task not in PIPELINE_MODEL_TASKS:
            return ("Choose a supported detector task.",)
        if runtime and self.backend == "hailo":
            if self.kind not in MODEL_ARCHITECTURES:
                return ("Hailo models require a supported model architecture kind",)
            if self.task not in HAILO_TASKS:
                return (f"Unsupported Hailo pipeline task: {self.task}",)
        elif self.task not in self.allowed_tasks:
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

        if runtime:
            if self.kind == "resnet" and not self.uri:
                return ("A ResNet classifier requires a model 'uri'",)
            if self.kind in MODEL_ARCHITECTURES and not self.uri:
                if self.kind not in MODEL_SIZE_URIS:
                    return (f"A {self.kind} model requires a model 'uri'",)
                available_sizes = MODEL_SIZE_URIS[self.kind].get(self.task, {})
                if self.size not in available_sizes:
                    options = ", ".join(available_sizes)
                    return (f"Unsupported model size '{self.size}'. Use: {options}",)
        else:
            descriptor = self.descriptor()
            if descriptor and descriptor.explicit_uri and not self.uri:
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
        errors = self._validation_messages(runtime=True)
        if errors:
            raise ValueError(errors[0])

    def descriptor(self) -> Optional[ModelDescriptor]:
        """Return the immutable capability descriptor for this model kind."""
        return MODEL_DESCRIPTORS.get(self.kind)

    def params_copy(self) -> Dict[str, Any]:
        """Return a mutable copy of backend parameters."""
        return dict(self.params)

    def editor_values(self, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return the editor-facing thresholds and index for this model.

        Pipeline documents historically store built-in model parameters under
        ``params`` and URI-backed model parameters at the model level. Keeping
        that compatibility mapping here prevents each editor or serializer
        from growing another model-kind conditional.
        """
        config = config or {}
        if not isinstance(config, Mapping):
            config = {}
        params = self.params

        def value(name, fallback=None):
            return params.get(name, config.get(name, fallback))

        if self.kind == "face":
            confidence = value("prob_threshold", config.get("conf_threshold"))
            if self.task == "recognition":
                confidence = value("threshold", confidence)
            iou = value("iou_threshold", config.get("iou_threshold"))
        elif self.kind == "license_plate":
            confidence = value("threshold", config.get("conf_threshold"))
            iou = value("iou_threshold", config.get("iou_threshold"))
        elif self.kind == "ocr":
            confidence = value("drop_score", config.get("conf_threshold"))
            iou = None
        else:
            confidence = config.get("conf_threshold")
            iou = config.get("iou_threshold")

        return {
            "conf_threshold": confidence,
            "iou_threshold": iou,
            "model_index": str(params.get("index", "")),
        }

    def to_pipeline_config(
        self,
        *,
        labels=(),
        conf_threshold=None,
        iou_threshold=None,
        approx=None,
        model_index="",
    ) -> Dict[str, Any]:
        """Serialize editor values into the stable pipeline model section."""
        model = {"task": self.task, "kind": self.kind}
        if self.uses_uri:
            model["uri"] = self.resolved_uri
            if labels:
                model["labels"] = list(labels)
            if conf_threshold is not None:
                model["conf_threshold"] = float(conf_threshold)
            if iou_threshold is not None:
                model["iou_threshold"] = float(iou_threshold)
            if approx is not None and self.task == "segmentation":
                model["approx"] = bool(approx)
            return model

        params = {}
        if conf_threshold is not None:
            if self.task == "recognition":
                parameter = "drop_score" if self.kind == "ocr" else "threshold"
            else:
                parameter = "prob_threshold" if self.kind == "face" else "threshold"
            params[parameter] = float(conf_threshold)
        if iou_threshold is not None:
            params["iou_threshold"] = float(iou_threshold)
        if self.kind == "face" and self.task == "recognition" and str(model_index).strip():
            params["index"] = str(model_index).strip()
        model["params"] = params
        return model


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
    "MODEL_DESCRIPTORS",
    "MODEL_TASKS_BY_KIND",
    "PIPELINE_HAILO_TASKS",
    "PIPELINE_MODEL_KINDS",
    "PIPELINE_MODEL_OPTIONS",
    "PIPELINE_MODEL_TASKS",
    "RESNET_MODEL_KINDS",
    "ModelSpec",
    "ModelDescriptor",
    "is_resnet_model_uri",
    "model_default_uri",
    "model_control_visibility",
    "model_defaults",
]
