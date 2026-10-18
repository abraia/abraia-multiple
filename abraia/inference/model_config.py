"""Built-in model kinds, defaults, and runtime option definitions."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional

from ..tasks import HAILO_TASKS, PIPELINE_TASKS, normalize_task


@dataclass(frozen=True)
class ModelDescriptor:
    """Static capabilities shared by editors, validation, and factories."""

    tasks: FrozenSet[str]
    uri_based: bool = False
    explicit_uri: bool = False
    runtime_options: bool = False


_MODEL_ARCHITECTURE_ORDER = ("yolov5", "yolov8", "yolo11", "rtmdet")
MODEL_ARCHITECTURES = frozenset(_MODEL_ARCHITECTURE_ORDER)
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
    "rtmdet": {
        "detection": {
            "small": "multiple/models/rtmdet_small_v1.onnx",
            "medium": "multiple/models/rtmdet_medium_v1.onnx",
            "large": "multiple/models/rtmdet_large_v1.onnx",
        },
        "segmentation": {
            "small": "multiple/models/rtmdet_ins_small_v1.onnx",
            "medium": "multiple/models/rtmdet_ins_medium_v1.onnx",
            "large": "multiple/models/rtmdet_ins_large_v1.onnx",
        },
    },
    "resnet": {
        "classification": {
            "small": "multiple/models/resnet18.onnx",
            "medium": "multiple/models/resnet50.onnx",
            "large": "multiple/models/resnet101.onnx",
        },
    },
    "grounding_dino": {
        "detection": {"small": "multiple/models/grounding_dino_tiny.onnx"},
    },
}

DEFAULT_MODEL_URIS = {
    kind: {
        task: sizes["small"]
        for task, sizes in tasks.items()
    }
    for kind, tasks in MODEL_SIZE_URIS.items()
}

PIPELINE_MODEL_TASKS = PIPELINE_TASKS
PIPELINE_HAILO_TASKS = HAILO_TASKS
MODEL_DESCRIPTORS = MappingProxyType({
    **{
        kind: ModelDescriptor(
            frozenset(
                {"detection", "segmentation"}
                if kind == "rtmdet"
                else {"detection", "segmentation", "pose"}
            ),
            uri_based=True,
            runtime_options=True,
        )
        for kind in _MODEL_ARCHITECTURE_ORDER
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
    "multispectral": ModelDescriptor(
        frozenset({"classification"}),
        uri_based=True,
        explicit_uri=True,
    ),
})
# The descriptor catalog is the supported pipeline model list.
PIPELINE_MODEL_KINDS = tuple(MODEL_DESCRIPTORS)
@dataclass(frozen=True)
class ModelSpec:
    """Normalized model configuration used by inference adapters.

    Pipeline documents remain dictionaries; this holds the normalized fields
    needed to choose and validate a runtime adapter.
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
        raw_uri = config.get("uri", "")
        uri_name = str(raw_uri or "").lower()
        default_task = (
            "segmentation"
            if kind == "rtmdet"
            and any(marker in uri_name for marker in ("rtmdet_ins", "rtmdet-ins"))
            else "classification" if kind == "resnet" else "detection"
        )
        task = normalize_task(
            config.get("task"),
            default=default_task,
        )
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
        descriptor = self.descriptor
        return descriptor.tasks if descriptor else frozenset()

    @property
    def default_uri(self) -> str:
        """Return the built-in URI for this kind/task/size combination."""
        return model_default_uri(self.kind, self.task, self.size)

    @property
    def resolved_uri(self) -> str:
        """Return the configured URI or its built-in default."""
        return self.uri or self.default_uri

    @property
    def result_field(self) -> str:
        """Return the output field used by a follow-up model step."""
        if self.kind == "ocr":
            return "ocr"
        if self.task == "classification":
            return "classification"
        if self.task == "recognition":
            return "recognition"
        return "result"

    def require_runtime_valid(self) -> None:
        """Validate the model before constructing a runtime adapter."""
        if not self.kind:
            raise ValueError("Pipeline model must define a model 'kind'")
        if self.kind not in PIPELINE_MODEL_KINDS:
            available = ", ".join(PIPELINE_MODEL_KINDS)
            raise ValueError(
                f"Unknown detector kind '{self.kind}'. Available detectors: {available}"
            )

        descriptor = self.descriptor
        if self.backend == "hailo":
            if self.kind not in MODEL_ARCHITECTURES:
                raise ValueError("Hailo models require a supported model architecture kind")
            if self.task not in HAILO_TASKS:
                raise ValueError(f"Unsupported Hailo pipeline task: {self.task}")
        elif descriptor and self.task not in descriptor.tasks:
            if self.kind in MODEL_ARCHITECTURES:
                raise ValueError(
                    f"{self.kind} models support detection, segmentation, or pose."
                )
            if self.kind == "resnet":
                raise ValueError("Classification models require the classification task.")
            if self.kind == "grounding_dino":
                raise ValueError("Grounding DINO models require the detection task.")
            if self.kind == "ocr":
                raise ValueError("OCR models only support the recognition task.")
            if self.kind == "multispectral":
                raise ValueError("Multispectral models require the classification task.")
            raise ValueError(f"{self.kind} models do not support the {self.task} task.")

        if descriptor and descriptor.explicit_uri and not self.uri:
            raise ValueError(f"A {self.kind} model requires a model 'uri'")
        if self.kind in MODEL_ARCHITECTURES and not self.uri:
            available_sizes = MODEL_SIZE_URIS.get(self.kind, {}).get(self.task, {})
            if not available_sizes:
                raise ValueError(f"A {self.kind} model requires a model 'uri'")
            if self.size not in available_sizes:
                options = ", ".join(available_sizes)
                raise ValueError(
                    f"Unsupported model size '{self.size}'. Use: {options}"
                )

    @property
    def descriptor(self) -> Optional[ModelDescriptor]:
        """Return the model's shared capability record."""
        return MODEL_DESCRIPTORS.get(self.kind)

    def params_copy(self) -> Dict[str, Any]:
        """Return a mutable copy of backend parameters."""
        return dict(self.params)


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
    "MODEL_RUN_OPTIONS",
    "MODEL_SIZE_URIS",
    "MODEL_ARCHITECTURES",
    "MODEL_DESCRIPTORS",
    "PIPELINE_HAILO_TASKS",
    "PIPELINE_MODEL_KINDS",
    "PIPELINE_MODEL_TASKS",
    "ModelSpec",
    "ModelDescriptor",
    "is_resnet_model_uri",
    "model_default_uri",
]
