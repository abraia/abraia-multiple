"""Optional model-assisted image annotation helpers."""

import json
import math
import os
import sys

from PIL import Image

from ..utils.image import load_image
from ..utils.remote import load_url


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
MOBILE_SAM_MODEL = "MobileSAM"
GROUNDING_DINO_MODEL = "Grounding DINO"
MOBILE_SAM_DISABLED_TASKS = frozenset(("classification",))


def annotation_color_rgb(annotation, class_names=None):
    """Return a class-list color, falling back to an explicit class ID."""
    from ..utils.draw import get_color, hex_to_rgb

    annotation = annotation if isinstance(annotation, dict) else {}
    label = annotation.get("label")
    if label is not None and class_names is not None:
        normalized = str(label).strip().casefold()
        for index, name in enumerate(class_names):
            if str(name).strip().casefold() == normalized:
                return hex_to_rgb(get_color(index))
        return hex_to_rgb(get_color(len(class_names)))
    class_id = annotation.get("class_id")
    try:
        class_id = int(class_id) if class_id is not None else 0
    except (TypeError, ValueError):
        class_id = 0
    return hex_to_rgb(get_color(class_id))


def model_route(model_name, task):
    """Return the inference route for an annotation model and task."""
    if model_name == MOBILE_SAM_MODEL:
        return None if task in MOBILE_SAM_DISABLED_TASKS else "sam"
    if model_name == GROUNDING_DINO_MODEL:
        operation = (
            "grounding_dino_segmentation"
            if task == "segmentation"
            else "grounding_dino"
        )
        return operation
    return "custom"


class Annotation:
    """Select annotation models and manage cached inference sessions."""

    DEFAULT_LABEL = "object"
    GROUNDING_DINO_MODEL = GROUNDING_DINO_MODEL
    MOBILE_SAM_MODEL = MOBILE_SAM_MODEL
    MOBILE_SAM_DISABLED_TASKS = MOBILE_SAM_DISABLED_TASKS

    @staticmethod
    def normalize_polygon(points):
        """Return finite ``(x, y)`` coordinates, or an empty invalid polygon."""
        normalized = []
        for point in points if points is not None else ():
            try:
                if hasattr(point, "x") and hasattr(point, "y"):
                    x, y = point.x(), point.y()
                else:
                    x, y = point[:2]
                x, y = float(x), float(y)
            except (TypeError, ValueError, IndexError):
                return []
            if not math.isfinite(x) or not math.isfinite(y):
                return []
            normalized.append((x, y))
        return normalized if len(normalized) >= 3 else []

    @classmethod
    def normalize_geometry(cls, annotation):
        """Return ``(points, box, invalid_box)`` in source-image coordinates."""
        if not isinstance(annotation, dict):
            return (), (), False
        points = cls.normalize_polygon(annotation.get("polygon", []))
        if points:
            return tuple(points), (), False
        box = annotation.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                coordinates = tuple(float(value) for value in box)
            except (TypeError, ValueError):
                return (), (), True
            if all(math.isfinite(value) for value in coordinates):
                if coordinates[2] > 0 and coordinates[3] > 0:
                    return (), coordinates, False
            return (), (), True
        return (), (), False

    @classmethod
    def polygon_bounds(cls, points):
        """Return an ``(x, y, width, height)`` tuple for a valid polygon."""
        points = cls.normalize_polygon(points)
        if not points:
            return None
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    @classmethod
    def model_options(cls, task, custom_models=()):
        """Return built-in and custom models compatible with a task."""
        from ..tasks import normalize_task

        task = normalize_task(task)
        options = [GROUNDING_DINO_MODEL]
        if task not in MOBILE_SAM_DISABLED_TASKS:
            options.append(MOBILE_SAM_MODEL)
        options.extend(
            model for model in custom_models or ()
            if model not in (MOBILE_SAM_MODEL, GROUNDING_DINO_MODEL)
        )
        return options

    @classmethod
    def default_model(cls, task, label_defined, available_models):
        """Choose a default compatible with task and annotation labels."""
        from ..tasks import normalize_task

        if not available_models:
            return ""
        task = normalize_task(task)
        preferred = (
            GROUNDING_DINO_MODEL
            if label_defined or task in MOBILE_SAM_DISABLED_TASKS
            else MOBILE_SAM_MODEL
        )
        if preferred in available_models:
            return preferred
        if GROUNDING_DINO_MODEL in available_models:
            return GROUNDING_DINO_MODEL
        return available_models[0]

    @classmethod
    def has_defined_label(cls, labels):
        """Return whether labels contain one beyond the editor's default."""
        return any(
            str(label or "").strip().casefold()
            not in ("", cls.DEFAULT_LABEL)
            for label in labels or ()
        )

    @staticmethod
    def route(model_name, task):
        """Return the inference route for a selected model and task."""
        from ..tasks import normalize_task

        return model_route(model_name, normalize_task(task))

    def __init__(self):
        import threading

        self._models = {}
        self._sam = None
        self._sam_image_key = None
        self._grounding_dino = None
        self._lock = threading.RLock()

    def _run_model(self, model_config, image):
        """Run a configured custom model through Abraia's inference API."""
        import numpy as np

        from ..inference import load_model

        key = (
            model_config["kind"],
            model_config["task"],
            model_config["uri"],
            json.dumps(model_config.get("params", {}), sort_keys=True),
        )
        with self._lock:
            if key not in self._models:
                self._models[key] = load_model(model_config)
            return self._models[key].run(np.asarray(image))

    def predict(self, model_name, task, image, *, model_config=None, label=None):
        """Run a selected model and return annotation records."""
        from ..inference.geometry import mask_to_polygon

        route = self.route(model_name, task)
        if route is None or route == "sam":
            raise ValueError("MobileSAM requires interactive point or box prompts")
        if route == "grounding_dino":
            objects = self._grounding_dino_predict(image, label)
        elif route == "grounding_dino_segmentation":
            objects = self._grounding_dino_sam_predict(image, label)
        else:
            if model_config is None:
                raise ValueError("A normalized model configuration is required")
            objects = self._run_model(model_config, image)

        annotations = []
        for item in objects:
            annotation = dict(item)
            mask = annotation.pop("mask", None)
            if mask is not None and "polygon" not in annotation:
                annotation["polygon"] = mask_to_polygon(mask)
            if "polygon" in annotation:
                polygon = self.normalize_polygon(annotation["polygon"])
                if polygon:
                    annotation["polygon"] = polygon
                else:
                    annotation.pop("polygon")
            if label is not None:
                annotation["label"] = label
            annotations.append(annotation)
        return annotations

    def sam_predict(self, image, points):
        """Return a polygon predicted from point prompts."""
        from ..inference.geometry import mask_to_polygon

        if not points:
            raise ValueError("MobileSAM requires at least one point")
        prompt = [
            {
                "type": "point",
                "data": [float(point[0]), float(point[1])],
                "label": int(point[2]),
            }
            for point in points
        ]
        return mask_to_polygon(self._sam_predict(image, prompt))

    def sam_predict_box(self, image, box):
        """Return a polygon predicted from an ``xywh`` box prompt."""
        from ..inference.geometry import mask_to_polygon

        if box is None or len(box) != 4:
            raise ValueError("SAM box prompts require [x, y, width, height]")
        x, y, width, height = (float(value) for value in box)
        if width <= 0 or height <= 0:
            raise ValueError("SAM box prompts require positive dimensions")
        return mask_to_polygon(self._sam_predict(image, [{
            "type": "rectangle",
            "data": [x, y, x + width, y + height],
        }]))

    def _grounding_dino_predict(self, image, prompt):
        """Detect objects described by a Grounding DINO text prompt."""
        import numpy as np

        from ..inference.models.grounding_dino import GroundingDINOModel

        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Grounding DINO requires a text prompt")
        with self._lock:
            if self._grounding_dino is None:
                self._grounding_dino = GroundingDINOModel()
            return self._grounding_dino.run(np.asarray(image), prompt=prompt)

    def _grounding_dino_sam_predict(self, image, prompt):
        """Refine Grounding DINO boxes into polygons with SAM."""
        segmented = []
        for detection in self._grounding_dino_predict(image, prompt):
            polygon = self.sam_predict_box(image, detection.get("box"))
            if polygon:
                segmented.append({**detection, "polygon": polygon})
        return segmented

    def close(self):
        """Close all cached model sessions."""
        with self._lock:
            resources = [*self._models.values(), self._sam, self._grounding_dino]
            self._models = {}
            self._sam = self._grounding_dino = None
            self._sam_image_key = None
            errors = []
            for resource in resources:
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as error:
                        errors.append(error)
            if errors:
                raise errors[0]

    def _sam_predict(self, image, prompt):
        import hashlib
        import numpy as np

        from ..inference.models.sam import SAM

        image = np.asarray(image)
        contiguous = np.ascontiguousarray(image)
        image_key = (
            contiguous.shape,
            contiguous.dtype.str,
            hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest(),
        )
        with self._lock:
            if self._sam is None:
                self._sam = SAM()
            if self._sam_image_key != image_key:
                self._sam.encode(image)
                self._sam_image_key = image_key
            return self._sam.predict(image, prompt=json.dumps(prompt))


if sys.platform == "darwin":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Annotator:
    """Transformers Grounding-DINO annotator with optional SAM refinement."""

    def __init__(self, segment=False):
        if sys.platform == "darwin":
            # Prevent native thread-pool contention when annotation follows
            # local PyTorch/Ultralytics training in the same process.
            import torch

            torch.set_num_threads(1)
        from transformers import pipeline

        self.pipe = pipeline(
            task="zero-shot-object-detection",
            model=DEFAULT_MODEL,
        )
        self.segment_enabled = segment
        if self.segment_enabled:
            from abraia.inference.models.sam import SAM

            self.sam = SAM()

    def detect(self, img, classes, threshold=0.3):
        classes = [str(label).lower().strip().rstrip(".") for label in classes]
        classes = list(dict.fromkeys(label for label in classes if label))
        if not classes:
            return []
        labels = [
            f"{label}." if not label.endswith(".") else label
            for label in classes
        ]
        results = self.pipe(
            Image.fromarray(img),
            candidate_labels=labels,
            threshold=threshold,
        )
        objects = []
        for result in results:
            score = float(result["score"])
            label = str(result["label"]).strip().lower().rstrip(".")
            if score >= threshold and label:
                xmin, ymin, xmax, ymax = result["box"].values()
                objects.append({
                    "label": label,
                    "score": score,
                    "box": [xmin, ymin, xmax - xmin, ymax - ymin],
                })
        return objects

    def classify(self, img, classes, threshold=0.3):
        """Return the best detected class as an image-level annotation."""
        objects = self.detect(img, classes, threshold=threshold)
        if not objects:
            return []
        result = max(objects, key=lambda annotation: annotation["score"])
        return [{
            "label": result["label"],
            "score": result["score"],
        }]

    def close(self):
        """Release the optional SAM session and drop the detector pipeline."""
        pipe = getattr(self, "pipe", None)
        for resource in (getattr(self, "sam", None), pipe):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self.pipe = None

    def segment(self, img, objects):
        from abraia.inference.geometry import mask_to_polygon

        self.sam.encode(img)
        for result in objects:
            x, y, w, h = result["box"]
            mask = self.sam.predict(
                img,
                prompt=json.dumps([{
                    "type": "rectangle",
                    "data": [x, y, x + w, y + h],
                }]),
            )
            height, width = mask.shape[:2]
            left = max(0, min(width, math.floor(x)))
            top = max(0, min(height, math.floor(y)))
            right = max(left, min(width, math.ceil(x + w)))
            bottom = max(top, min(height, math.ceil(y + h)))
            result["polygon"] = mask_to_polygon(
                mask[top:bottom, left:right], (left, top)
            )
        return objects


def annotate_image(
    image_data,
    classes,
    segment=False,
    annotator=None,
    include_empty=False,
    classification=False,
    image_loader=load_image,
    url_loader=load_url,
):
    """Load and annotate one image row."""
    owns_annotator = annotator is None
    annotator = annotator or Annotator(segment=segment)
    try:
        url, filename = image_data["url"], image_data["name"]
        img = image_loader(url_loader(url))
        objects = (
            annotator.classify(img, classes)
            if classification
            else annotator.detect(img, classes)
        )
        if objects and segment and not classification:
            try:
                objects = annotator.segment(img, objects)
            except Exception:
                objects = None
        if not objects and not include_empty:
            return None
        return {"url": url, "filename": filename, "objects": objects}
    finally:
        if owns_annotator:
            annotator.close()


__all__ = [
    "Annotation",
    "annotation_color_rgb",
    "Annotator",
    "GROUNDING_DINO_MODEL",
    "MOBILE_SAM_DISABLED_TASKS",
    "MOBILE_SAM_MODEL",
    "annotate_image",
    "model_route",
]
