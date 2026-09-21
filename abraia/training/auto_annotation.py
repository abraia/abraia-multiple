"""Optional model-assisted image annotation helpers."""

import json
import math
import os
import sys

from PIL import Image

from ..utils import load_image, load_url


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

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
        from abraia.inference.postprocess.masks import mask_to_polygon

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


__all__ = ["Annotator", "annotate_image"]
