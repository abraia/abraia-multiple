"""Optional model-assisted image annotation helpers."""

import json
import math
import os
import sys

from PIL import Image

from ..utils import load_image, load_url

if sys.platform == "darwin":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Annotator:
    """Grounding-DINO annotator with optional SAM polygon refinement."""

    def __init__(self, model="IDEA-Research/grounding-dino-tiny", segment=False):
        if sys.platform == "darwin":
            # Prevent native thread-pool contention when annotation follows
            # local PyTorch/Ultralytics training in the same process.
            import torch

            torch.set_num_threads(1)
        from transformers import pipeline

        self.pipe = pipeline(task="zero-shot-object-detection", model=model)
        self.segment_enabled = segment
        if self.segment_enabled:
            from abraia.inference.models.sam import SAM

            self.sam = SAM()

    def detect(self, img, classes, threshold=0.3):
        classes = [label.lower().strip() for label in classes]
        labels = [f"{label}." if not label.endswith(".") else label for label in classes]
        results = self.pipe(
            Image.fromarray(img),
            candidate_labels=labels,
            threshold=threshold,
        )
        objects = []
        for result in results:
            score = result["score"]
            if score > threshold:
                label = result["label"].rpartition(".")[0]
                xmin, ymin, xmax, ymax = result["box"].values()
                objects.append({
                    "label": label,
                    "score": score,
                    "box": [xmin, ymin, xmax - xmin, ymax - ymin],
                })
        return objects

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
    image_loader=load_image,
    url_loader=load_url,
):
    """Load and annotate one image row."""
    annotator = annotator or Annotator(segment=segment)
    url, filename = image_data["url"], image_data["name"]
    img = image_loader(url_loader(url))
    objects = annotator.detect(img, classes)
    if objects and segment:
        try:
            objects = annotator.segment(img, objects)
        except Exception:
            objects = None
    if not objects and not include_empty:
        return None
    return {"url": url, "filename": filename, "objects": objects}


__all__ = ["Annotator", "annotate_image"]
