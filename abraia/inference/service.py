"""High-level inference services for application integrations.

These services own inference model lifecycles so clients such as Studio only
need to deal with application data and callbacks.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import threading
from typing import Callable

import numpy as np

from ..utils import as_array
from .contracts import InferenceModel


def _image_signature(image):
    """Return a stable, mutation-sensitive key for an image array."""
    image = as_array(image)
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(
        contiguous.view(np.uint8), digest_size=16
    ).digest()
    return contiguous.shape, contiguous.dtype.str, digest


def _create_onnx_model(model_uri):
    """Create the default ONNX backend lazily."""
    from .model_config import is_resnet_model_uri

    if is_resnet_model_uri(model_uri):
        from .models.classification import ResNetClassifier

        return ResNetClassifier(model_uri)
    from . import Model

    return Model(model_uri)


class ModelSession:
    """Own one backend model instance and its resource lifecycle.

    Backend factories must return an object exposing ``run`` or
    ``iter_inference``. A ``close`` method is optional but is used when
    available.
    """

    def __init__(self, model_uri, factory: Callable[[str], InferenceModel]):
        self.model_uri = model_uri
        self._model: InferenceModel = factory(model_uri)
        if not any(
            callable(getattr(self._model, name))
            for name in ("run", "iter_inference")
        ):
            raise TypeError(
                "Inference backend models must expose run or iter_inference"
            )
        self._closed = False
        self._lock = threading.RLock()

    def run(self, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("Model session has already been closed")
            return self._model.run(*args, **kwargs)

    def iter_inference(self, source, **kwargs):
        """Delegate the streaming model protocol when the backend supports it."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Model session has already been closed")
            iterator = getattr(self._model, "iter_inference", None)
            if not callable(iterator):
                raise TypeError("This inference backend only supports run")
            # Keep the operation lock for the whole iterator lifetime. A
            # backend session is not assumed to be safe for concurrent
            # streaming and one-shot calls.
            yield from iterator(source, **kwargs)

    def close(self):
        with self._lock:
            if self._closed:
                return
            model, self._model = self._model, None
            close = getattr(model, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class InferenceService:
    """Run reusable inference sessions and interactive segmentation sessions."""

    def __init__(self, backend_factories=None):
        self._backend_factories = {"onnx": _create_onnx_model}
        if backend_factories:
            self._backend_factories.update(backend_factories)
        self._sessions = {}
        self._lock = threading.RLock()
        self._sam = None
        self._sam_image = None
        self._sam_image_key = None
        self._grounding_dino = None

    def get_session(self, model_uri, backend="onnx"):
        """Return a cached model session for ``model_uri`` and ``backend``."""
        with self._lock:
            try:
                factory = self._backend_factories[backend]
            except KeyError as exc:
                available = ", ".join(sorted(self._backend_factories))
                raise ValueError(
                    f"Unknown inference backend '{backend}'. Available backends: {available}"
                ) from exc

            key = self._session_key(model_uri, backend)
            if key not in self._sessions:
                self._sessions[key] = ModelSession(model_uri, factory)
            return self._sessions[key]

    @staticmethod
    def _session_key(model_uri, backend):
        """Use one cache key for equivalent local path spellings."""
        uri = os.fspath(model_uri)
        if not uri.lower().startswith(("http://", "https://")):
            path = Path(uri).expanduser()
            if path.exists():
                uri = str(path.resolve())
        return backend, uri

    def register_backend(self, name, factory):
        """Register or replace a backend factory used by this service."""
        if not name or not callable(factory):
            raise ValueError("Backend name and callable factory are required")
        with self._lock:
            self._backend_factories[name] = factory

    def create_model(self, config, base_dir=None):
        """Create a configured built-in model using the shared registry."""
        from .registry import create_model

        return create_model(config, base_dir=base_dir)

    def run(self, model_uri, image, backend="onnx", **kwargs):
        """Run a model, reusing its loaded session for subsequent calls."""
        session = self.get_session(model_uri, backend=backend)
        return session.run(as_array(image), **kwargs)

    def close(self):
        """Close all cached model sessions and interactive models."""
        errors = []
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            sam, self._sam = self._sam, None
            grounding_dino, self._grounding_dino = self._grounding_dino, None
            self._sam_image = None
            self._sam_image_key = None
            for resource in [*sessions, sam, grounding_dino]:
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _sam_predict(self, image, prompt):
        """Predict a mask from a serialized MobileSAM prompt."""
        image = as_array(image)
        image_key = _image_signature(image)
        with self._lock:
            from . import SAM

            if self._sam is None:
                self._sam = SAM()
            sam = self._sam
            if self._sam_image_key != image_key:
                sam.encode(image)
                self._sam_image = image
                self._sam_image_key = image_key
            return sam.predict(image, prompt=json.dumps(prompt))

    def sam_predict(self, image, points):
        """Predict a mask from point prompts, reusing the image embedding."""
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
        return self._sam_predict(image, prompt)

    def sam_predict_box(self, image, box):
        """Predict a MobileSAM mask from an ``xywh`` box prompt."""
        if box is None or len(box) != 4:
            raise ValueError("MobileSAM box prompts require [x, y, width, height]")
        x, y, width, height = (float(value) for value in box)
        if width <= 0 or height <= 0:
            raise ValueError("MobileSAM box prompts require positive dimensions")
        prompt = [{
            "type": "rectangle",
            "data": [x, y, x + width, y + height],
        }]
        return self._sam_predict(image, prompt)

    def grounding_dino_predict(self, image, prompt):
        """Detect objects described by a text prompt using Grounding DINO."""
        if not str(prompt or "").strip():
            raise ValueError("Grounding DINO requires a text prompt")

        from . import GroundingDINOModel

        image = as_array(image)
        with self._lock:
            if self._grounding_dino is None:
                self._grounding_dino = GroundingDINOModel()
            return self._grounding_dino.run(image, prompt=str(prompt).strip())

    def grounding_dino_sam_predict(self, image, prompt):
        """Detect prompted objects and refine each box with MobileSAM."""
        from .postprocess.masks import mask_to_polygon

        detections = self.grounding_dino_predict(image, prompt)
        segmented = []
        for detection in detections:
            polygon = mask_to_polygon(
                self.sam_predict_box(image, detection.get("box"))
            )
            if not polygon:
                continue
            annotation = dict(detection)
            annotation["polygon"] = polygon
            annotation.pop("box", None)
            segmented.append(annotation)
        return segmented
