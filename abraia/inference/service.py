"""High-level inference services for application integrations.

These services own inference model lifecycles so clients such as Studio only
need to deal with application data and callbacks.
"""

from __future__ import annotations

import json

from ..utils import as_array


def _create_onnx_model(model_uri):
    """Create the default ONNX backend lazily."""
    from . import Model

    return Model(model_uri)


class ModelSession:
    """Own one backend model instance and its resource lifecycle.

    Backend factories must return an object exposing ``run``. A ``close``
    method is optional but is used when available.
    """

    def __init__(self, model_uri, factory):
        self.model_uri = model_uri
        self._model = factory(model_uri)
        self._closed = False

    def run(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Model session has already been closed")
        return self._model.run(*args, **kwargs)

    def close(self):
        if self._closed:
            return
        model, self._model = self._model, None
        close = getattr(model, "close", None)
        if callable(close):
            close()
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
        self._sam = None
        self._sam_image = None

    def get_session(self, model_uri, backend="onnx"):
        """Return a cached model session for ``model_uri`` and ``backend``."""
        try:
            factory = self._backend_factories[backend]
        except KeyError as exc:
            available = ", ".join(sorted(self._backend_factories))
            raise ValueError(
                f"Unknown inference backend '{backend}'. Available backends: {available}"
            ) from exc

        key = (backend, str(model_uri))
        if key not in self._sessions:
            self._sessions[key] = ModelSession(model_uri, factory)
        return self._sessions[key]

    def register_backend(self, name, factory):
        """Register or replace a backend factory used by this service."""
        if not name or not callable(factory):
            raise ValueError("Backend name and callable factory are required")
        self._backend_factories[name] = factory

    def run(self, model_uri, image, backend="onnx", **kwargs):
        """Run a model, reusing its loaded session for subsequent calls."""
        session = self.get_session(model_uri, backend=backend)
        return session.run(as_array(image), **kwargs)

    def close(self):
        """Close all cached model sessions and interactive models."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        if self._sam is not None:
            close = getattr(self._sam, "close", None)
            if callable(close):
                close()
        self._sam = None
        self._sam_image = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def sam_predict(self, image, points):
        """Predict a mask from point prompts, reusing the image embedding."""
        if not points:
            raise ValueError("MobileSAM requires at least one point")

        from . import SAM

        image = as_array(image)
        if self._sam is None:
            self._sam = SAM()
        if self._sam_image is not image:
            self._sam.encode(image)
            self._sam_image = image

        prompt = [
            {
                "type": "point",
                "data": [float(point[0]), float(point[1])],
                "label": int(point[2]),
            }
            for point in points
        ]
        return self._sam.predict(image, prompt=json.dumps(prompt))
