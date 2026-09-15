"""High-level inference services for application integrations.

These services own inference model lifecycles so clients such as Studio only
need to deal with application data and callbacks.
"""

from __future__ import annotations

import json

from ..utils import as_array


class InferenceService:
    """Run Abraia models and interactive segmentation sessions."""

    def __init__(self):
        self._sam = None
        self._sam_image = None

    def run(self, model_uri, image):
        """Run a remote model identified by ``model_uri`` on ``image``."""
        from . import Model

        return Model(model_uri).run(as_array(image))

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
