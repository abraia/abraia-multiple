"""ONNX inference adapter for anomaly models."""

import numpy as np
from PIL import Image

from abraia.inference.session import OnnxSessionMixin
from abraia.utils.remote import resolve_model_file


def render_anomaly_heatmap(image, anomaly_map):
    """Overlay calibrated pixel detections as an Anomalib-style JET map."""
    import cv2

    image = np.asarray(image)
    height, width = image.shape[:2]
    heat_values = cv2.resize(
        np.asarray(anomaly_map, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    anomaly_strength = np.clip((heat_values - 0.5) * 2.0, 0.0, 1.0)
    heatmap = cv2.applyColorMap(
        (anomaly_strength * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    alpha = anomaly_strength[..., None] * 0.65
    return (
        image.astype(np.float32) * (1.0 - alpha)
        + heatmap.astype(np.float32) * alpha
    ).clip(0, 255).astype(image.dtype)


class AnomalyModel(OnnxSessionMixin):
    """Run saved anomaly models through the SDK's ONNX session lifecycle."""

    def __init__(self, model_uri, providers=None, accelerator=None):
        self._init_onnx_session(
            resolve_model_file(model_uri),
            providers=providers, accelerator=accelerator,
        )
        try:
            model_input = self.session.get_inputs()[0]
            self.input_name = model_input.name
            self.input_shape = model_input.shape
            metadata = self.session.get_modelmeta().custom_metadata_map
            if metadata["task"] != "anomaly":
                raise ValueError("Select an exported anomaly model.")
            self.algorithm = metadata["algorithm"]
            self.image_threshold = float(metadata["image_threshold"])
            self.pixel_threshold = float(metadata["pixel_threshold"])
            normalization_keys = (
                "image_min", "image_max", "pixel_min", "pixel_max",
            )
            self.has_calibration = all(
                key in metadata for key in normalization_keys
            )
            if self.has_calibration:
                self.image_min = float(metadata["image_min"])
                self.image_max = float(metadata["image_max"])
                self.pixel_min = float(metadata["pixel_min"])
                self.pixel_max = float(metadata["pixel_max"])
            self.last_normalized_anomaly_map = None
        except Exception:
            self.close()
            raise

    @staticmethod
    def _normalize(values, minimum, maximum, threshold):
        value_range = maximum - minimum
        if value_range == 0:
            return np.where(
                values > threshold,
                1.0,
                np.where(values < threshold, 0.0, 0.5),
            ).astype(np.float32)
        return np.clip(
            ((values - threshold) / value_range) + 0.5,
            0.0,
            1.0,
        ).astype(np.float32)

    def predict(self, image):
        """Return the label, score, and dense anomaly map for a uint8 RGB image."""
        self._ensure_open()
        height, width = self.input_shape[2:]
        interpolation = (
            Image.Resampling.BICUBIC
            if self.algorithm in ("dinomaly", "subspacead")
            else Image.Resampling.BILINEAR
        )
        resized = Image.fromarray(image).convert("RGB").resize(
            (width, height), interpolation
        )
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
        scores, maps = self.session.run(
            ["score", "anomaly_map"], {self.input_name: tensor}
        )
        score = float(scores[0])
        anomaly_map = maps[0, 0]
        return self._prediction(score, anomaly_map)

    def _prediction(self, score, anomaly_map):
        """Apply the saved anomaly thresholds and calibration to model outputs."""
        if self.has_calibration:
            normalized_score = float(self._normalize(
                np.asarray(score),
                self.image_min,
                self.image_max,
                self.image_threshold,
            ))
            normalized_map = self._normalize(
                anomaly_map,
                self.pixel_min,
                self.pixel_max,
                self.pixel_threshold,
            )
            label = "anomaly" if normalized_score > 0.5 else "normal"
        else:
            normalized_score = score
            # Without validation calibration, expose detections without
            # inventing a per-image intensity scale.
            normalized_map = (
                anomaly_map > self.pixel_threshold
            ).astype(np.float32)
            label = "anomaly" if score > self.image_threshold else "normal"
        return {
            "label": label,
            "score": score,
            "normalized_score": normalized_score,
            "anomaly_map": anomaly_map,
            "normalized_anomaly_map": normalized_map,
        }

    def run(self, image, prediction=None):
        """Return pipeline results: image classification and defect polygons."""
        import cv2

        if prediction is None:
            prediction = self.predict(image)
        anomaly_map = prediction["anomaly_map"]
        normalized_map = prediction["normalized_anomaly_map"]
        self.last_normalized_anomaly_map = normalized_map
        height, width = image.shape[:2]
        results = [{
            "label": prediction["label"],
            "score": prediction["normalized_score"],
            "class_id": int(prediction["label"] == "anomaly"),
            "box": [0, 0, width, height],
        }]
        mask = (normalized_map > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        scale = np.array([width / mask.shape[1], height / mask.shape[0]])
        for contour in contours:
            if len(contour) < 3:
                continue
            polygon = np.round(contour[:, 0] * scale).astype(int)
            polygon[:, 0] = polygon[:, 0].clip(0, width - 1)
            polygon[:, 1] = polygon[:, 1].clip(0, height - 1)
            left, top = polygon.min(axis=0)
            right, bottom = polygon.max(axis=0)
            x, y, box_width, box_height = cv2.boundingRect(contour)
            score = float(anomaly_map[y:y + box_height, x:x + box_width].max())
            results.append({
                "label": "defect", "score": score, "class_id": 1,
                "box": [
                    int(left), int(top),
                    int(right - left + 1), int(bottom - top + 1),
                ],
                "polygon": polygon.tolist(),
            })
        return results



__all__ = ["AnomalyModel", "render_anomaly_heatmap"]
