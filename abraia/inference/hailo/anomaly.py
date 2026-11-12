"""Hailo inference adapter for Multiple anomaly exports."""

import numpy as np
from pathlib import Path

import numpy as np
from PIL import Image

from ..models.anomaly import AnomalyModel


class HailoAnomalyModel(AnomalyModel):
    """Run a compiled anomaly HEF through Abraia's shared Hailo runtime."""

    def __init__(self, model_uri):
        from abraia.inference.hailo.models import (
            load_hailo_model_config,
            resolve_hef_path,
        )
        from abraia.inference.hailo.toolbox import HAILO_AVAILABLE, HailoInfer

        self.inference = None
        self._closed = False
        try:
            if not HAILO_AVAILABLE or HailoInfer is None:
                raise RuntimeError(
                    "Hailo support is unavailable; install hailo_platform on a "
                    "Hailo-enabled host"
                )
            metadata = load_hailo_model_config(model_uri)
            if (
                metadata.get("format") != "hailo"
                or metadata.get("kind") != "anomaly"
                or metadata.get("task") != "anomaly"
            ):
                raise ValueError("Select a compiled anomaly Hailo model.")

            self.algorithm = str(metadata["model"]).strip().lower()
            if self.algorithm not in (
                "patchcore", "efficientad", "dinomaly", "subspacead",
            ):
                raise ValueError(
                    f"Unsupported anomaly model in Hailo metadata: {self.algorithm!r}."
                )
            self.image_threshold = float(metadata["image_threshold"])
            self.pixel_threshold = float(metadata["pixel_threshold"])
            calibration_keys = (
                "image_min", "image_max", "pixel_min", "pixel_max",
            )
            self.has_calibration = all(
                metadata.get(key) is not None for key in calibration_keys
            )
            if self.has_calibration:
                self.image_min = float(metadata["image_min"])
                self.image_max = float(metadata["image_max"])
                self.pixel_min = float(metadata["pixel_min"])
                self.pixel_max = float(metadata["pixel_max"])

            input_shape = metadata.get("inputShape")
            if (
                not isinstance(input_shape, (list, tuple))
                or len(input_shape) != 4
                or any(int(value) <= 0 for value in input_shape)
                or tuple(int(value) for value in input_shape[:2]) != (1, 3)
            ):
                raise ValueError(
                    "Anomaly Hailo metadata must declare a static "
                    "[1, 3, H, W] input."
                )
            self.input_shape = tuple(int(value) for value in input_shape)
            output_roles = metadata.get("outputRoles") or {}
            self.postprocessing = metadata.get("postprocessing", "anomaly_host")
            required_roles = (
                ("features",)
                if self.postprocessing == "patchcore_cpu"
                else ("score", "anomaly_map")
            )
            if not all(output_roles.get(role) for role in required_roles):
                raise ValueError(
                    f"Anomaly Hailo metadata must map outputs: {required_roles}."
                )
            self.output_roles = output_roles
            if self.postprocessing == "patchcore_cpu":
                from abraia.utils.remote import resolve_model_file
                state_path = (
                    Path(model_uri).with_suffix("")
                    / metadata["postprocessingState"]
                )
                with np.load(resolve_model_file(state_path)) as state:
                    self.memory = np.asarray(state["memory"], dtype=np.float32)
                self.feature_shape = tuple(
                    int(value) for value in metadata["featureShape"]
                )
                self.normalize_features = (
                    metadata.get("featureNormalization") == "l2_cpu"
                )

            hef_path = resolve_hef_path(model_uri, "anomaly")
            if hef_path is None:
                raise FileNotFoundError(
                    f"Unable to resolve Hailo anomaly model: {model_uri}"
                )
            self.inference = HailoInfer(
                str(hef_path), input_type="FLOAT32", output_type="FLOAT32"
            )
            input_infos, output_infos = self.inference.get_vstream_info()
            if len(input_infos) != 1:
                raise ValueError(
                    "Anomaly Hailo models require one image input; "
                    f"found {len(input_infos)}."
                )
            input_height, input_width, channels = input_infos[0].shape
            if (
                channels != 3
                or (input_height, input_width) != self.input_shape[2:]
            ):
                raise ValueError(
                    "The Hailo input shape does not match its anomaly metadata."
                )
            self.input_height = int(input_height)
            self.input_width = int(input_width)
            self.output_names = tuple(info.name for info in output_infos)
            self.last_normalized_anomaly_map = None
            self.accelerator = "HAILO"
        except Exception:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Inference model has already been closed")

    def _infer(self, tensor):
        outputs = []

        def receive(completion_info, bindings_list, **_kwargs):
            if completion_info.exception:
                outputs.append(RuntimeError(str(completion_info.exception)))
                return
            try:
                binding = bindings_list[0]
                outputs.append({
                    name: np.array(
                        binding.output(name).get_buffer(),
                        dtype=np.float32,
                        copy=True,
                    )
                    for name in self.output_names
                })
            except Exception as error:
                outputs.append(error)

        self.inference.run([tensor], receive)
        self.inference.wait()
        if not outputs:
            raise RuntimeError(
                "Hailo inference completed without returning outputs."
            )
        if isinstance(outputs[0], Exception):
            raise outputs[0]
        return outputs[0]

    def _output(self, outputs, role):
        wanted = str(self.output_roles[role])
        if wanted in outputs:
            return outputs[wanted]
        matches = [
            value for name, value in outputs.items()
            if str(name).rsplit("/", 1)[-1].split(":", 1)[0] == wanted
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve the Hailo {role!r} output {wanted!r}."
            )
        return matches[0]

    def predict(self, image):
        """Return the label, raw score, and dense anomaly map for one RGB image."""
        self._ensure_open()
        interpolation = (
            Image.Resampling.BICUBIC
            if self.algorithm in ("dinomaly", "subspacead")
            else Image.Resampling.BILINEAR
        )
        resized = Image.fromarray(image).convert("RGB").resize(
            (self.input_width, self.input_height), interpolation
        )
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        outputs = self._infer(np.ascontiguousarray(tensor))
        if self.postprocessing == "patchcore_cpu":
            return self._patchcore_prediction(
                np.asarray(self._output(outputs, "features"), dtype=np.float32)
            )
        score_values = np.asarray(
            self._output(outputs, "score"), dtype=np.float32
        )
        map_values = np.asarray(
            self._output(outputs, "anomaly_map"), dtype=np.float32
        )
        if score_values.size != 1:
            raise ValueError(
                "Anomaly Hailo score output must contain one value; "
                f"found {score_values.shape}."
            )
        expected_map_size = self.input_height * self.input_width
        if map_values.size != expected_map_size:
            raise ValueError(
                "Anomaly Hailo map output must contain one score per input "
                "pixel; "
                f"found shape {map_values.shape}."
            )
        return self._prediction(
            float(score_values.reshape(-1)[0]),
            map_values.reshape(self.input_height, self.input_width),
        )

    def _patchcore_prediction(self, features):
        channels, height, width = self.feature_shape[1:]
        if features.size != channels * height * width:
            raise ValueError("PatchCore Hailo feature output has an unexpected size.")
        feature_map = features.reshape(height, width, channels).transpose(2, 0, 1)

        def weights(length):
            result = np.zeros((16, length), dtype=np.float32)
            for index in range(16):
                start, end = index * length // 16, ((index + 1) * length + 15) // 16
                result[index, start:end] = 1.0 / (end - start)
            return result

        pooled = weights(height) @ feature_map @ weights(width).T
        queries = pooled.transpose(1, 2, 0).reshape(-1, channels)
        if self.normalize_features:
            query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / np.maximum(query_norm, np.finfo(np.float32).tiny)
        memory_norm = np.square(self.memory).sum(axis=1)
        distances = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), 32):
            query = queries[start:start + 32]
            squared = (
                np.square(query).sum(axis=1, keepdims=True)
                + memory_norm[None]
                - 2 * query @ self.memory.T
            )
            distances[start:start + len(query)] = np.sqrt(
                np.maximum(squared, 0).min(axis=1)
            )
        anomaly_map = Image.fromarray(distances.reshape(16, 16)).resize(
            (self.input_width, self.input_height), Image.Resampling.BILINEAR
        )
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        score = float(np.quantile(anomaly_map, 0.99, interpolation="linear"))
        return self._prediction(score, anomaly_map)

    def close(self):
        if self._closed:
            return
        self._closed = True
        inference, self.inference = self.inference, None
        if inference is not None:
            inference.close()



__all__ = ["HailoAnomalyModel"]
