"""Pipeline stage for measurements derived from model outputs."""

from collections.abc import Mapping

from .anomaly import anomaly_inspection_configuration, inspect_anomaly_map
from .geometry import (
    add_calibrated_measurements,
    mask_measurements,
    result_measurements,
)


DEFAULT_FEATURES = (
    "area_px", "perimeter_px", "centroid_px", "bounds_px",
    "width_px", "height_px", "aspect_ratio", "orientation_deg",
    "major_axis_px", "minor_axis_px", "compactness", "solidity",
)


def _select_features(measurements, features):
    return {
        name: measurements[name]
        for name in features
        if name in measurements
    }


def model_output_source(model_configuration):
    """Return the geometry source supported by a model configuration."""
    if not isinstance(model_configuration, Mapping):
        return None
    task = model_configuration.get("task")
    if model_configuration.get("kind") == "anomaly" or task == "anomaly":
        return "anomaly"
    if isinstance(task, str) and task in {"detection", "pose"}:
        return "detection"
    if task == "segmentation":
        return "segmentation"
    return None


class FeatureExtractionStep:
    """Measure geometry produced by the preceding model stage."""

    def __init__(self, step_id, configuration, *, models=None):
        self.step_id = str(step_id)
        self.model_entry = next(reversed((models or {}).values()), None)
        model_configuration = (
            self.model_entry["configuration"]
            if self.model_entry else {}
        )
        self.source = model_output_source(model_configuration)
        self.features = tuple(configuration.get("features") or DEFAULT_FEATURES)
        self.calibration = configuration.get("calibration")
        self.time_in_region = bool(configuration.get("time_in_region"))
        self.region = configuration.get("polygon", ())
        self.max_absent_frames = configuration.get(
            "tracker_max_absent_frames", 30
        )
        self._region_entries = {}
        self._region_absent_frames = {}

    def __call__(self, context):
        if self.source is None:
            raise ValueError(
                "Feature extraction requires an earlier detection, "
                "segmentation, or anomaly model"
            )
        if self.source == "anomaly":
            if self.time_in_region:
                raise ValueError(
                    "Time in region requires detections or segmentation results"
                )
            return self._measure_anomaly(context)
        values = context.current
        current_is_results = (
            isinstance(values, list)
            and all(isinstance(value, Mapping) for value in values)
        )
        if not current_is_results:
            values = context.results
        if not isinstance(values, list):
            raise ValueError("Feature extraction requires model results")
        measured = []
        records = [dict(value) for value in values if isinstance(value, Mapping)]
        self._track_regions(records, context)
        for record in records:
            if self.source == "detection":
                geometry = {"box": record.get("box")}
            else:
                geometry = {
                    key: record.get(key)
                    for key in ("mask", "polygon", "points")
                    if record.get(key) is not None
                }
                if "mask" in geometry:
                    geometry["box"] = record.get("box")
            has_geometry = any(value is not None for value in geometry.values())
            measurements = (
                _select_features(result_measurements(geometry), self.features)
                if has_geometry else {}
            )
            add_calibrated_measurements(measurements, self.calibration)
            if (
                self.time_in_region
                and "time_in_region_s" in record.get("measurements", {})
            ):
                measurements["time_in_region_s"] = record["measurements"]["time_in_region_s"]
            record["measurements"] = measurements
            if self.source == "segmentation" and not has_geometry:
                record["measurement_status"] = "No segmentation geometry"
            elif not has_geometry:
                record["measurement_status"] = "No shape geometry"
            measured.append(record)
        if current_is_results:
            context.current = measured
        context.results = measured
        context.artifacts[f"{self.step_id}.results"] = measured
        return context

    def _track_regions(self, records, context):
        if not self.time_in_region:
            return
        from abraia.inference.geometry import point_in_polygon
        if self.max_absent_frames is None:
            raise ValueError("Time in region requires an earlier tracker")
        max_absent_frames = self.max_absent_frames
        trackable = [
            record for record in records
            if isinstance(record.get("box"), (list, tuple))
            and record.get("score") is not None
        ]
        active_ids = set()
        in_region = []
        for record in trackable:
            track_id = record.get("track_id")
            if track_id is None:
                continue
            x, y, width, height = record["box"]
            center = x + width / 2, y + height / 2
            if point_in_polygon(center, self.region):
                active_ids.add(track_id)
                self._region_absent_frames[track_id] = 0
                in_region.append(record)
                start = self._region_entries.setdefault(track_id, context.frame_time)
                record.setdefault("measurements", {})["time_in_region_s"] = max(
                    0.0, round(float(context.frame_time - start), 3)
                )
            else:
                self._region_entries.pop(track_id, None)
                self._region_absent_frames.pop(track_id, None)
        for track_id in list(self._region_entries):
            if track_id not in active_ids:
                self._region_absent_frames[track_id] = (
                    self._region_absent_frames.get(track_id, 0) + 1
                )
                if self._region_absent_frames[track_id] >= max_absent_frames:
                    del self._region_entries[track_id]
                    del self._region_absent_frames[track_id]
        context.metrics["time_in_region"] = {
            "count": len(in_region), "polygon": self.region,
        }

    def _measure_anomaly(self, context):
        model_entry = self.model_entry
        model = model_entry["model"]
        anomaly_map = model.last_normalized_anomaly_map
        if anomaly_map is None:
            regions = []
        else:
            configuration = {"steps": [{
                "type": "model",
                "model": model_entry["configuration"],
            }]}
            summary = inspect_anomaly_map(
                anomaly_map,
                context.frame.shape,
                context.results,
                configuration,
                include_masks=True,
            )
            regions = []
            for index, region in enumerate(summary["defect_regions"]):
                mask = region.pop("mask")
                measured = dict(region)
                measured["id"] = f"{self.step_id}-{index + 1}"
                measured["label"] = f"Anomaly {index + 1}"
                geometry = mask_measurements(mask)
                measured["measurements"] = _select_features(
                    geometry, self.features
                )
                add_calibrated_measurements(
                    measured["measurements"], self.calibration
                )
                regions.append(measured)
            context.metrics["anomaly"] = summary
        context.artifacts[f"{self.step_id}.results"] = regions
        context.metrics["feature_measurements"] = regions
        return context


__all__ = [
    "DEFAULT_FEATURES", "FeatureExtractionStep", "model_output_source",
]
