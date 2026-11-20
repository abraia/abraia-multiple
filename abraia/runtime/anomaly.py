"""Anomaly inspection post-processing for pipeline applications."""

from collections.abc import Mapping
import copy

from ..utils.annotation_colors import anomaly_color_rgb


DEFAULT_ANOMALY_INSPECTION = {
    "image_threshold": 0.5,
    "pixel_threshold": 0.5,
    "min_area_px": 0,
    "roi": [],
}


def _model_configuration(config):
    for step in config.get("steps", []) or ():
        if (
            not isinstance(step, Mapping)
            or step.get("type") != "model"
            or step.get("enabled") is False
        ):
            continue
        model = step.get("model") or {}
        if (
            model.get("kind") == "anomaly"
            and model.get("task") == "anomaly"
        ):
            return model
    return {}


def _points(value):
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    return points


def anomaly_inspection_configuration(config):
    """Return validated anomaly inspection settings from a pipeline config."""
    model = _model_configuration(config)
    inspection = model.get("inspection") or {}
    if not isinstance(inspection, Mapping):
        raise ValueError("Anomaly inspection settings must be an object")
    values = dict(DEFAULT_ANOMALY_INSPECTION)
    values.update({key: inspection[key] for key in values if key in inspection})
    try:
        values["image_threshold"] = float(values["image_threshold"])
        values["pixel_threshold"] = float(values["pixel_threshold"])
        values["min_area_px"] = int(values["min_area_px"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Anomaly inspection thresholds must be numeric"
        ) from error
    if not 0 <= values["image_threshold"] <= 1:
        raise ValueError("Anomaly image threshold must be between 0 and 1")
    if not 0 <= values["pixel_threshold"] <= 1:
        raise ValueError("Anomaly pixel threshold must be between 0 and 1")
    if values["min_area_px"] < 0:
        raise ValueError("Anomaly minimum area must not be negative")
    values["roi"] = _points(values["roi"])
    if values["roi"] and len(values["roi"]) < 3:
        raise ValueError(
            "Anomaly inspection ROI requires at least three points"
        )
    return values


def _image_score(results):
    for result in results or ():
        if not isinstance(result, Mapping):
            continue
        if (result.get("prediction") or result.get("label")) not in ("normal", "anomaly"):
            continue
        try:
            return float(result["score"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _roi_mask(height, width, roi):
    import cv2
    import numpy as np

    mask = np.ones((height, width), dtype=np.uint8)
    if roi:
        mask.fill(0)
        polygon = np.round(np.asarray(roi)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def mask_anomaly_heatmap(anomaly_map, frame_shape, roi):
    """Suppress heatmap values outside the configured frame region."""
    import numpy as np

    height, width = anomaly_map.shape[:2]
    frame_height, frame_width = frame_shape[:2]
    polygon = [
        [x * width / frame_width, y * height / frame_height] for x, y in roi
    ]
    return np.where(_roi_mask(height, width, polygon), anomaly_map, 0.5)


def inspect_anomaly_map(
    anomaly_map, frame_shape, results, configuration, *, include_masks=False
):
    """Summarize thresholded defect regions, optionally including pixel masks."""
    import cv2
    import numpy as np

    values = anomaly_inspection_configuration(configuration)
    height, width = frame_shape[:2]
    heatmap = cv2.resize(
        np.asarray(anomaly_map, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    valid = _roi_mask(height, width, values["roi"])
    mask = (heatmap > values["pixel_threshold"]) & valid
    component_count, component_labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    regions = []
    for component_id in range(1, component_count):
        component = component_labels == component_id
        area = int(component.sum())
        if area < values["min_area_px"]:
            continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea)
        polygon = contour[:, 0, :].astype(int).tolist()
        x, y, box_width, box_height = cv2.boundingRect(contour)
        score = float(heatmap[component].max())
        region = {
            "polygon": polygon,
            "box": [int(x), int(y), int(box_width), int(box_height)],
            "score": score,
            "area_px": area,
        }
        if include_masks:
            region["mask"] = component
        regions.append(region)
    regions.sort(key=lambda region: region["area_px"], reverse=True)
    score = _image_score(results)
    defect_area = sum(region["area_px"] for region in regions)
    inspected_area = int(valid.sum())
    if score is None:
        decision = "review"
    elif score > values["image_threshold"] or regions:
        decision = "fail"
    else:
        decision = "pass"
    return {
        "decision": decision,
        "score": score,
        "defect_regions": regions,
        "defect_area_px": defect_area,
        "defect_fraction": (
            float(defect_area / inspected_area) if inspected_area else 0.0
        ),
        "thresholds": {
            "image": values["image_threshold"],
            "pixel": values["pixel_threshold"],
            "min_area_px": values["min_area_px"],
        },
        "roi": copy.deepcopy(values["roi"]),
    }


def consolidate_anomaly_results(results, anomaly_map, frame_shape, inspection=None):
    # Keep the image decision and its accepted defect regions in one record.
    image_result = next((
        result for result in results
        if isinstance(result, Mapping)
        and (result.get("prediction") or result.get("label")) in ("normal", "anomaly")
    ), None)
    if image_result is None:
        return results

    if anomaly_map is None:
        regions = [
            result for result in results
            if isinstance(result, Mapping) and result.get("label") == "defect"
        ]
        prediction = (
            "anomaly" if regions else (
                image_result.get("prediction") or image_result["label"]
            )
        )
        decision = "FAIL" if prediction == "anomaly" else "PASS"
    else:
        config = {"steps": [{"type": "model", "model": {
            "kind": "anomaly", "task": "anomaly", "inspection": inspection,
        }}]}
        summary = inspect_anomaly_map(anomaly_map, frame_shape, [image_result], config)
        regions = summary["defect_regions"]
        decision = summary["decision"].upper()
        prediction = {"PASS": "normal", "FAIL": "anomaly"}.get(decision)

    record = {**image_result, "defect_regions": regions, "defect_count": len(regions)}
    if prediction is not None:
        record.update(
            label=prediction, prediction=prediction,
            class_id=int(prediction == "anomaly"), decision_result=decision,
        )
    return [
        record if result is image_result else result
        for result in results
        if not isinstance(result, Mapping) or result.get("label") != "defect"
    ]


class AnomalyInspectionRenderer:
    """Add accepted defect region overlays to a renderer."""

    def __init__(self, renderer, model, configuration, *, draw_defects=True):
        self.renderer = renderer
        self.model = model
        self.configuration = configuration
        self.draw_defects = draw_defects

    def __call__(self, context):
        anomaly_map = self.model.last_normalized_anomaly_map
        if anomaly_map is None:
            return (
                self.renderer(context)
                if self.renderer else context.frame.copy()
            )

        values = anomaly_inspection_configuration(self.configuration)
        if values["roi"]:
            self.model.last_normalized_anomaly_map = mask_anomaly_heatmap(
                anomaly_map, context.frame.shape, values["roi"],
            )
        try:
            frame = (
                self.renderer(context)
                if self.renderer
                else context.frame.copy()
            )
        finally:
            self.model.last_normalized_anomaly_map = anomaly_map

        if not self.draw_defects:
            return frame

        summary = inspect_anomaly_map(
            anomaly_map, context.frame.shape, context.results, self.configuration,
        )
        import cv2
        import numpy as np

        for region in summary["defect_regions"]:
            points = np.asarray(region["polygon"], dtype="int32")
            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color=anomaly_color_rgb("defect"),
                thickness=3,
            )
        return frame


__all__ = [
    "AnomalyInspectionRenderer",
    "DEFAULT_ANOMALY_INSPECTION",
    "anomaly_inspection_configuration",
    "inspect_anomaly_map",
]
