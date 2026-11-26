"""Anomaly inspection post-processing for pipeline applications."""

from collections.abc import Mapping

import cv2
import numpy as np

from ..inference.geometry import mask_to_polygons


DEFAULT_ANOMALY_INSPECTION = {
    "image_threshold": 0.5,
    "pixel_threshold": 0.5,
    "min_area_px": 0,
    "roi": [],
}

ANOMALY_MEASUREMENT_REGIONS = "anomaly.measurement_regions"


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
    return _inspection_settings(_model_configuration(config).get("inspection"))


def _inspection_settings(inspection):
    inspection = inspection or {}
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
    """Summarize thresholded defect regions, optionally with frame-sized masks."""
    summary = _inspect_anomaly_map(
        anomaly_map, frame_shape, results,
        _model_configuration(configuration).get("inspection"),
        include_masks=include_masks,
    )
    if include_masks:
        import numpy as np

        for region in summary["defect_regions"]:
            x, y, width, height = region["box"]
            mask = np.zeros(frame_shape[:2], dtype=bool)
            mask[y:y + height, x:x + width] = region["mask"]
            region["mask"] = mask
    return summary


def _inspect_anomaly_map(
    anomaly_map, frame_shape, results, inspection=None, *, include_masks=False
):
    """Inspect defects, keeping optional masks cropped to their bounding boxes."""
    values = _inspection_settings(inspection)
    height, width = frame_shape[:2]
    heatmap = cv2.resize(
        np.asarray(anomaly_map, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    valid = _roi_mask(height, width, values["roi"])
    mask = (heatmap > values["pixel_threshold"]) & valid
    _, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    regions = []
    for polygon in mask_to_polygons(mask):
        point_x, point_y = polygon[0]
        component_id = int(component_labels[int(point_y), int(point_x)])
        if not component_id:
            continue
        x, y, box_width, box_height, area = map(int, stats[component_id])
        if area < values["min_area_px"]:
            continue
        box_slice = np.s_[y:y + box_height, x:x + box_width]
        component = component_labels[box_slice] == component_id
        score = float(heatmap[box_slice][component].max())
        region = {
            "polygon": polygon,
            "box": [x, y, box_width, box_height],
            "score": score,
            "area": area,
        }
        if include_masks:
            region["mask"] = component
        regions.append(region)
    regions.sort(key=lambda region: region["area"], reverse=True)
    score = _image_score(results)
    defect_area = sum(region["area"] for region in regions)
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
        "defect_area": defect_area,
        "defect_fraction": (
            float(defect_area / inspected_area) if inspected_area else 0.0
        ),
        "thresholds": {
            "image": values["image_threshold"],
            "pixel": values["pixel_threshold"],
            "min_area_px": values["min_area_px"],
        },
        "roi": values["roi"],
    }


def _regions_without_masks(regions):
    return [{key: value for key, value in region.items() if key != "mask"} for region in regions]


def _consolidate_anomaly_results(
    results, anomaly_map, frame_shape, inspection=None, *, measure_defects=False
):
    """Return image-level records and optional cropped masks for measurement."""
    image_result = next((
        result for result in results
        if isinstance(result, Mapping)
        and (result.get("prediction") or result.get("label")) in ("normal", "anomaly")
    ), None)
    if image_result is None:
        return results, []

    measurement_regions = []
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
        summary = _inspect_anomaly_map(
            anomaly_map, frame_shape, [image_result], inspection, include_masks=measure_defects,
        )
        regions = summary["defect_regions"]
        if measure_defects:
            measurement_regions = regions
            regions = _regions_without_masks(regions)
        decision = summary["decision"].upper()
        prediction = {"PASS": "normal", "FAIL": "anomaly"}.get(decision)

    record = {
        **image_result, "defect_regions": regions, "defect_count": len(regions),
        "decision_result": decision,
    }
    if prediction is not None:
        record.update(
            label=prediction, prediction=prediction,
            class_id=int(prediction == "anomaly"),
        )
    records = [
        record if result is image_result else result
        for result in results
        if not isinstance(result, Mapping) or result.get("label") != "defect"
    ]
    return records, measurement_regions


__all__ = [
    "DEFAULT_ANOMALY_INSPECTION",
    "anomaly_inspection_configuration",
    "inspect_anomaly_map",
]
