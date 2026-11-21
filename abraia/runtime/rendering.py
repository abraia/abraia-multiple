"""Shared pipeline renderers for model-specific visual results."""

import copy

from ..utils.annotation_colors import anomaly_color_rgb, label_color_rgb
from .anomaly import mask_anomaly_heatmap


_DECISION_COLORS = {
    "PASS": anomaly_color_rgb("normal"),
    "FAIL": anomaly_color_rgb("anomaly"),
}


def render_region_labels(frame, labels):
    """Draw labels fitted to their regions in frame coordinates."""
    import cv2

    from ..utils.draw import (
        calculate_optimal_text_scale, calculate_optimal_thickness,
        render_label,
    )

    frame_height, frame_width = frame.shape[:2]
    thickness = calculate_optimal_thickness(frame.shape[:2])
    default_scale = calculate_optimal_text_scale(frame.shape[:2])
    padding = thickness * 3
    for label, box, color in labels:
        x, y, width, height = map(int, box)
        if width <= 0 or height <= 0:
            continue
        scale = default_scale
        while True:
            text_width, text_height = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, scale, 1
            )[0]
            if scale <= 0.2 or (
                text_width + 2 * padding <= width
                and text_height + 3 * padding <= height
            ):
                break
            scale *= 0.8
        render_label(
            frame, label,
            (
                max(0, min(x + padding, frame_width - text_width - 2 * padding)),
                max(0, min(y + text_height + 3 * padding, frame_height)),
            ),
            color, text_scale=scale, thickness=thickness,
        )
    return frame


def _render_decisions(frame, results, *, render_anomaly=True, frame_decision=None):
    from ..utils.draw import render_polygon

    frame_height, frame_width = frame.shape[:2]
    frame_box = (0, 0, frame_width, frame_height)
    labels = []
    for result in results:
        if not isinstance(result, dict):
            continue
        anomaly = result.get("anomaly")
        box = result.get("box")
        box = box if box is not None and len(box) else frame_box
        decision = result.get("decision_result")
        if isinstance(anomaly, dict):
            prediction = anomaly["prediction"]
            decision = anomaly["decision"].upper()
            score = anomaly.get("score")
            regions = anomaly.get("defect_regions", ())
            boxes = [box]
        elif "defect_count" in result:
            prediction = result["label"]
            decision = decision or "REVIEW"
            score = result.get("score")
            regions = result["defect_regions"]
            boxes = [region["box"] for region in regions] if decision == "FAIL" else []
            if not boxes:
                boxes = [frame_box]
        else:
            if decision in ("PASS", "FAIL"):
                labels.append((decision, box, _DECISION_COLORS[decision]))
            continue
        if not render_anomaly:
            continue
        for region in regions:
            polygon = region.get("polygon")
            if polygon is not None and len(polygon) >= 3:
                render_polygon(frame, polygon, anomaly_color_rgb("defect"))

        label = prediction.upper()
        if label != decision:
            label = f"{label} {decision}"
        if score is not None:
            label += f" {float(score):.2f}"
        color = _DECISION_COLORS.get(decision) or anomaly_color_rgb(
            "review" if decision == "REVIEW" else prediction
        )
        labels.extend((label, box, color) for box in boxes)
    if frame_decision in ("PASS", "FAIL"):
        labels.append((frame_decision, frame_box, _DECISION_COLORS[frame_decision]))
    return render_region_labels(frame, labels) if labels else frame


def render_anomaly_decisions(frame, results):
    """Draw anomaly decisions and defect contours in frame coordinates."""
    return _render_decisions(frame, (
        result for result in results if isinstance(result, dict)
        and (isinstance(result.get("anomaly"), dict) or "defect_count" in result)
    ))


def render_pipeline_results(frame, results, *, render_anomaly=True, frame_decision=None):
    """Draw model annotations and pipeline decisions over one frame."""
    from ..utils.draw import render_results

    detections = []
    for result in results:
        if render_anomaly and isinstance(result.get("anomaly"), dict):
            result = dict(result)
            result.pop("result", None)
            if result.get("track_id") is None and "color" not in result:
                result["color"] = "#%02X%02X%02X" % label_color_rgb(result)
        detections.append(result)
    frame = render_results(frame, detections)
    return _render_decisions(
        frame, results, render_anomaly=render_anomaly, frame_decision=frame_decision,
    )


class AnomalyPipelineRenderer:
    """Blend calibrated anomaly maps beneath the model's image-level result."""

    def __init__(self, renderer, model, *, roi=()):
        self.renderer = renderer
        self.model = model
        self.roi = roi

    def __call__(self, context):
        anomaly_map = self.model.last_normalized_anomaly_map
        from ..inference.models.anomaly import render_anomaly_heatmap

        if anomaly_map is not None and self.roi:
            anomaly_map = mask_anomaly_heatmap(
                anomaly_map, context.frame.shape, self.roi,
            )
        blended = (
            render_anomaly_heatmap(context.frame, anomaly_map)
            if anomaly_map is not None else context.frame.copy()
        )
        render_context = copy.copy(context)
        render_context.frame = blended
        results = []
        for result in context.results or ():
            if (
                isinstance(result, dict)
                and result.get("label") in ("normal", "anomaly")
            ):
                label = result["label"]
                rendered = dict(result) if anomaly_map is None else {
                    "label": label, "score": result.get("score"),
                    "class_id": result.get("class_id", 0),
                }
                if "defect_count" in result:
                    rendered.update(
                        defect_count=result["defect_count"],
                        defect_regions=result["defect_regions"],
                        decision_result=result.get("decision_result", "REVIEW"),
                    )
                rendered["color"] = "#%02X%02X%02X" % anomaly_color_rgb(label)
                results.append(rendered)
            elif anomaly_map is None:
                results.append(result)
        render_context.results = results
        return (
            self.renderer(render_context)
            if self.renderer
            else blended
        )


__all__ = [
    "AnomalyPipelineRenderer", "render_anomaly_decisions", "render_region_labels",
    "render_pipeline_results",
]
