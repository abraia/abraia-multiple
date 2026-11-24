"""Geometry helpers for detection and segmentation pipeline applications."""

from collections.abc import Mapping
import math


def add_calibrated_measurements(measurements, calibration):
    """Add real-world geometric values using a uniform pixel scale."""
    if not calibration:
        return measurements
    pixels_per_unit = float(calibration["pixels_per_unit"])
    unit = str(calibration["unit"])
    for name in ("width", "height", "perimeter", "major_axis", "minor_axis"):
        key = f"{name}_px"
        if measurements.get(key) is not None:
            measurements[f"{name}_{unit}"] = measurements[key] / pixels_per_unit
    if measurements.get("area_px") is not None:
        measurements[f"area_{unit}2"] = measurements["area_px"] / pixels_per_unit ** 2
    if measurements.get("centroid_px") is not None:
        measurements[f"centroid_{unit}"] = [
            coordinate / pixels_per_unit
            for coordinate in measurements["centroid_px"]
        ]
    if measurements.get("bounds_px") is not None:
        x, y, width, height = measurements["bounds_px"]
        measurements[f"bounds_{unit}"] = [
            x / pixels_per_unit, y / pixels_per_unit,
            width / pixels_per_unit, height / pixels_per_unit,
        ]
    rectangle = measurements.get("rotated_rectangle_px")
    if rectangle is not None:
        measurements[f"rotated_rectangle_{unit}"] = {
            "center": [value / pixels_per_unit for value in rectangle["center"]],
            "size": [value / pixels_per_unit for value in rectangle["size"]],
            "angle_deg": rectangle["angle_deg"],
        }
    return measurements


def _points(value):
    if not isinstance(value, (list, tuple)):
        return []
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points


def box_polygon(box):
    """Return the four corners of an ``x, y, width, height`` box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return []
    return [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]


def result_polygon(result):
    """Return the best available polygon geometry for a result."""
    if not isinstance(result, Mapping):
        return []
    raw_polygon = result.get("polygon")
    if raw_polygon is None:
        raw_polygon = result.get("points")
    polygon = _points(raw_polygon)
    return polygon or box_polygon(result.get("box"))


def polygon_area(points):
    """Return the absolute area of a polygon in square pixels."""
    points = _points(points)
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def polygon_centroid(points):
    """Return a polygon centroid, falling back to its vertex mean."""
    points = _points(points)
    if not points:
        return None
    area_twice = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if math.isclose(area_twice, 0.0):
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    factor = 1.0 / (3.0 * area_twice)
    return (
        factor * sum(
            (points[index][0] + points[(index + 1) % len(points)][0])
            * (
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
            )
            for index in range(len(points))
        ),
        factor * sum(
            (points[index][1] + points[(index + 1) % len(points)][1])
            * (
                points[index][0] * points[(index + 1) % len(points)][1]
                - points[(index + 1) % len(points)][0] * points[index][1]
            )
            for index in range(len(points))
        ),
    )


def result_measurements(result):
    """Return stable geometric measurements for a detection-like result."""
    if isinstance(result, Mapping) and result.get("mask") is not None:
        measurements = mask_measurements(result["mask"])
        box = result.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x, y = float(box[0]), float(box[1])
            except (TypeError, ValueError):
                return measurements
            centroid = measurements["centroid_px"]
            bounds = measurements["bounds_px"]
            if centroid is not None:
                measurements["centroid_px"] = [
                    centroid[0] + x, centroid[1] + y
                ]
                measurements["centroid"] = measurements["centroid_px"]
            if bounds is not None:
                measurements["bounds_px"] = [
                    bounds[0] + x, bounds[1] + y, bounds[2], bounds[3]
                ]
                measurements["bounds"] = measurements["bounds_px"]
            rectangle = measurements["rotated_rectangle_px"]
            if rectangle is not None:
                rectangle["center"] = [
                    rectangle["center"][0] + x, rectangle["center"][1] + y,
                ]
        return measurements
    if (
        isinstance(result, Mapping)
        and result.get("polygon") is None
        and result.get("points") is None
    ):
        box = result.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x, y, width, height = map(float, box)
            except (TypeError, ValueError):
                pass
            else:
                area = max(0.0, width) * max(0.0, height)
                perimeter = 2 * (max(0.0, width) + max(0.0, height))
                bounds = [x, y, width, height]
                centroid = [x + width / 2, y + height / 2]
                return {
                    "area_px": area,
                    "perimeter_px": perimeter,
                    "centroid_px": centroid,
                    "centroid": centroid,
                    "bounds_px": bounds,
                    "bounds": bounds,
                    "width_px": width,
                    "height_px": height,
                    "aspect_ratio": width / height if height else 0.0,
                    "rotated_rectangle_px": {
                        "center": centroid,
                        "size": [width, height],
                        "angle_deg": 0.0,
                    },
                    "compactness": (
                        4 * math.pi * area / (perimeter * perimeter)
                        if perimeter else 0.0
                    ),
                    "solidity": 1.0 if area else 0.0,
                    "vertex_count": 4,
                }
    polygon = result_polygon(result)
    if polygon:
        import numpy as np

        measurements = _contour_measurements(
            np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2)),
            vertex_count=len(polygon),
        )
        centroid = polygon_centroid(polygon)
        xs, ys = zip(*polygon)
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        measurements["centroid_px"] = list(centroid) if centroid else None
        measurements["centroid"] = measurements["centroid_px"]
        measurements["bounds_px"] = [min(xs), min(ys), width, height]
        measurements["bounds"] = measurements["bounds_px"]
        measurements["width_px"] = float(width)
        measurements["height_px"] = float(height)
        measurements["aspect_ratio"] = float(width / height) if height else 0.0
        return measurements
    return _empty_measurements()


def _empty_measurements():
    return {
        "area_px": 0,
        "perimeter_px": 0.0,
        "centroid_px": None,
        "centroid": None,
        "bounds_px": None,
        "bounds": None,
        "width_px": 0.0,
        "height_px": 0.0,
        "aspect_ratio": 0.0,
        "rotated_rectangle_px": None,
        "compactness": 0.0,
        "solidity": 0.0,
        "vertex_count": 0,
    }


def _contour_measurements(contour, *, vertex_count=None):
    """Measure one contour using source-image pixel coordinates."""
    import cv2
    import numpy as np

    points = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    if len(points) < 3:
        return _empty_measurements()
    area = float(cv2.contourArea(points))
    perimeter = float(cv2.arcLength(points, True))
    x, y, width, height = cv2.boundingRect(points)
    (center_x, center_y), (rotated_width, rotated_height), angle = (
        cv2.minAreaRect(points)
    )
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    bounds = [int(x), int(y), int(width), int(height)]
    centroid = [float(center_x), float(center_y)]
    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "centroid_px": centroid,
        "centroid": centroid,
        "bounds_px": bounds,
        "bounds": bounds,
        "width_px": float(width),
        "height_px": float(height),
        "aspect_ratio": float(width / height) if height else 0.0,
        "rotated_rectangle_px": {
            "center": [float(center_x), float(center_y)],
            "size": [float(rotated_width), float(rotated_height)],
            "angle_deg": float(angle),
        },
        "compactness": (
            float(4 * math.pi * area / (perimeter * perimeter))
            if perimeter else 0.0
        ),
        "solidity": float(area / hull_area) if hull_area else 0.0,
        "vertex_count": int(vertex_count or len(points)),
    }


def mask_measurements(mask):
    """Return pixel measurements for a binary segmentation mask."""
    import numpy as np

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("Segmentation masks must be two-dimensional")
    active = values.astype(bool)
    if not active.any():
        return {**_empty_measurements(), "fraction": 0.0}
    import cv2

    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    points = np.concatenate(contours, axis=0)
    measurements = _contour_measurements(max(contours, key=cv2.contourArea))
    ys, xs = np.nonzero(active)
    hull = cv2.convexHull(points)
    hull_mask = np.zeros(active.shape, dtype=np.uint8)
    if len(hull) >= 3:
        cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 1)
        hull_area = int(hull_mask.sum())
    else:
        hull_area = int(active.sum())
    padded = np.pad(active, 1)
    perimeter = int(
        np.count_nonzero(active & ~padded[:-2, 1:-1])
        + np.count_nonzero(active & ~padded[2:, 1:-1])
        + np.count_nonzero(active & ~padded[1:-1, :-2])
        + np.count_nonzero(active & ~padded[1:-1, 2:])
    )
    measurements.update({
        "area_px": int(active.sum()),
        "perimeter_px": float(perimeter),
        "bounds_px": [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
        ],
        "width_px": float(xs.max() - xs.min() + 1),
        "height_px": float(ys.max() - ys.min() + 1),
        "compactness": (
            float(4 * math.pi * active.sum() / (perimeter * perimeter))
            if perimeter else 0.0
        ),
        "solidity": min(1.0, float(active.sum() / hull_area)) if hull_area else 0.0,
    })
    measurements["aspect_ratio"] = (
        measurements["width_px"] / measurements["height_px"]
        if measurements["height_px"] else 0.0
    )
    measurements["centroid_px"] = [
        float(value) for value in np.mean(np.argwhere(active)[:, ::-1], axis=0)
    ]
    measurements["centroid"] = measurements["centroid_px"]
    measurements["bounds"] = measurements["bounds_px"]
    measurements["fraction"] = float(active.mean())
    return measurements


def anomaly_map_measurements(anomaly_map, threshold):
    """Return pixel measurements for a thresholded anomaly heatmap."""
    import numpy as np

    values = np.asarray(anomaly_map)
    if values.ndim != 2:
        raise ValueError("Anomaly maps must be two-dimensional")
    mask = values > float(threshold)
    measurements = mask_measurements(mask)
    measurements["threshold"] = float(threshold)
    return measurements


__all__ = [
    "box_polygon",
    "anomaly_map_measurements",
    "mask_measurements",
    "polygon_area",
    "polygon_centroid",
    "result_measurements",
    "result_polygon",
]
