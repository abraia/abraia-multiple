"""Geometry helpers for detection and segmentation pipeline applications."""

from collections.abc import Mapping
import math


def add_calibrated_measurements(measurements, calibration):
    """Add real-world geometric values using a uniform pixel scale."""
    if not calibration:
        return measurements
    pixels_per_unit = float(calibration["pixels_per_unit"])
    unit = str(calibration["unit"])
    if measurements.get("perimeter") is not None:
        measurements[f"perimeter_{unit}"] = measurements["perimeter"] / pixels_per_unit
    if measurements.get("area") is not None:
        measurements[f"area_{unit}2"] = measurements["area"] / pixels_per_unit ** 2
    if measurements.get("centroid") is not None:
        measurements[f"centroid_{unit}"] = [
            coordinate / pixels_per_unit
            for coordinate in measurements["centroid"]
        ]
    if measurements.get("bounds") is not None:
        x, y, width, height = measurements["bounds"]
        measurements[f"bounds_{unit}"] = [
            x / pixels_per_unit, y / pixels_per_unit,
            width / pixels_per_unit, height / pixels_per_unit,
        ]
    rectangle = measurements.get("obb")
    if rectangle is not None:
        center_x, center_y, width, height, angle = rectangle
        measurements[f"obb_{unit}"] = [
            center_x / pixels_per_unit, center_y / pixels_per_unit,
            width / pixels_per_unit, height / pixels_per_unit, angle,
        ]
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
            centroid = measurements["centroid"]
            bounds = measurements["bounds"]
            if centroid is not None:
                measurements["centroid"] = [
                    centroid[0] + x, centroid[1] + y
                ]
            if bounds is not None:
                measurements["bounds"] = [
                    bounds[0] + x, bounds[1] + y, bounds[2], bounds[3]
                ]
            obb = measurements["obb"]
            if obb is not None:
                obb[0] += x
                obb[1] += y
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
                obb = [*centroid, width, height, 0.0]
                return {
                    "area": area,
                    "perimeter": perimeter,
                    "centroid": centroid,
                    "bounds": bounds,
                    "aspect_ratio": obb[2] / obb[3] if obb[3] else 0.0,
                    "obb": obb,
                    "compactness": (
                        4 * math.pi * area / (perimeter * perimeter)
                        if perimeter else 0.0
                    ),
                    "solidity": 1.0 if area else 0.0,
                    "vertex_count": 4,
                }
    polygon = result_polygon(result)
    if polygon:
        import cv2
        import numpy as np

        contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
        measurements = _contour_measurements(
            contour,
            vertex_count=len(polygon),
        )
        moments = cv2.moments(contour)
        centroid = (
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            if moments["m00"] else polygon_centroid(polygon)
        )
        if centroid is not None:
            centroid = list(centroid)
        xs, ys = zip(*polygon)
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        measurements["centroid"] = centroid
        measurements["bounds"] = [min(xs), min(ys), width, height]
        obb = measurements["obb"]
        measurements["aspect_ratio"] = obb[2] / obb[3] if obb[3] else 0.0
        return measurements
    return _empty_measurements()


def _empty_measurements():
    return {
        "area": 0,
        "perimeter": 0.0,
        "centroid": None,
        "bounds": None,
        "aspect_ratio": 0.0,
        "obb": None,
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
    obb = [float(center_x), float(center_y), float(rotated_width),
           float(rotated_height), float(angle)]
    return {
        "area": area,
        "perimeter": perimeter,
        "centroid": centroid,
        "bounds": bounds,
        "aspect_ratio": float(rotated_width / rotated_height) if rotated_height else 0.0,
        "obb": obb,
        "compactness": (
            float(4 * math.pi * area / (perimeter * perimeter))
            if perimeter else 0.0
        ),
        "solidity": float(area / hull_area) if hull_area else 0.0,
        "vertex_count": int(vertex_count or len(points)),
    }


def mask_measurements(mask):
    """Return pixel measurements for a binary segmentation mask."""
    import cv2
    import numpy as np

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("Segmentation masks must be two-dimensional")
    binary = values.astype(bool).astype(np.uint8)
    area = cv2.countNonZero(binary) if binary.size else 0
    if not area:
        return {**_empty_measurements(), "fraction": 0.0}

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    obb, vertex_count = None, 0
    if len(largest) >= 3:
        center, size, angle = cv2.minAreaRect(largest)
        obb = [*center, *size, angle]
        vertex_count = len(largest)
    x, y, width, height = cv2.boundingRect(binary)
    hull = cv2.convexHull(np.concatenate(contours, axis=0))
    hull_area = area
    if len(hull) >= 3:
        hull_mask = np.zeros_like(binary)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        hull_area = cv2.countNonZero(hull_mask)
    perimeter_kernel = np.array(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.int16,
    )
    perimeter_map = cv2.filter2D(
        binary, cv2.CV_16S, perimeter_kernel, borderType=cv2.BORDER_CONSTANT,
    )
    perimeter = int(cv2.sumElems(cv2.bitwise_and(
        perimeter_map, perimeter_map, mask=binary,
    ))[0])
    moments = cv2.moments(binary, binaryImage=True)
    return {
        "area": area,
        "perimeter": float(perimeter),
        "centroid": [
            moments["m10"] / moments["m00"], moments["m01"] / moments["m00"],
        ],
        "bounds": [x, y, width, height],
        "aspect_ratio": obb[2] / obb[3] if obb and obb[3] else width / height,
        "obb": obb,
        "compactness": (
            float(4 * math.pi * area / (perimeter * perimeter))
            if perimeter else 0.0
        ),
        "solidity": min(1.0, float(area / hull_area)) if hull_area else 0.0,
        "vertex_count": vertex_count,
        "fraction": area / binary.size,
    }


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
