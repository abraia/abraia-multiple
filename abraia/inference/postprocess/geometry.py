"""General polygon coordinate validation and bounds helpers."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AnnotationGeometry:
    """Validated polygon or box coordinates in source-image space."""

    points: tuple = ()
    box: tuple = ()
    invalid_box: bool = False


def normalize_annotation_geometry(annotation):
    """Return valid annotation geometry, preferring a valid polygon."""
    if not isinstance(annotation, dict):
        return AnnotationGeometry()

    points = normalize_polygon(annotation.get("polygon", []))
    if points:
        return AnnotationGeometry(points=tuple(points))

    box = annotation.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            coordinates = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            return AnnotationGeometry(invalid_box=True)
        if all(isfinite(value) for value in coordinates):
            _x, _y, width, height = coordinates
            if width > 0 and height > 0:
                return AnnotationGeometry(box=coordinates)
        return AnnotationGeometry(invalid_box=True)
    return AnnotationGeometry()


def normalize_polygon(points):
    """Return finite ``(x, y)`` coordinates, or an empty invalid polygon."""
    normalized = []
    for point in points if points is not None else ():
        try:
            if hasattr(point, "x") and hasattr(point, "y"):
                x, y = point.x(), point.y()
            else:
                x, y = point[:2]
            x, y = float(x), float(y)
        except (TypeError, ValueError, IndexError):
            return []
        if not isfinite(x) or not isfinite(y):
            return []
        normalized.append((x, y))
    return normalized if len(normalized) >= 3 else []


def polygon_bounds(points):
    """Return an ``(x, y, width, height)`` tuple for a valid polygon."""
    normalized = normalize_polygon(points)
    if not normalized:
        return None
    xs, ys = zip(*normalized)
    minimum_x, maximum_x = min(xs), max(xs)
    minimum_y, maximum_y = min(ys), max(ys)
    return (
        minimum_x,
        minimum_y,
        maximum_x - minimum_x,
        maximum_y - minimum_y,
    )


__all__ = [
    "AnnotationGeometry",
    "normalize_annotation_geometry",
    "normalize_polygon",
    "polygon_bounds",
]
