"""General polygon coordinate validation and bounds helpers."""

from math import isfinite


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


__all__ = ["normalize_polygon", "polygon_bounds"]
