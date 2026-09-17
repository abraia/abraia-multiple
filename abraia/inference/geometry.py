"""Geometry helpers shared by inference and runtime components."""

import cv2
import numpy as np


def is_clockwise(contour):
    """Check if a contour is oriented clockwise."""
    value = 0
    for index, point in enumerate(contour):
        next_point = contour[index + 1] if index < len(contour) - 1 else contour[0]
        value += (next_point[0] - point[0]) * (next_point[1] + point[1])
    return value < 0


def get_merge_point_idx(contour1, contour2):
    """Find the closest point indices between two contours."""
    idx1, idx2 = 0, 0
    distance_min = -1
    for i, point1 in enumerate(contour1):
        for j, point2 in enumerate(contour2):
            distance = (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
            if distance_min < 0 or distance < distance_min:
                distance_min = distance
                idx1, idx2 = i, j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    """Merge two contours at given point indices."""
    contour = list(contour1[:idx1 + 1])
    contour.extend(contour2[idx2:])
    contour.extend(contour2[:idx2 + 1])
    contour.extend(contour1[idx1:])
    return np.array(contour)


def merge_with_parent(contour_parent, contour):
    """Merge a child contour into a parent contour."""
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def approx_contour(contour, approx=0.001):
    """Approximate a contour using the Douglas-Peucker algorithm."""
    epsilon = approx * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)


def approximate_polygon(polygon, approx=0.02):
    """Approximate a polygon using the Douglas-Peucker algorithm."""
    return approx_contour(np.array([polygon]).astype(np.int32), approx).tolist()


def triplet_orientation(point_a, point_b, point_c):
    """Return the orientation of a point triplet."""
    value = (point_c[1] - point_a[1]) * (point_b[0] - point_a[0])
    value -= (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    return 1 if value > 0 else -1 if value < 0 else 0


def segments_intersect(point_a, point_b, point_c, point_d):
    """Check if line segments AB and CD intersect."""
    orientations = (
        triplet_orientation(point_a, point_b, point_c),
        triplet_orientation(point_a, point_b, point_d),
        triplet_orientation(point_c, point_d, point_a),
        triplet_orientation(point_c, point_d, point_b),
    )
    o1, o2, o3, o4 = orientations
    return (1 if o1 > 0 else -1) if o1 != o2 and o3 != o4 else 0


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    x, y = point
    inside = False
    for index in range(len(polygon)):
        (x1, y1), (x2, y2) = polygon[index - 1], polygon[index]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        )
        if intersects:
            inside = not inside
    return inside


def polygon_to_box(polygon):
    """Calculate the bounding box from a polygon."""
    return cv2.boundingRect(np.array(polygon, dtype=np.int32))


__all__ = [
    "approx_contour",
    "approximate_polygon",
    "get_merge_point_idx",
    "is_clockwise",
    "merge_contours",
    "merge_with_parent",
    "point_in_polygon",
    "polygon_to_box",
    "segments_intersect",
    "triplet_orientation",
]
