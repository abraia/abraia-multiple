"""Output decoders and post-processing for inference models."""

from .geometry import (
    AnnotationGeometry,
    normalize_annotation_geometry,
    normalize_polygon,
    polygon_bounds,
)
from .masks import (
    annotation_color_rgb,
    colored_prediction_layers,
    compare_mask_layers,
    mask_to_box,
    mask_to_polygon,
)

__all__ = [
    "AnnotationGeometry",
    "annotation_color_rgb",
    "colored_prediction_layers",
    "compare_mask_layers",
    "mask_to_box",
    "mask_to_polygon",
    "normalize_annotation_geometry",
    "normalize_polygon",
    "polygon_bounds",
]
