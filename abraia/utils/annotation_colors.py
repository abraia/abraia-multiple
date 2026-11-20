"""Canonical annotation label color resolution."""

import zlib

# First 32 colors from ColorCET glasbey_bw_minc_20, used under CC BY 4.0.
_ANNOTATION_COLORS = (
    "#D70000", "#8C3CFF", "#028800", "#00ACC7",
    "#98FF00", "#FF7FD1", "#6C004F", "#FFA52F",
    "#583B00", "#005759", "#0000DD", "#00FDCF",
    "#A1756A", "#BCB7FF", "#95B478", "#C004B9",
    "#645474", "#790000", "#0774D8", "#FEF591",
    "#004B00", "#8F7A00", "#FF7266", "#EEB9B9",
    "#5E7E66", "#9BE4FF", "#EC0077", "#A67BB9",
    "#5A00A4", "#04C600", "#9E4B00", "#9C3B50",
)
_SEMANTIC_ANNOTATION_COLORS = {
    "normal": (2, 136, 0),
    "anomaly": (215, 0, 0),
    "background": (119, 118, 111),
}


def label_color_rgb(annotation, class_names=None):
    """Return canonical RGB color for an annotation mapping or label."""
    if isinstance(annotation, str):
        annotation = {"label": annotation}
    annotation = annotation if isinstance(annotation, dict) else {}
    class_id = annotation.get("class_id")
    try:
        class_index = int(class_id) if class_id is not None else 0
    except (TypeError, ValueError):
        class_index = 0
    label = annotation.get("label")
    if label is None:
        index = class_index
    else:
        normalized = str(label).strip().casefold()
        if normalized in _SEMANTIC_ANNOTATION_COLORS:
            return _SEMANTIC_ANNOTATION_COLORS[normalized]
        if class_names is not None:
            index = next(
                (
                    index for index, name in enumerate(class_names)
                    if str(name).strip().casefold() == normalized
                ),
                len(class_names),
            )
        elif class_id is not None:
            index = class_index
        else:
            index = zlib.crc32(normalized.encode("utf-8"))
    color = _ANNOTATION_COLORS[index % len(_ANNOTATION_COLORS)]
    return tuple(int(color[offset:offset + 2], 16) for offset in (1, 3, 5))


def anomaly_color_rgb(prediction):
    """Resolve semantic colors for anomaly predictions and defect overlays."""
    label = str(prediction).strip().casefold()
    if label == "defect":
        label = "anomaly"
    return _SEMANTIC_ANNOTATION_COLORS.get(label, _SEMANTIC_ANNOTATION_COLORS["background"])
