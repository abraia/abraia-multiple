"""Color palette and contrast helpers."""


_COLORS = [
    "#D0021B", "#F5A623", "#F8E71C", "#8B572A", "#7ED321",
    "#417505", "#BD10E0", "#9013FE", "#4A90E2", "#50E3C2",
    "#B8E986", "#000000", "#545454", "#737373", "#A6A6A6",
    "#D9D9D9", "#FFFFFF",
]


def get_color(idx):
    """Return a palette color for the given index."""
    return _COLORS[idx % (len(_COLORS) - 1)]


def hex_to_rgb(hex):
    """Convert a hexadecimal color string to an RGB tuple."""
    value = hex.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def calculate_contrast_text_color(background_color):
    """Choose black or white text using the shared luminance threshold."""
    r, g, b = background_color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return (0, 0, 0) if brightness > 150 else (255, 255, 255)


__all__ = [
    "calculate_contrast_text_color",
    "get_color",
    "hex_to_rgb",
]
