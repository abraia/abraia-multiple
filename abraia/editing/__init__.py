"""Image editing API with lazy loading for model-backed implementations."""

from .operations import (
    anonymize_image,
    blur_background,
    build_mask,
    clean_image,
    detect_faces,
    detect_plates,
    detect_smartcrop,
    inpaint_image,
    remove_background,
    smartcrop_image,
    upscale_image,
)

__all__ = [
    'anonymize_image',
    'blur_background',
    'build_mask',
    'clean_image',
    'detect_faces',
    'detect_plates',
    'detect_smartcrop',
    'inpaint_image',
    'remove_background',
    'smartcrop_image',
    'upscale_image',
]
