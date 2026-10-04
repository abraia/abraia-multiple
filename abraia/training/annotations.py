"""Dependency-free dataset annotation metadata helpers."""

from ..datasets import canonical_filename


def image_filename(image):
    """Return the best comparable filename from an image record."""
    return canonical_filename(image.get("name") or image.get("path"))


def find_image(images, filename):
    """Find an image record by name or path basename."""
    key = canonical_filename(filename)
    return next(
        (image for image in images or [] if image_filename(image) == key),
        None,
    )


def objects_for_image(dataset, path, display_name=""):
    """Return saved annotation objects for an image record."""
    if not dataset:
        return []
    filenames = {canonical_filename(display_name), canonical_filename(path)}
    for annotation in dataset.annotations or []:
        if canonical_filename(annotation.get("filename")) in filenames:
            return annotation.get("objects") or []
    return []


def annotation_counts(annotations):
    """Return saved annotation object counts keyed by filename."""
    counts = {}
    for annotation in annotations or []:
        filename = canonical_filename(annotation.get("filename"))
        if filename:
            counts[filename] = counts.get(filename, 0) + len(
                annotation.get("objects") or []
            )
    return counts


def dataset_has_annotations(dataset):
    """Return whether a dataset contains images and annotation metadata."""
    return bool(dataset and dataset.images and dataset.annotations)


__all__ = [
    "annotation_counts",
    "canonical_filename",
    "dataset_has_annotations",
    "find_image",
    "image_filename",
    "objects_for_image",
]
