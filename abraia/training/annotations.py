"""Dependency-free dataset annotation metadata helpers."""

import os


def canonical_filename(value):
    """Return a comparable filename from a path or display label."""
    if value is None:
        return ""
    lines = str(value).splitlines()
    text = lines[0].strip() if lines else ""
    return os.path.basename(text)


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


def prune_orphaned_annotations(dataset):
    """Remove annotations whose image is no longer in the dataset.

    The dataset image listing is the source of truth. When records are
    removed, persist the cleaned annotation list so future training runs do
    not encounter the same missing file.
    """
    images = getattr(dataset, "images", None) or []
    annotations = getattr(dataset, "annotations", None) or []
    image_names = {image_filename(image) for image in images}
    remaining = [
        annotation
        for annotation in annotations
        if (
            isinstance(annotation, dict)
            and canonical_filename(annotation.get("filename")) in image_names
        )
    ]
    removed = len(annotations) - len(remaining)
    if removed:
        dataset.annotations = remaining
        save = getattr(dataset, "save", None)
        if callable(save):
            save()
    return removed


def upsert_annotation(dataset, filename, objects):
    """Update an image annotation or append a new annotation record."""
    key = canonical_filename(filename)
    annotations = dataset.annotations or []
    dataset.annotations = annotations
    for annotation in annotations:
        if canonical_filename(annotation.get("filename")) == key:
            annotation["objects"] = objects
            return True

    image = find_image(dataset.images, key)
    if not image:
        return False
    annotations.append({
        "url": image.get("url"),
        "filename": image.get("name") or key,
        "objects": objects,
    })
    return True


__all__ = [
    "annotation_counts",
    "canonical_filename",
    "dataset_has_annotations",
    "find_image",
    "image_filename",
    "objects_for_image",
    "prune_orphaned_annotations",
    "upsert_annotation",
]
