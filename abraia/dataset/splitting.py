"""Balanced, deterministic dataset splitting helpers."""

from __future__ import annotations

import random
from collections import defaultdict


SPLIT_NAMES = ("train", "validation", "test")
# Physical dataset directory names used by the training backends. The
# presentation-facing split name is ``validation``; backends use ``val``.
DATASET_SPLITS = ("train", "val", "test")
DEFAULT_EVALUATION_SPLIT = DATASET_SPLITS[-1]
SPLIT_RATIO_KEYS = tuple(f"{name}_ratio" for name in SPLIT_NAMES)
DEFAULT_RANDOM_STATE = 42
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.20,
    "test": 0.10,
}
DEFAULT_SPLIT_OPTIONS = {
    **{
        f"{name}_ratio": DEFAULT_SPLIT_RATIOS[name]
        for name in SPLIT_NAMES
    },
    "random_state": DEFAULT_RANDOM_STATE,
}


def normalize_split_ratios(
    train_ratio=DEFAULT_SPLIT_RATIOS["train"],
    validation_ratio=DEFAULT_SPLIT_RATIOS["validation"],
    test_ratio=DEFAULT_SPLIT_RATIOS["test"],
):
    """Validate and normalize the requested train/validation/test ratios."""
    ratios = {
        "train": float(train_ratio),
        "validation": float(validation_ratio),
        "test": float(test_ratio),
    }
    if any(value < 0 or value > 1 for value in ratios.values()):
        raise ValueError("Split ratios must be between 0 and 1")
    if ratios["train"] <= 0:
        raise ValueError("The train split must be greater than zero")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("Split ratios must add up to 1")
    return ratios


def _target_counts(total, ratios):
    """Allocate a total across splits using largest remainders."""
    raw = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw[name] - counts[name], -SPLIT_NAMES.index(name)),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _annotation_class_counts(annotation):
    """Count labeled objects by class in one image record."""
    counts = defaultdict(int)
    if isinstance(annotation, dict):
        for obj in annotation.get("objects", []) or []:
            if isinstance(obj, dict) and obj.get("label"):
                counts[str(obj["label"])] += 1
    return counts


def _annotation_labels(annotation):
    """Return the unique class labels represented by one image."""
    return set(_annotation_class_counts(annotation))


def _assign_annotated_records(annotations, ratios, random_state):
    """Balance class coverage first and object counts second."""
    rng = random.Random(random_state)
    records = [
        (annotation, _annotation_class_counts(annotation))
        for annotation in annotations
    ]
    records = [record for record in records if record[1]]
    if not records:
        return {name: [] for name in SPLIT_NAMES}

    image_totals = defaultdict(int)
    object_totals = defaultdict(int)
    for _annotation, counts in records:
        for label, count in counts.items():
            image_totals[label] += 1
            object_totals[label] += count
    image_targets = {
        label: _target_counts(total, ratios)
        for label, total in image_totals.items()
    }
    object_targets = {
        label: _target_counts(total, ratios)
        for label, total in object_totals.items()
    }
    total_targets = _target_counts(len(records), ratios)

    rng.shuffle(records)
    records.sort(
        key=lambda item: (
            min(image_totals[label] for label in item[1]),
            -len(item[1]),
        )
    )
    assignments = {name: [] for name in SPLIT_NAMES}
    image_counts = {name: defaultdict(int) for name in SPLIT_NAMES}
    object_counts = {name: defaultdict(int) for name in SPLIT_NAMES}
    for annotation, counts in records:
        def score(split_name):
            image_deficit = sum(
                max(
                    image_targets[label][split_name]
                    - image_counts[split_name][label],
                    0,
                )
                for label in counts
            )
            image_pressure = sum(
                max(
                    image_targets[label][split_name]
                    - image_counts[split_name][label],
                    0,
                ) / max(1, image_targets[label][split_name])
                for label in counts
            )
            object_improvement = sum(
                abs(
                    object_targets[label][split_name]
                    - object_counts[split_name][label]
                )
                - abs(
                    object_targets[label][split_name]
                    - object_counts[split_name][label]
                    - count
                )
                for label, count in counts.items()
            )
            return (
                image_deficit,
                image_pressure,
                object_improvement,
                -len(assignments[split_name]),
            )

        candidates = [
            name for name in SPLIT_NAMES
            if len(assignments[name]) < total_targets[name]
        ] or list(SPLIT_NAMES)
        destination = max(candidates, key=score)
        assignments[destination].append(annotation)
        for label, count in counts.items():
            image_counts[destination][label] += 1
            object_counts[destination][label] += count
    return assignments


def split_annotation_records(
    annotations,
    train_ratio=DEFAULT_SPLIT_RATIOS["train"],
    validation_ratio=DEFAULT_SPLIT_RATIOS["validation"],
    test_ratio=DEFAULT_SPLIT_RATIOS["test"],
    random_state=DEFAULT_RANDOM_STATE,
):
    """Return ``(train, validation, test)`` with class-balanced records.

    Each image remains intact. Class coverage guides assignment first, object
    instance counts second, and confirmed background images follow the requested
    split ratios.
    """
    ratios = normalize_split_ratios(
        train_ratio, validation_ratio, test_ratio
    )
    annotations = list(annotations or [])
    labelled = [
        annotation for annotation in annotations
        if _annotation_labels(annotation)
    ]
    backgrounds = [
        annotation for annotation in annotations
        if not _annotation_labels(annotation)
    ]
    assignments = _assign_annotated_records(labelled, ratios, random_state)
    rng = random.Random(random_state + 1)
    rng.shuffle(backgrounds)
    background_targets = _target_counts(len(backgrounds), ratios)
    offset = 0
    for name in SPLIT_NAMES:
        count = background_targets[name]
        assignments[name].extend(backgrounds[offset:offset + count])
        offset += count
    return tuple(assignments[name] for name in SPLIT_NAMES)


def summarize_split_records(splits, classes=None, ratios=None):
    """Return image, object, and background counts for each split."""
    if len(splits) != len(SPLIT_NAMES):
        raise ValueError("Expected train, validation, and test splits")
    class_names = list(classes or [])
    known = set(class_names)
    for split in splits:
        for annotation in split or []:
            known.update(_annotation_labels(annotation))
    class_names.extend(label for label in sorted(known) if label not in class_names)

    records = []
    backgrounds = {}
    totals = {}
    for split_name, split in zip(SPLIT_NAMES, splits):
        annotations = list(split or [])
        totals[split_name] = len(annotations)
        backgrounds[split_name] = sum(
            not _annotation_labels(annotation) for annotation in annotations
        )
    for class_name in class_names:
        counts = {}
        for split_name, split in zip(SPLIT_NAMES, splits):
            annotations = list(split or [])
            counts[split_name] = sum(
                class_name in _annotation_labels(annotation)
                for annotation in annotations
            )
            counts[f"{split_name}_objects"] = sum(
                _annotation_class_counts(annotation).get(class_name, 0)
                for annotation in annotations
            )
        records.append({"name": str(class_name), **counts})
    result = {
        "mode": "training_split",
        "classes": records,
        "totals": totals,
        "backgrounds": backgrounds,
    }
    if ratios is not None:
        result["ratios"] = normalize_split_ratios(**ratios)
    return result


__all__ = [
    "DATASET_SPLITS",
    "DEFAULT_EVALUATION_SPLIT",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_OPTIONS",
    "DEFAULT_RANDOM_STATE",
    "SPLIT_NAMES",
    "SPLIT_RATIO_KEYS",
    "normalize_split_ratios",
    "split_annotation_records",
    "summarize_split_records",
]
