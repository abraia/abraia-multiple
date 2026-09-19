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


def _annotation_labels(annotation):
    """Return the unique class labels represented by one annotation record."""
    if not isinstance(annotation, dict):
        return set()
    labels = set()
    for obj in annotation.get("objects", []) or []:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label")
        if label:
            labels.add(str(label))
    return labels


def _assign_annotated_records(annotations, ratios, random_state):
    """Assign labelled records while minimizing per-class ratio deficits."""
    rng = random.Random(random_state)
    records = [(annotation, _annotation_labels(annotation)) for annotation in annotations]
    records = [record for record in records if record[1]]
    if not records:
        return {name: [] for name in SPLIT_NAMES}

    class_records = defaultdict(int)
    for _annotation, labels in records:
        for label in labels:
            class_records[label] += 1
    class_targets = {
        label: _target_counts(total, ratios)
        for label, total in class_records.items()
    }
    total_targets = _target_counts(len(records), ratios)

    # Place rare and multi-label records first. The random tie breaker keeps
    # the result reproducible while avoiding input-order bias.
    rng.shuffle(records)
    records.sort(key=lambda item: (min(class_records[label] for label in item[1]),
                                   -len(item[1])))

    assignments = {name: [] for name in SPLIT_NAMES}
    class_counts = {
        name: defaultdict(int)
        for name in SPLIT_NAMES
    }
    for annotation, labels in records:
        def score(split_name):
            class_deficit = sum(
                max(class_targets[label][split_name] - class_counts[split_name][label], 0)
                for label in labels
            )
            class_pressure = sum(
                (class_targets[label][split_name] - class_counts[split_name][label])
                / max(1, class_targets[label][split_name])
                for label in labels
            )
            total_deficit = max(
                total_targets[split_name] - len(assignments[split_name]), 0
            )
            return (
                class_deficit,
                class_pressure,
                total_deficit,
                -len(assignments[split_name]),
            )

        destination = max(SPLIT_NAMES, key=score)
        assignments[destination].append(annotation)
        for label in labels:
            class_counts[destination][label] += 1
    return assignments


def split_annotation_records(
    annotations,
    train_ratio=DEFAULT_SPLIT_RATIOS["train"],
    validation_ratio=DEFAULT_SPLIT_RATIOS["validation"],
    test_ratio=DEFAULT_SPLIT_RATIOS["test"],
    random_state=DEFAULT_RANDOM_STATE,
):
    """Return ``(train, validation, test)`` with class-balanced records.

    Each annotated image is assigned once. Multi-label images contribute to
    each represented class, while background images remain in training as in
    the original training workflow.
    """
    ratios = normalize_split_ratios(
        train_ratio, validation_ratio, test_ratio
    )
    annotations = list(annotations or [])
    labelled = [annotation for annotation in annotations if _annotation_labels(annotation)]
    backgrounds = [
        annotation for annotation in annotations
        if not _annotation_labels(annotation)
    ]
    assignments = _assign_annotated_records(labelled, ratios, random_state)
    assignments["train"].extend(backgrounds)
    return tuple(assignments[name] for name in SPLIT_NAMES)


def summarize_split_records(splits, classes=None, ratios=None):
    """Return class counts and split totals for a rendered balance chart."""
    if len(splits) != len(SPLIT_NAMES):
        raise ValueError("Expected train, validation, and test splits")
    class_names = list(classes or [])
    known = set(class_names)
    for split in splits:
        for annotation in split or []:
            known.update(_annotation_labels(annotation))
    class_names.extend(label for label in sorted(known) if label not in class_names)

    records = []
    for class_name in class_names:
        counts = {}
        for split_name, split in zip(SPLIT_NAMES, splits):
            counts[split_name] = sum(
                class_name in _annotation_labels(annotation)
                for annotation in split or []
            )
        records.append({"name": str(class_name), **counts})
    result = {
        "classes": records,
        "totals": {
            split_name: len(split or [])
            for split_name, split in zip(SPLIT_NAMES, splits)
        },
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
