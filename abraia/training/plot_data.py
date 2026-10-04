"""Normalize model metric data for presentation clients."""


def normalize_confusion_matrix(matrix, labels=None):
    """Return rectangular numeric rows and matching class labels."""
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    rows = []
    for row in matrix if matrix is not None else []:
        if not isinstance(row, (list, tuple)):
            continue
        values = []
        for value in row:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
        rows.append(values)
    if not rows or not any(any(value != 0 for value in row) for row in rows):
        return [], []

    columns = max(len(row) for row in rows)
    normalized_rows = [row + [0.0] * (columns - len(row)) for row in rows]
    normalized_labels = [str(label) for label in labels or []]
    while len(normalized_labels) < columns:
        normalized_labels.append(
            "background"
            if len(normalized_labels) == columns - 1
            else str(len(normalized_labels) + 1)
        )
    return normalized_rows, normalized_labels


__all__ = ["normalize_confusion_matrix"]
