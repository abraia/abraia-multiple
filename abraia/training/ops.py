import numpy as np


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets"""
    if not arrays:
        raise ValueError("At least one array is required")
    n_samples = len(arrays[0])
    if any(len(array) != n_samples for array in arrays):
        raise ValueError("All arrays must have the same length")
    if stratify is not None and len(stratify) != n_samples:
        raise ValueError("stratify must have the same length as the arrays")
    if test_size is None and train_size is None:
        test_size = 0.25
    if train_size is not None:
        n_train = int(train_size * n_samples) if isinstance(train_size, float) else train_size
        n_test = n_samples - n_train
    else:
        n_test = int(test_size * n_samples) if isinstance(test_size, float) else test_size
        n_train = n_samples - n_test
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        if stratify is not None:
            unique_classes, class_indices = np.unique(stratify, return_inverse=True)
            train_indices, test_indices = [], []
            for i in range(len(unique_classes)):
                cls_indices = indices[class_indices == i]
                rng.shuffle(cls_indices)
                n_cls_test = int(n_test * len(cls_indices) / n_samples)
                test_indices.extend(cls_indices[:n_cls_test])
                train_indices.extend(cls_indices[n_cls_test:])
            indices = np.array(train_indices + test_indices)
            n_train = len(train_indices)
        else:
            rng.shuffle(indices)
    res = []
    for arr in arrays:
        if isinstance(arr, list):
            res.append([arr[i] for i in indices[:n_train]])
            res.append([arr[i] for i in indices[n_train:]])
        else:
            res.append(arr[indices[:n_train]])
            res.append(arr[indices[n_train:]])
    return res


def resample(*arrays, n_samples=None, random_state=None, replace=True):
    """Resample arrays or matrices in a consistent way"""
    if not arrays:
        raise ValueError("At least one array is required")
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("All arrays must have the same length")
    n_samples = len(arrays[0]) if n_samples is None else n_samples
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(arrays[0]), size=n_samples, replace=replace)
    res = []
    for arr in arrays:
        if isinstance(arr, list):
            res.append([arr[i] for i in indices])
        else:
            res.append(arr[indices])
    return res[0] if len(res) == 1 else res
