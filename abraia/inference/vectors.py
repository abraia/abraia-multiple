"""Vector distance and similarity helpers."""

import numpy as np


def euclidean_distance(feat1, feat2):
    """Compute the Euclidean distance between two feature vectors."""
    return float(np.linalg.norm(feat1 - feat2))


def cosine_similarity(feat1, feat2):
    """Compute pairwise cosine similarity between two vectors."""
    return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))


def search_vector(vector, index, max_results=1):
    """Search for nearest vectors in an index using cosine similarity."""
    idxs, scores = search_vectors(np.asarray(vector)[None, :], index, max_results)
    return idxs[0], scores[0]


def search_vectors(vectors, index, max_results=1):
    """Search several query vectors against an index in one vectorized pass."""
    if max_results < 1:
        raise ValueError("max_results must be at least 1")
    queries = np.asarray(vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries[None, :]
    if queries.ndim != 2:
        raise ValueError("vectors must be a one- or two-dimensional array")
    if not index:
        return [np.array([], dtype=np.int64) for _ in queries], [[] for _ in queries]

    index_vectors = np.asarray([row["vector"] for row in index], dtype=np.float32)
    if index_vectors.ndim != 2 or index_vectors.shape[1] != queries.shape[1]:
        raise ValueError("Query and index vectors must have matching dimensions")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    index_norms = np.linalg.norm(index_vectors, axis=1, keepdims=True)
    normalized_queries = np.divide(
        queries, query_norms, out=np.zeros_like(queries), where=query_norms != 0
    )
    normalized_index = np.divide(
        index_vectors,
        index_norms,
        out=np.zeros_like(index_vectors),
        where=index_norms != 0,
    )
    similarities = normalized_queries @ normalized_index.T
    count = min(max_results, len(index))
    order = np.argsort(similarities, axis=1)[:, -count:][:, ::-1]
    return [row for row in order], [
        similarities[i, row].tolist() for i, row in enumerate(order)
    ]


def normalize_vector(vector):
    """Normalize a vector to unit length."""
    return vector / np.linalg.norm(vector)


__all__ = [
    "cosine_similarity",
    "euclidean_distance",
    "normalize_vector",
    "search_vector",
    "search_vectors",
]
