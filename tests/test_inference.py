import numpy as np
from abraia.inference import ops


def test_softmax_values():
    logits = np.array([0, 10, -10])
    assert np.isclose(np.sum(ops.softmax(logits)), 1)
