import numpy as np
from abraia.inference import ops, Clip


def test_softmax_values():
    logits = np.array([0, 10, -10])
    assert np.isclose(np.sum(ops.softmax(logits)), 1)


def test_clip_import():
    assert Clip is not None
