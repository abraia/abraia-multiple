import numpy as np
import pytest

from abraia.media import ImageSequence


def test_image_sequence_loads_frames_lazily_without_retaining_them():
    calls = []

    def loader(source):
        calls.append(source)
        return np.full((2, 3, 3), source, dtype=np.uint8)

    sequence = ImageSequence([1, 2, 3], loader=loader)

    assert calls == []
    assert sequence.shape == (3, 2, 3, 3)
    assert calls == [1]
    assert np.array_equal(sequence[0], sequence[0])
    assert calls == [1, 1, 1]
    assert np.array_equal(sequence[-1], np.full((2, 3, 3), 3, dtype=np.uint8))
    assert calls == [1, 1, 1, 3]


def test_image_sequence_slice_preserves_order_loader_and_metadata():
    sequence = ImageSequence(
        ["frame-2", "frame-1", "frame-3"],
        loader=lambda source: np.zeros((4, 5), dtype=np.uint16),
        metadata={"timestamps": [2, 1, 3]},
    )

    selected = sequence[1:]

    assert selected.sources == ("frame-1", "frame-3")
    assert selected.metadata == {"timestamps": [2, 1, 3]}
    assert selected.frame_shape == (4, 5)


def test_image_sequence_from_array_supports_nonzero_time_axis():
    cube = np.arange(2 * 3 * 4).reshape(3, 2, 4)

    sequence = ImageSequence.from_array(cube, axis=1)

    assert sequence.shape == (2, 3, 4)
    assert np.array_equal(sequence[1], cube[:, 1, :])


def test_image_sequence_from_array_keeps_frames_backed_by_input_array():
    cube = np.arange(3 * 4 * 2).reshape(3, 4, 2)

    sequence = ImageSequence.from_array(cube)

    assert np.shares_memory(sequence[0], cube)


def test_image_sequence_validates_frame_shape_and_dtype():
    sequence = ImageSequence(
        [0, 1],
        loader=lambda source: (
            np.zeros((2, 2), dtype=np.uint8)
            if source == 0
            else np.zeros((3, 2), dtype=np.uint8)
        ),
    )

    with pytest.raises(ValueError, match="Frame 1 has shape"):
        sequence.validate()


def test_image_sequence_from_directory_uses_sorted_files(tmp_path):
    for name in ("frame-10.jpg", "frame-02.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    sequence = ImageSequence.from_directory(tmp_path, pattern="*.jpg")

    assert [path.name for path in sequence.sources] == [
        "frame-02.jpg", "frame-10.jpg"
    ]
