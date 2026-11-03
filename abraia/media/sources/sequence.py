"""Lazy image sequences for media and pipeline inputs."""

from functools import partial
import numbers
import os
from pathlib import Path

from ..images import load_image
from .types import IMAGE_SUFFIXES


class ImageSequence:
    """An indexable sequence that decodes frames when accessed."""

    def __init__(self, frames, loader=None, metadata=None):
        if isinstance(frames, (str, bytes, os.PathLike)):
            raise TypeError("frames must be an iterable of frame sources")
        self._sources = tuple(frames)
        if not self._sources:
            raise ValueError("ImageSequence requires at least one frame")
        self._loader = (
            partial(load_image, max_size=None) if loader is None else loader
        )
        self._metadata = dict(metadata or {})
        self.accelerator = None

    @classmethod
    def from_files(cls, paths, loader=None, metadata=None, sort=False):
        """Create a sequence from image paths."""
        paths = list(paths)
        if sort:
            paths.sort(key=os.fspath)
        return cls(paths, loader=loader, metadata=metadata)

    @classmethod
    def from_directory(
        cls, directory, pattern="*", recursive=False, loader=None, metadata=None
    ):
        """Create a sequence from supported image files in a directory."""
        directory = Path(directory)
        paths = (
            directory.rglob(pattern) if recursive else directory.glob(pattern)
        )
        files = sorted(
            path for path in paths
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        return cls.from_files(files, loader=loader, metadata=metadata)

    @classmethod
    def from_array(cls, array, axis=0, metadata=None):
        """Create a sequence from an array whose selected axis is time."""
        ndim = getattr(array, "ndim", None)
        if ndim is None:
            raise TypeError("array must expose an ndim attribute")
        axis = int(axis) % int(ndim)

        def load_frame(index):
            key = [slice(None)] * int(ndim)
            key[axis] = index
            return array[tuple(key)]

        return cls(range(array.shape[axis]), loader=load_frame, metadata=metadata)

    def frame(self, index):
        """Decode and return one frame."""
        index = self._normalize_index(index)
        return self._loader(self._sources[index])

    def validate(self):
        """Validate that every frame has the same shape and data type."""
        first = self.frame(0)
        expected_shape = getattr(first, "shape", None)
        if expected_shape is None:
            raise TypeError("decoded frames must expose a shape attribute")
        expected_shape = tuple(expected_shape)
        expected_dtype = getattr(first, "dtype", None)
        for index in range(1, len(self)):
            frame = self.frame(index)
            shape = tuple(getattr(frame, "shape", ()))
            dtype = getattr(frame, "dtype", None)
            if shape != expected_shape:
                raise ValueError(
                    f"Frame {index} has shape {shape}; expected {expected_shape}"
                )
            if dtype != expected_dtype:
                raise ValueError(
                    f"Frame {index} has dtype {dtype}; expected {expected_dtype}"
                )
        return self

    def _normalize_index(self, index):
        if not isinstance(index, numbers.Integral):
            raise TypeError("frame index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("image sequence index out of range")
        return index

    def __getitem__(self, item):
        if isinstance(item, slice):
            return type(self)(
                self._sources[item],
                loader=self._loader,
                metadata=self._metadata,
            )
        return self.frame(item)

    def __iter__(self):
        for index in range(len(self)):
            yield self.frame(index)

    def __len__(self):
        return len(self._sources)

    @property
    def metadata(self):
        return dict(self._metadata)

    @property
    def sources(self):
        return self._sources

    @property
    def frame_rate(self):
        return float(
            self._metadata.get("frame_rate", self._metadata.get("fps", 0)) or 0
        )

    @property
    def frame_shape(self):
        shape = getattr(self.frame(0), "shape", None)
        if shape is None:
            raise TypeError("decoded frames must expose a shape attribute")
        return tuple(shape)

    @property
    def width(self):
        return self.frame_shape[1]

    @property
    def height(self):
        return self.frame_shape[0]

    @property
    def resolution(self):
        return self.width, self.height

    @property
    def shape(self):
        return (len(self), *self.frame_shape)

    @property
    def dtype(self):
        return getattr(self.frame(0), "dtype", None)

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"{type(self).__name__}(length={len(self)})"


__all__ = ["ImageSequence"]
