"""Small protocols shared by inference services and model adapters."""

from typing import Any, Protocol


class InferenceModel(Protocol):
    """Protocol implemented by synchronous inference model adapters."""

    def run(self, image: Any, **kwargs: Any):
        """Run inference for one image."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...


__all__ = ["InferenceModel"]
