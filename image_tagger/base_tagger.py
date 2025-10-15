from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


from typing import Union

ImageInput = Union[str, Path, bytes]


class BaseTagger(ABC):
    """Abstract base class for implementing image taggers."""

    @abstractmethod
    def tag(self, image: ImageInput, condition: str | None = None) -> list[str]:
        """Generate descriptive tags for the given image."""

    @staticmethod
    def _load_image_bytes(image: ImageInput) -> bytes:
        if isinstance(image, (str, Path)):
            return Path(image).expanduser().read_bytes()
        if isinstance(image, bytes):
            return image
        raise TypeError(
            "Unsupported image input type. Expected path-like or bytes, "
            f"got {type(image)!r}."
        )
