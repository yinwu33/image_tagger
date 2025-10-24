from __future__ import annotations

from typing import Protocol, runtime_checkable

from .llm_prompt import DEFAULT_PROVIDER_CONFIGS, LLMPromptTagger
from .base_tagger import BaseTagger, ImageInput

_PROVIDER_NAMES = {cfg["name"] for cfg in DEFAULT_PROVIDER_CONFIGS}

__all__ = [
    "create_tagger",
    "ImageTagger",
    "ImageInput",
    "DEFAULT_PROVIDER_CONFIGS",
    "LLMPromptTagger",
]


@runtime_checkable
class ImageTagger(Protocol):
    """Common interface for all image tagging backends."""

    def tag(self, image: ImageInput, condition: str | None = None) -> dict | list:
        """
        Generate descriptive tags for an image.

        Args:
            image: Path to the image file or raw image bytes.
            condition: Optional textual hint or instruction.
        """


def create_tagger(name: str = "openai", **kwargs) -> ImageTagger:
    """
    Factory that instantiates an image tagger backend.

    Args:
        name: Tagger backend identifier.
        **kwargs: Extra parameters forwarded to the backend implementation.
    """
    if name == "llm_prompt":
        provider = kwargs.pop("provider", "openai")
        return LLMPromptTagger(provider=provider, **kwargs)
    if name in _PROVIDER_NAMES:
        return LLMPromptTagger(provider=name, **kwargs)

    raise ValueError(f"Unsupported tagger backend: {name!r}")
