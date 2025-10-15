from __future__ import annotations

import base64
import imghdr
import json
import re
from typing import Any, Mapping, Sequence

from litellm import completion

from .base_tagger import BaseTagger, ImageInput

DEFAULT_PROVIDER_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "openai",
        "model": "gpt-4.1-mini",
    },
    {
        "name": "azure_openai",
        "model": "azure/gpt-5-mini",
        "api_version": "2024-05-01-preview",
    },
    {
        "name": "gemini",
        "model": "gemini/gemini-2.5-pro",
    },
]
_PROVIDER_LOOKUP = {cfg["name"]: cfg for cfg in DEFAULT_PROVIDER_CONFIGS}


class LLMPromptTagger(BaseTagger):
    """
    Prompt an LLM with an image + instruction to generate tags.

    Uses LiteLLM as the unified interface for multiple LLM providers.
    """

    AVAILABLE_PROVIDERS = tuple(cfg.copy() for cfg in DEFAULT_PROVIDER_CONFIGS)

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str | None = None,
        max_tags: int = 10,
        llm_options: Mapping[str, Any] | None = None,
    ) -> None:
        if provider not in _PROVIDER_LOOKUP:
            raise ValueError(
                f"Unsupported LLM provider: {provider!r}. "
                f"Available providers: {', '.join(_PROVIDER_LOOKUP)}"
            )
        preset = _PROVIDER_LOOKUP[provider]
        self._provider = provider
        self._model = model or preset["model"]
        preset_options = {k: v for k, v in preset.items() if k not in {"name", "model"}}
        user_options = dict(llm_options or {})
        max_tokens = user_options.pop("max_output_tokens", None)
        if max_tokens is not None:
            user_options.setdefault("max_tokens", max_tokens)
        user_options.setdefault("max_tokens", 256)

        self._completion_kwargs: dict[str, Any] = {**preset_options, **user_options}
        self._max_tags = max_tags

    def tag(self, image: ImageInput, condition: str | None = None) -> list[str]:
        image_bytes = self._load_image_bytes(image)
        instruction = self._build_instruction(condition)
        raw_text = self._call_llm(image_bytes=image_bytes, instruction=instruction)
        tags = self._parse_tags(raw_text)
        return tags[: self._max_tags]

    def _build_instruction(self, condition: str | None) -> str:
        instruction = (
            "You are an assistant that summarises images into concise tags. "
            "Return between 3 and {max_tags} lowercase tags sorted by relevance. "
            "Respond ONLY with a JSON array of strings."
        ).format(max_tags=self._max_tags)
        if condition:
            instruction += (
                " Incorporate this additional instruction when deciding on the tags: "
                f"{condition!s}"
            )
        return instruction

    def _call_llm(self, *, image_bytes: bytes, instruction: str) -> str:
        data_url = _image_bytes_to_data_url(image_bytes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        response = completion(
            model=self._model,
            messages=messages,
            **self._completion_kwargs,
        )
        return _extract_text_from_response(response)

    def _parse_tags(self, candidate_text: str) -> list[str]:
        try:
            parsed = json.loads(candidate_text)
            if isinstance(parsed, dict):
                parsed = parsed.get("tags", [])
            if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                tags = [str(item).strip().lower() for item in parsed]
                return [tag for tag in tags if tag]
        except json.JSONDecodeError:
            pass

        tags = [
            token.strip().lower()
            for token in re.split(r"[,;\n]+", candidate_text)
            if token.strip()
        ]
        # Deduplicate while preserving order.
        seen = set()
        ordered_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                ordered_tags.append(tag)
        return ordered_tags


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_type = imghdr.what(None, h=image_bytes) or "jpeg"
    mime = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
    }.get(image_type, "image/jpeg")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_response(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM response did not contain any choices.")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        for fragment in content:
            if isinstance(fragment, dict):
                text = fragment.get("text") or fragment.get("content")
                if text:
                    return text

    raise RuntimeError("Unable to extract text content from LLM response.")
