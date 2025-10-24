from __future__ import annotations

import base64, json, re, yaml
from typing import Any, Mapping, Sequence
from pathlib import Path
from collections.abc import Sequence
from litellm import completion
from jinja2 import Environment, BaseLoader

from .base_tagger import BaseTagger, ImageInput


DEFAULT_PROVIDER_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "openai",
        "model": "gpt-4.1-mini",
    },
    {
        "name": "azure_openai",
        "model": "azure/gpt-5-mini",
        # "api_version": "2024-05-01-preview",
    },
    {
        "name": "gemini",
        "model": "gemini/gemini-2.5-pro",
    },
]
_PROVIDER_LOOKUP = {cfg["name"]: cfg for cfg in DEFAULT_PROVIDER_CONFIGS}


def load_prompt(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def render_prompt(tpl: str, **kwargs) -> str:
    env = Environment(
        loader=BaseLoader, autoescape=False, trim_blocks=True, lstrip_blocks=True
    )
    return env.from_string(tpl).render(**kwargs)


class PromptProvider:
    """Load + render prompts from files / registry."""

    def __init__(self, path: str):
        self._cfg = load_prompt(path)

    def render(self, **kwargs) -> dict:
        required = set(self._cfg.get("placeholders", []))
        missing = required - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")

        system = render_prompt(self._cfg["system"], **kwargs)
        user = render_prompt(self._cfg["user"], **kwargs)

        if rules := self._cfg.get("rules"):
            user = user + "\n" + "\n".join([f"- {r}" for r in rules])

        return {"system": system, "user": user}


class LLMPromptTagger(BaseTagger):
    """
    Prompt an LLM with an image + instruction to generate tags.

    Uses LiteLLM as the unified interface for multiple LLM providers.
    """

    AVAILABLE_PROVIDERS = tuple(cfg.copy() for cfg in DEFAULT_PROVIDER_CONFIGS)

    def __init__(
        self,
        *,
        prompt_path: str = Path(__file__).parent / "prompts/traffic_sign_v1.yaml",
        provider: str = "openai",
        model: str | None = None,
        max_tags: int = 10,
        llm_options: Mapping[str, Any] | None = None,
    ) -> None:
        if provider not in _PROVIDER_LOOKUP:
            raise ValueError(
                f"Unsupported LLM provider: {provider!r}. "
                f"Available providers: {', '.join(_PROVIDER_LOOKUP.keys())}"
            )
        preset = _PROVIDER_LOOKUP[provider]
        self._provider = provider
        self._model = model or preset["model"]

        preset_options = {k: v for k, v in preset.items() if k not in {"name", "model"}}
        user_options = dict(llm_options or {})

        # normalize max tokens key
        max_tokens = user_options.pop("max_output_tokens", None)
        if max_tokens is not None:
            user_options.setdefault("max_tokens", max_tokens)
        user_options.setdefault("max_tokens", 2048)

        self._completion_kwargs: dict[str, Any] = {**preset_options, **user_options}
        self._max_tags = max_tags

        self._prompt = PromptProvider(prompt_path)

    # ---------- public API ----------
    def tag(self, image: ImageInput, keys: dict = {}) -> list | dict:
        image_bytes = self._load_image_bytes(image)
        messages = self._build_messages(**keys)
        raw_text = self._call_llm(image_bytes=image_bytes, messages=messages)
        tags = self._parse_tags(raw_text)
        return tags[: self._max_tags]

    # ---------- internals ----------
    def _build_messages(self, **kwargs) -> list[dict[str, Any]]:
        payload = self._prompt.render(**kwargs)
        return [
            {"role": "system", "content": payload["system"]},
            {"role": "user", "content": payload["user"]},
        ]

    def _call_llm(self, *, image_bytes: bytes, messages: list[dict[str, Any]]) -> str:
        """
        Append the image to the *last user message* and call LLM.
        """
        data_url = _image_bytes_to_data_url(image_bytes)

        # Find last user message; if none, create one.
        idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                idx = i
                break
        if idx is None:
            messages.append({"role": "user", "content": ""})
            idx = len(messages) - 1

        # Normalize "content" to list-of-parts form
        content = messages[idx].get("content", "")
        parts: list[Any]
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = content
        else:
            parts = [{"type": "text", "text": str(content)}]

        # Append image part
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
        messages[idx]["content"] = parts

        # print("CALLLLLLLLLLLL")
        response = completion(
            model=self._model,
            messages=messages,
            **self._completion_kwargs,
        )
        return _extract_text_from_response(response)

    def _parse_tags(self, candidate_text: str):
        """Parse candidate text into a list or dict of tags.

        Returns:
            list | dict | None
        """
        text = candidate_text.strip()

        # --- strip code fences ```...``` ---
        fence = re.match(r"^```[a-zA-Z0-9]*\s*(.*?)\s*```$", text, re.S)
        if fence:
            text = fence.group(1).strip()

        # --- try JSON first ---
        try:
            parsed = json.loads(text)
            # ensure only list or dict are accepted
            if isinstance(parsed, (list, dict)):
                return parsed
            # sometimes model wraps with {"tags": [...]} — handle that
            if isinstance(parsed, dict) and "tags" in parsed:
                return parsed["tags"]
        except json.JSONDecodeError:
            pass

        # --- fallback: split plain text into list[str] ---
        tokens = [
            tok.strip().lower() for tok in re.split(r"[,;\n]+", text) if tok.strip()
        ]

        # remove duplicates while preserving order
        seen, out = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                out.append(t)
        if not out:
            return None
        return out[: self._max_tags]


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    # NOTE: imghdr is deprecated in Python 3.13; we use a simple JPEG default.
    # If you prefer stricter detection, consider Pillow (PIL) or python-magic.
    try:
        from PIL import Image
        from io import BytesIO

        fmt = Image.open(BytesIO(image_bytes)).format or "JPEG"
        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "GIF": "image/gif",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
            "TIF": "image/tiff",
        }
        mime = mime_map.get(fmt.upper(), "image/jpeg")
    except Exception:
        mime = "image/jpeg"

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_response(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LLM response did not contain any choices.")
    first = choices[0]
    message = getattr(first, "message", None)
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        for frag in content:
            if isinstance(frag, dict):
                text = frag.get("text") or frag.get("content")
                if text:
                    return text

    raise RuntimeError("Unable to extract text content from LLM response.")
