"""Pluggable LLM providers for ResearchLens AI.

A tiny abstraction so the rest of the app never imports a vendor SDK directly.
``get_llm()`` returns an object with a single ``generate(prompt, ...) -> str``
method. Add a new provider by writing a class with that method and wiring it
into ``get_llm()``.

If no API key is configured, ``get_llm()`` returns a :class:`NullLLM` that
raises :class:`LLMNotConfigured` when asked to generate. Callers catch this and
show the user a helpful configuration message — the app never pretends to have
produced a real answer without credentials.
"""

from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)


class LLMNotConfigured(Exception):
    """Raised when generation is attempted without a valid API key."""


class LLMError(Exception):
    """Raised when the provider call fails at runtime."""


class NullLLM:
    """Stand-in used when no provider key is set."""

    name = "none"
    available = False

    def generate(self, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        raise LLMNotConfigured(config.missing_key_message())


class GeminiLLM:
    """Google Gemini provider (google-generativeai)."""

    available = True

    def __init__(self, api_key: str, model: str = config.GEMINI_MODEL):
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model)
        self.name = f"gemini:{model}"

    def generate(self, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini request failed: {exc}") from exc


class OpenAILLM:
    """OpenAI provider (openai>=1.0). Optional — only used if selected."""

    available = True

    def __init__(self, api_key: str, model: str = config.OPENAI_MODEL):
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.name = f"openai:{model}"

    def generate(self, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI request failed: {exc}") from exc


_LLM = None


def get_llm():
    """Return a cached provider instance based on config / environment."""
    global _LLM
    if _LLM is not None:
        return _LLM

    if not config.llm_is_configured():
        logger.info("No LLM key configured; using NullLLM.")
        _LLM = NullLLM()
        return _LLM

    try:
        if config.LLM_PROVIDER == "gemini":
            _LLM = GeminiLLM(config.GEMINI_API_KEY)
        elif config.LLM_PROVIDER == "openai":
            _LLM = OpenAILLM(config.OPENAI_API_KEY)
        else:
            _LLM = NullLLM()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize LLM provider: %s", exc)
        _LLM = NullLLM()
    return _LLM


def reset_llm() -> None:
    """Clear the cached provider (used by tests)."""
    global _LLM
    _LLM = None
