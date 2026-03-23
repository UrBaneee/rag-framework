"""OpenAI LLM client — wraps the OpenAI Chat Completions API."""

import json
import logging
import time
from typing import Any, Optional

from rag.core.interfaces.llm_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.2


class OpenAILLMClient(BaseLLMClient):
    """LLM client backed by the OpenAI Chat Completions API.

    Sends a single user message to the chosen model and returns the
    generated text together with token usage and wall-clock latency.

    Structured output is supported via ``generate_structured()``, which
    sets ``response_format={"type": "json_object"}`` and parses the
    response as JSON.

    Usage::

        client = OpenAILLMClient(model="gpt-4o-mini")
        response = client.generate("Summarise retrieval augmented generation.")
        print(response.text, response.total_tokens, response.latency_ms)

    Args:
        model: OpenAI model identifier. Defaults to ``"gpt-4o-mini"``.
        api_key: OpenAI API key. If None, reads ``OPENAI_API_KEY`` env var.
        max_tokens: Maximum tokens in the completion. Defaults to 1024.
        temperature: Sampling temperature. Defaults to 0.2.
        system_prompt: Optional system message prepended to every request.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLMClient. "
                "Install with: pip install openai"
            ) from exc

        from openai import OpenAI  # re-import for type narrowing

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # ── BaseLLMClient ─────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Identifier of the model used by this client."""
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a prompt to the OpenAI Chat Completions API.

        Args:
            prompt: The full prompt string (sent as a user message).
            **kwargs: Overrides for ``max_tokens``, ``temperature``,
                and any other ChatCompletion parameter.

        Returns:
            LLMResponse with text, token counts, and latency populated.

        Raises:
            openai.OpenAIError: On API-level errors.
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": kwargs.pop("temperature", self._temperature),
            **kwargs,
        }

        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            logger.error("OpenAI chat completion failed: %s", exc)
            raise
        latency_ms = (time.monotonic() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage

        logger.debug(
            "OpenAI generate: model=%s tokens=%d latency=%.0fms",
            self._model,
            usage.total_tokens if usage else 0,
            latency_ms,
        )

        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Generate a JSON-structured response.

        Uses ``response_format={"type": "json_object"}`` to request JSON
        output from the model. The returned text is parsed and returned as
        a dict, or None if parsing fails.

        Args:
            prompt: Prompt string (should instruct the model to return JSON).
            schema: JSON Schema dict (for documentation — not enforced by API).
            **kwargs: Additional ChatCompletion parameters.

        Returns:
            Parsed dict on success, or None if JSON parsing fails.
        """
        kwargs["response_format"] = {"type": "json_object"}
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse structured JSON response: %s", exc)
            return None

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken (if available) or char heuristic.

        Falls back to ``len(text) // 4`` if tiktoken is not installed,
        matching the ~1 token per 4 characters approximation used elsewhere.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated integer token count.
        """
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            # tiktoken not installed or model not recognised — use heuristic
            return max(1, len(text) // 4)
