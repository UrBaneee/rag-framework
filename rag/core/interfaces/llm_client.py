"""Abstract base class for LLM client plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# LLMResponse — carries generated text, usage, and latency
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """The response returned by an LLM client generate call.

    Attributes:
        text: Generated text content from the model.
        model: Identifier of the model that produced this response.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Sum of prompt and completion tokens.
        latency_ms: Wall-clock time for the API call in milliseconds.
        structured: Parsed structured output (dict), if a JSON schema was
            requested and the model returned valid JSON. None otherwise.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    structured: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# BaseLLMClient — interface all LLM client plugins must implement
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    """Interface that all LLM client plugins must implement.

    LLM clients send prompts to language models and return generated text
    along with token usage and latency information for observability.

    Implementations include OpenAI, Azure OpenAI, and local Ollama models.

    Core methods:
    - ``generate``          — primary method, returns full ``LLMResponse``
    - ``complete``          — convenience wrapper, returns text string only
    - ``generate_structured`` — optional structured (JSON) output
    - ``count_tokens``      — token budget estimation without an API call
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the model used by this client.

        Returns:
            Model identifier string, e.g. ``"gpt-4o-mini"``.
        """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a prompt to the LLM and return a full LLMResponse.

        The response includes generated text, token counts, and latency so
        that callers can record observability data without a separate call.

        Args:
            prompt: The full prompt string to send to the model.
            **kwargs: Optional generation parameters (temperature, max_tokens,
                etc.) that override defaults set at construction time.

        Returns:
            LLMResponse with text, usage, latency, and model fields populated.

        Raises:
            RuntimeError: If the API call fails after retries.
        """

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt and return generated text only.

        Convenience wrapper around :meth:`generate` for callers that do not
        need token or latency information.

        Args:
            prompt: The full prompt string to send to the model.
            **kwargs: Optional generation parameters forwarded to ``generate``.

        Returns:
            The model's text response.
        """
        return self.generate(prompt, **kwargs).text

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Request structured (JSON) output conforming to a given schema.

        Implementations that support JSON mode or function calling should
        override this method. The default implementation returns ``None``,
        indicating that structured output is not supported.

        Args:
            prompt: The full prompt string to send to the model.
            schema: JSON Schema dict describing the expected response structure.
            **kwargs: Optional generation parameters.

        Returns:
            A dict conforming to ``schema``, or ``None`` if the model did not
            return valid structured output or if the implementation does not
            support it.
        """
        return None

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate the token count for a text string.

        Used by the ContextPacker to enforce token budget limits without
        making an API call.

        Args:
            text: The text string to count tokens for.

        Returns:
            Estimated token count.
        """
