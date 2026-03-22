"""Abstract base class for LLM client plugins."""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Interface that all LLM client plugins must implement.

    LLM clients send prompts to language models and return generated text.
    Implementations include OpenAI, Azure OpenAI, and local Ollama models.

    All clients must report the model identifier they are using so that
    AnswerTrace can record which model produced each response.
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the model used by this client.

        Returns:
            Model identifier string, e.g. "gpt-4o-mini".
        """

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt to the LLM and return the generated text.

        Args:
            prompt: The full prompt string to send to the model.
            **kwargs: Optional generation parameters (temperature, max_tokens,
                etc.) that override defaults set at construction time.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If the API call fails after retries.
        """

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
