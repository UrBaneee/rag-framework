"""Tests for BaseLLMClient interface and LLMResponse contract — Task 7.1."""

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, Optional

import pytest

from rag.core.interfaces.llm_client import BaseLLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Concrete stub for testing the ABC
# ---------------------------------------------------------------------------


class StubLLMClient(BaseLLMClient):
    """Minimal concrete implementation for interface tests."""

    def __init__(self, model_name: str = "stub-model") -> None:
        self._model = model_name
        self.last_prompt: Optional[str] = None

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(
            text="stub answer",
            model=self._model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=42.0,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


def test_llm_response_fields():
    r = LLMResponse(
        text="hello",
        model="gpt-4o",
        prompt_tokens=20,
        completion_tokens=10,
        latency_ms=123.0,
    )
    assert r.text == "hello"
    assert r.model == "gpt-4o"
    assert r.prompt_tokens == 20
    assert r.completion_tokens == 10
    assert r.latency_ms == pytest.approx(123.0)


def test_llm_response_total_tokens_auto():
    """total_tokens should be auto-computed when not explicitly set."""
    r = LLMResponse(text="hi", model="m", prompt_tokens=15, completion_tokens=7)
    assert r.total_tokens == 22


def test_llm_response_total_tokens_explicit():
    """Explicit total_tokens overrides auto-calculation."""
    r = LLMResponse(text="hi", model="m", prompt_tokens=5, completion_tokens=5, total_tokens=99)
    assert r.total_tokens == 99


def test_llm_response_structured_defaults_none():
    r = LLMResponse(text="x", model="m")
    assert r.structured is None


def test_llm_response_structured_field():
    r = LLMResponse(text="{}", model="m", structured={"key": "value"})
    assert r.structured == {"key": "value"}


# ---------------------------------------------------------------------------
# BaseLLMClient interface
# ---------------------------------------------------------------------------


def test_stub_implements_interface():
    client = StubLLMClient()
    assert isinstance(client, BaseLLMClient)


def test_generate_returns_llm_response():
    client = StubLLMClient()
    response = client.generate("What is RAG?")
    assert isinstance(response, LLMResponse)
    assert response.text == "stub answer"
    assert response.latency_ms > 0


def test_complete_returns_text():
    """complete() must delegate to generate() and return text string."""
    client = StubLLMClient()
    text = client.complete("Hello world")
    assert isinstance(text, str)
    assert text == "stub answer"


def test_count_tokens_returns_int():
    client = StubLLMClient()
    count = client.count_tokens("one two three")
    assert isinstance(count, int)
    assert count > 0


def test_generate_structured_default_returns_none():
    """Default generate_structured must return None (not raise)."""
    client = StubLLMClient()
    result = client.generate_structured("prompt", {"type": "object"})
    assert result is None


def test_model_property():
    client = StubLLMClient(model_name="my-model")
    assert client.model == "my-model"


# ---------------------------------------------------------------------------
# No SDK imports in pipeline modules
# ---------------------------------------------------------------------------


def _get_imports(filepath: Path) -> set[str]:
    """Extract top-level module names imported in a Python file."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


_CONCRETE_SDKS = {"openai", "voyageai", "anthropic", "cohere", "boto3"}
_PIPELINE_DIR = Path(__file__).parent.parent / "rag" / "pipelines"


@pytest.mark.parametrize(
    "pipeline_file",
    list(_PIPELINE_DIR.glob("*.py")),
    ids=lambda p: p.name,
)
def test_no_concrete_sdk_imports_in_pipeline(pipeline_file: Path):
    """Pipeline modules must not import concrete LLM/embedding SDKs directly."""
    imports = _get_imports(pipeline_file)
    bad = imports & _CONCRETE_SDKS
    assert not bad, (
        f"{pipeline_file.name} directly imports concrete SDK(s): {bad}. "
        "Use abstract interfaces instead."
    )
