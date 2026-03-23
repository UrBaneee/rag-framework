"""Tests for OpenAILLMClient — Task 7.2 (all mocked, no real API calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag.core.interfaces.llm_client import BaseLLMClient, LLMResponse
from rag.infra.llm.openai_llm_client import OpenAILLMClient


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------


def _mock_usage(prompt=10, completion=20):
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    return usage


def _mock_response(text: str, prompt_tokens=10, completion_tokens=20) -> MagicMock:
    """Build a fake OpenAI ChatCompletion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = _mock_usage(prompt_tokens, completion_tokens)
    return resp


def _make_client(mock_response_text: str = "stub answer", **kwargs) -> tuple[OpenAILLMClient, MagicMock]:
    """Create an OpenAILLMClient with a mocked underlying OpenAI client."""
    with patch("openai.OpenAI") as MockOpenAI:
        mock_openai = MockOpenAI.return_value
        mock_openai.chat.completions.create.return_value = _mock_response(mock_response_text)
        client = OpenAILLMClient(api_key="test-key", **kwargs)
        client._client = mock_openai
    return client, mock_openai


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


def test_implements_base_interface():
    client, _ = _make_client()
    assert isinstance(client, BaseLLMClient)


def test_model_property():
    client, _ = _make_client(model="gpt-4o")
    assert client.model == "gpt-4o"


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_llm_response():
    client, mock = _make_client("This is a grounded answer.")
    mock.chat.completions.create.return_value = _mock_response("This is a grounded answer.")
    result = client.generate("What is RAG?")
    assert isinstance(result, LLMResponse)
    assert result.text == "This is a grounded answer."


def test_generate_populates_token_counts():
    client, mock = _make_client()
    mock.chat.completions.create.return_value = _mock_response("answer", prompt_tokens=15, completion_tokens=8)
    result = client.generate("prompt")
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 8
    assert result.total_tokens == 23


def test_generate_populates_latency():
    client, mock = _make_client()
    mock.chat.completions.create.return_value = _mock_response("answer")
    result = client.generate("prompt")
    assert result.latency_ms >= 0.0


def test_generate_populates_model():
    client, mock = _make_client(model="gpt-4o-mini")
    mock.chat.completions.create.return_value = _mock_response("answer")
    result = client.generate("prompt")
    assert result.model == "gpt-4o-mini"


def test_generate_sends_user_message():
    client, mock = _make_client()
    mock.chat.completions.create.return_value = _mock_response("ok")
    client.generate("Hello world")
    call_kwargs = mock.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Hello world"


def test_generate_with_system_prompt():
    client, mock = _make_client(system_prompt="You are a helpful assistant.")
    mock.chat.completions.create.return_value = _mock_response("ok")
    client.generate("What is RAG?")
    call_kwargs = mock.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    sys_msgs = [m for m in messages if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert "helpful assistant" in sys_msgs[0]["content"]


# ---------------------------------------------------------------------------
# complete() convenience wrapper
# ---------------------------------------------------------------------------


def test_complete_returns_string():
    client, mock = _make_client("short answer")
    mock.chat.completions.create.return_value = _mock_response("short answer")
    text = client.complete("Prompt here")
    assert isinstance(text, str)
    assert text == "short answer"


# ---------------------------------------------------------------------------
# generate_structured()
# ---------------------------------------------------------------------------


def test_generate_structured_parses_json():
    payload = {"answer": "RAG combines search and generation.", "confidence": 0.9}
    client, mock = _make_client(json.dumps(payload))
    mock.chat.completions.create.return_value = _mock_response(json.dumps(payload))
    result = client.generate_structured("Give JSON answer", schema={})
    assert result == payload


def test_generate_structured_returns_none_on_bad_json():
    client, mock = _make_client("not valid json {{")
    mock.chat.completions.create.return_value = _mock_response("not valid json {{")
    result = client.generate_structured("prompt", schema={})
    assert result is None


def test_generate_structured_sets_json_response_format():
    client, mock = _make_client("{}")
    mock.chat.completions.create.return_value = _mock_response("{}")
    client.generate_structured("prompt", schema={})
    call_kwargs = mock.chat.completions.create.call_args[1]
    assert call_kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------


def test_count_tokens_returns_int():
    client, _ = _make_client()
    count = client.count_tokens("hello world this is a test")
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_heuristic_fallback(monkeypatch):
    """When tiktoken is unavailable, falls back to len//4 heuristic."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("tiktoken not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    client, _ = _make_client()
    text = "a" * 100
    count = client.count_tokens(text)
    assert count == 25  # 100 // 4
