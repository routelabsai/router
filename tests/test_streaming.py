import json

import httpx
import pytest

from routelabs_router.adapters.anthropic import AnthropicChatAdapter
from routelabs_router.adapters.base import ProviderExecutionError
from routelabs_router.adapters.ollama import OllamaChatAdapter
from routelabs_router.adapters.openai_compatible import OpenAICompatibleChatAdapter
from routelabs_router.config import DEFAULT_CONFIG
from routelabs_router.models import ChatCompletionRequest, ChatMessage
from routelabs_router.streaming import StreamAccumulator, StreamEnd, TextDelta, text_chunks

REQUEST = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
REAL_CLIENT = httpx.Client


def _patch_transport(monkeypatch, body: str, status: int = 200) -> list[dict]:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(status, text=body)

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: REAL_CLIENT(transport=httpx.MockTransport(handler)),
    )
    return sent


def _collect(events) -> StreamAccumulator:
    accumulator = StreamAccumulator()
    for event in events:
        accumulator.feed(event)
    return accumulator


def _sse(*payloads: dict, done: bool = False) -> str:
    body = "".join(f"data: {json.dumps(payload)}\n\n" for payload in payloads)
    return body + ("data: [DONE]\n\n" if done else "")


def test_text_chunks_round_trip_exactly() -> None:
    text = "def f():\n    return 1\n\n\n  trailing  "
    assert "".join(text_chunks(text)) == text


def test_openai_compatible_stream_parses_text_tools_and_usage(monkeypatch) -> None:
    body = _sse(
        {"model": "gpt-5-mini", "choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo\n  world"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "function": {"name": "ls", "arguments": "{\"pa"}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": 1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
        done=True,
    )
    sent = _patch_transport(monkeypatch, body)
    adapter = OpenAICompatibleChatAdapter(
        DEFAULT_CONFIG.providers.cloud.openai_compatible.model_copy(update={"api_key": "k"})
    )
    result = _collect(adapter.stream(REQUEST, model="gpt-5-mini"))

    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert result.content == "Hello\n  world"
    assert result.tool_calls[0]["function"] == {"name": "ls", "arguments": "{\"path\": 1}"}
    assert result.resolved_finish_reason == "tool_calls"
    assert result.usage["total_tokens"] == 7
    assert result.model == "gpt-5-mini"


def test_ollama_stream_parses_ndjson(monkeypatch) -> None:
    lines = [
        {"model": "qwen3:4b", "message": {"role": "assistant", "content": "a\n"}, "done": False},
        {"message": {"content": "  b"}, "done": False},
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "ls", "arguments": {"path": "."}}}],
            },
            "done": False,
        },
        {"done": True, "done_reason": "stop", "prompt_eval_count": 2, "eval_count": 5},
    ]
    sent = _patch_transport(monkeypatch, "\n".join(json.dumps(line) for line in lines))
    adapter = OllamaChatAdapter(DEFAULT_CONFIG.providers.local.ollama)
    result = _collect(
        adapter.stream(
            REQUEST.model_copy(update={"max_completion_tokens": 64}),
        )
    )

    assert sent[0]["stream"] is True
    assert sent[0]["options"]["num_predict"] == 64
    assert result.content == "a\n  b"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"path": "."}
    assert result.resolved_finish_reason == "tool_calls"
    assert result.usage == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}


def test_anthropic_stream_parses_text_and_tool_use(monkeypatch) -> None:
    body = "".join(
        f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"
        for payload in [
            {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 9}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hi\n"}},
            {"type": "ping"},
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}},
            },
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"a\""}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ": 1}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"},
        ]
    )
    sent = _patch_transport(monkeypatch, body)
    adapter = AnthropicChatAdapter(
        DEFAULT_CONFIG.providers.cloud.anthropic.model_copy(update={"api_key": "k"})
    )
    result = _collect(adapter.stream(REQUEST, model="claude-sonnet-5-5"))

    assert sent[0]["stream"] is True
    assert result.content == "Hi\n"
    assert result.tool_calls == [
        {"id": "toolu_1", "type": "function", "function": {"name": "ls", "arguments": "{\"a\": 1}"}}
    ]
    assert result.resolved_finish_reason == "tool_calls"
    assert result.usage == {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}


@pytest.mark.parametrize(
    "adapter",
    [
        OpenAICompatibleChatAdapter(DEFAULT_CONFIG.providers.local.llamacpp),
        OllamaChatAdapter(DEFAULT_CONFIG.providers.local.ollama),
        AnthropicChatAdapter(DEFAULT_CONFIG.providers.cloud.anthropic),
    ],
)
def test_stream_http_errors_raise_provider_errors(monkeypatch, adapter) -> None:
    _patch_transport(monkeypatch, '{"error": "model not found"}', status=404)
    with pytest.raises(ProviderExecutionError, match="404"):
        list(adapter.stream(REQUEST))


def test_accumulator_defaults_empty_tool_arguments_to_object() -> None:
    accumulator = StreamAccumulator()
    from routelabs_router.streaming import ToolCallDelta

    accumulator.feed(TextDelta("x"))
    accumulator.feed(ToolCallDelta(index=0, id="t", name="noop"))
    accumulator.feed(StreamEnd())
    assert accumulator.tool_calls[0]["function"]["arguments"] == "{}"


def test_reasoning_fields_become_thinking_deltas(monkeypatch) -> None:
    from routelabs_router.streaming import ThinkingDelta

    lines = [
        {"message": {"content": "", "thinking": "Let me think"}, "done": False},
        {"message": {"content": "Answer"}, "done": False},
        {"done": True, "done_reason": "stop"},
    ]
    _patch_transport(monkeypatch, "\n".join(json.dumps(line) for line in lines))
    events = list(OllamaChatAdapter(DEFAULT_CONFIG.providers.local.ollama).stream(REQUEST))
    assert ThinkingDelta("Let me think") in events
    accumulator = _collect(events)
    assert accumulator.content == "Answer"  # reasoning never reaches the verifier
    assert accumulator.thinking == "Let me think"

    body = _sse(
        {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
        done=True,
    )
    _patch_transport(monkeypatch, body)
    events = list(OpenAICompatibleChatAdapter(DEFAULT_CONFIG.providers.local.llamacpp).stream(REQUEST))
    assert events[0] == ThinkingDelta("hmm")
