"""Provider-neutral streaming events.

Adapters translate their wire format (OpenAI SSE, Ollama NDJSON, Anthropic
SSE) into these events, and the server translates them back into whichever
API surface the client called. The same events are synthesized from a
completed result when a response has to be buffered for verification, so
both paths share one set of formatters.
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    """Model reasoning, streamed separately from the answer text. It is
    shown to clients that understand it and never fed to the verifier."""

    text: str


@dataclass
class ToolCallDelta:
    """A fragment of one tool call. `id` and `name` arrive on the first
    fragment for an index; `arguments` fragments concatenate to JSON."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class StreamEnd:
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None


StreamEvent = TextDelta | ThinkingDelta | ToolCallDelta | StreamEnd

_CHUNK_PATTERN = re.compile(r"\s*\S+|\s+")


def text_chunks(text: str) -> list[str]:
    """Split text into word-sized pieces that concatenate back exactly."""
    return _CHUNK_PATTERN.findall(text)


class StreamAccumulator:
    """Rebuilds a complete result from streamed events."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._tools: dict[int, dict[str, str]] = {}
        self.finish_reason = "stop"
        self.usage: dict[str, int] = {}
        self.model: str | None = None
        self.started = False

    def feed(self, event: StreamEvent) -> None:
        self.started = True
        if isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ThinkingDelta):
            self._thinking.append(event.text)
        elif isinstance(event, ToolCallDelta):
            tool = self._tools.setdefault(
                event.index, {"id": "", "name": "", "arguments": ""}
            )
            if event.id:
                tool["id"] = event.id
            if event.name:
                tool["name"] = event.name
            tool["arguments"] += event.arguments
        elif isinstance(event, StreamEnd):
            self.finish_reason = event.finish_reason or "stop"
            self.usage = dict(event.usage)
            self.model = event.model or self.model

    @property
    def content(self) -> str:
        return "".join(self._text)

    @property
    def thinking(self) -> str:
        return "".join(self._thinking)

    @property
    def tool_calls(self) -> list[dict[str, object]] | None:
        if not self._tools:
            return None
        return [
            {
                "id": tool["id"] or f"call_{index}",
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": tool["arguments"] or "{}",
                },
            }
            for index, tool in sorted(self._tools.items())
        ]

    @property
    def resolved_finish_reason(self) -> str:
        if self._tools and self.finish_reason == "stop":
            return "tool_calls"
        return self.finish_reason


def events_from_completion(
    content: str | None,
    tool_calls: list[dict[str, object]] | None,
    finish_reason: str,
    usage: dict[str, int],
    model: str | None,
) -> Iterator[StreamEvent]:
    """Replay a finished completion as stream events, byte for byte."""
    for piece in text_chunks(content or ""):
        yield TextDelta(piece)
    for index, tool_call in enumerate(tool_calls or []):
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        yield ToolCallDelta(
            index=index,
            id=str(tool_call.get("id") or f"call_{index}"),
            name=str(function.get("name", "")) if isinstance(function, dict) else "",
            arguments=arguments,
        )
    yield StreamEnd(finish_reason=finish_reason, usage=usage, model=model)


def iter_sse_data(lines: Iterator[str]) -> Iterator[str]:
    """Yield the `data:` payloads of a server-sent event stream."""
    for line in lines:
        if line.startswith("data:"):
            yield line[5:].strip()
