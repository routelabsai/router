from typing import Any

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    task: str = Field(..., min_length=1)
    private: bool = False


class RouteDecision(BaseModel):
    target: str
    provider: str
    model: str
    reason: str
    complexity: str
    verify: bool


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None
    private: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    route: RouteDecision


class ProviderResult(BaseModel):
    content: str
    model: str
    finish_reason: str = "stop"
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    raw: dict[str, Any] = Field(default_factory=dict)
