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
    provider_available: bool | None = None
    provider_status: str | None = None
    fallback_available: bool | None = None
    fallback_status: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None
    private: bool = False
    stream: bool = False
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class ResponsesTextConfig(BaseModel):
    format: dict[str, Any] | None = None


class ResponsesRequest(BaseModel):
    input: str | list[Any]
    model: str | None = None
    instructions: str | None = None
    private: bool = False
    stream: bool = False
    text: ResponsesTextConfig | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    private: bool = False


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    route: RouteDecision
    trace: "DecisionTrace"


class ResponsesOutputText(BaseModel):
    type: str = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class ResponsesOutputMessage(BaseModel):
    id: str
    type: str = "message"
    status: str = "completed"
    role: str = "assistant"
    content: list[ResponsesOutputText] = Field(default_factory=list)


class ResponsesFunctionCall(BaseModel):
    id: str
    type: str = "function_call"
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


class ResponsesUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    model: str
    output: list[ResponsesOutputMessage | ResponsesFunctionCall]
    output_text: str = ""
    usage: ResponsesUsage
    route: RouteDecision
    trace: "DecisionTrace"


class AnthropicMessageRequest(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class AnthropicToolChoice(BaseModel):
    type: str
    name: str | None = None


class AnthropicMessagesRequest(BaseModel):
    messages: list[AnthropicMessageRequest] = Field(..., min_length=1)
    model: str | None = None
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    private: bool = False
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: AnthropicToolChoice | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[dict[str, Any]]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
    route: RouteDecision
    trace: "DecisionTrace"


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingsUsage
    route: RouteDecision


class ProviderEmbeddingResult(BaseModel):
    data: list[EmbeddingObject]
    model: str
    usage: EmbeddingsUsage


class ProviderResult(BaseModel):
    content: str | None = None
    model: str
    finish_reason: str = "stop"
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    tool_calls: list[dict[str, Any]] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    passed: bool
    confidence: float
    grounded: bool
    hallucination_signals: list[str] = Field(default_factory=list)
    reason: str
    should_escalate: bool


class ProviderAttempt(BaseModel):
    provider: str
    model: str
    target: str
    outcome: str
    reason: str | None = None
    duration_ms: float | None = None


class DecisionTrace(BaseModel):
    privacy: "PrivacyDetectionResult | None" = None
    initial_route: RouteDecision
    attempts: list[ProviderAttempt] = Field(default_factory=list)
    verification: VerificationResult | None = None
    escalated: bool = False
    escalation_reason: str | None = None
    final_route: RouteDecision
    total_latency_ms: float | None = None
    completion_tokens_per_second: float | None = None


class RouterStats(BaseModel):
    total_requests: int = 0
    chat_requests: int = 0
    embedding_requests: int = 0
    local_responses: int = 0
    cloud_responses: int = 0
    escalations: int = 0
    verification_checks: int = 0
    verification_failures: int = 0
    private_requests: int = 0
    auto_private_requests: int = 0
    estimated_total_cost_usd: float = 0.0
    estimated_baseline_cloud_cost_usd: float = 0.0
    estimated_cost_saved_usd: float = 0.0
    estimated_cloud_requests_avoided: int = 0

    @property
    def local_response_rate(self) -> float:
        return _safe_ratio(self.local_responses, self.total_requests)

    @property
    def cloud_response_rate(self) -> float:
        return _safe_ratio(self.cloud_responses, self.total_requests)

    @property
    def escalation_rate(self) -> float:
        return _safe_ratio(self.escalations, self.total_requests)

    @property
    def verification_failure_rate(self) -> float:
        return _safe_ratio(self.verification_failures, self.verification_checks)


class RouterStatsResponse(BaseModel):
    total_requests: int
    chat_requests: int
    embedding_requests: int
    local_responses: int
    cloud_responses: int
    escalations: int
    verification_checks: int
    verification_failures: int
    private_requests: int
    auto_private_requests: int
    estimated_total_cost_usd: float
    estimated_baseline_cloud_cost_usd: float
    estimated_cost_saved_usd: float
    estimated_cloud_requests_avoided: int
    local_response_rate: float
    cloud_response_rate: float
    escalation_rate: float
    verification_failure_rate: float
    avg_total_latency_ms: float
    avg_chat_latency_ms: float
    avg_embedding_latency_ms: float
    avg_local_latency_ms: float
    avg_cloud_latency_ms: float
    avg_completion_tokens_per_second: float


class PrivacyDetectionResult(BaseModel):
    detected: bool
    categories: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    forced_local: bool


class RouteLogEntry(BaseModel):
    request_id: str
    request_kind: str
    task_preview: str
    private: bool
    auto_private: bool
    estimated_request_cost_usd: float
    estimated_baseline_cloud_cost_usd: float
    estimated_cost_saved_usd: float
    total_latency_ms: float | None = None
    completion_tokens_per_second: float | None = None
    trace: DecisionTrace


class RouteLogResponse(BaseModel):
    entries: list[RouteLogEntry]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str
    provider: str | None = None
    source: str | None = None
    installed: bool | None = None
    status: str | None = None
    size_bytes: int | None = None


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ProviderHealth(BaseModel):
    available: bool
    status: str


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, ProviderHealth]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 3)
