from __future__ import annotations

from typing import Any

import httpx


class RouteLabsClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def route(self, task: str, private: bool = False) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/route",
            json={"task": task, "private": private},
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        private: bool = False,
        model: str | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "private": private,
            "stream": stream,
        }
        if model is not None:
            payload["model"] = model
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        return self._request("POST", "/v1/chat/completions", json=payload)

    def responses(
        self,
        input: str | list[dict[str, Any]],
        model: str | None = None,
        instructions: str | None = None,
        private: bool = False,
        stream: bool = False,
        text: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": input,
            "private": private,
            "stream": stream,
        }
        if model is not None:
            payload["model"] = model
        if instructions is not None:
            payload["instructions"] = instructions
        if text is not None:
            payload["text"] = text
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        return self._request("POST", "/v1/responses", json=payload)

    def messages(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        private: bool = False,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "private": private,
            "stream": stream,
        }
        if model is not None:
            payload["model"] = model
        if system is not None:
            payload["system"] = system
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop_sequences is not None:
            payload["stop_sequences"] = stop_sequences
        return self._request("POST", "/v1/messages", json=payload)

    def embeddings(
        self,
        input: str | list[str],
        private: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": input,
            "private": private,
        }
        if model is not None:
            payload["model"] = model
        return self._request("POST", "/v1/embeddings", json=payload)

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    def logs(self) -> dict[str, Any]:
        return self._request("GET", "/v1/logs")

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")

    def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
            )
            response.raise_for_status()
            return response.json()
