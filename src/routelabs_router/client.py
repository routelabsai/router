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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "private": private,
            "stream": stream,
        }
        if model is not None:
            payload["model"] = model
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
        return self._request("POST", "/v1/chat/completions", json=payload)

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
