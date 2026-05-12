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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "private": private,
        }
        if model is not None:
            payload["model"] = model
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
