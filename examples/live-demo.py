"""Live terminal demo of RouteLabs Router.

Start the router first (`router start`), then run:

    python examples/live-demo.py

It shows three things against whatever models you have configured:
1. routing decisions for a sensitive and a harmless prompt
2. a streamed answer, with model reasoning shown live as it happens
3. the router's running stats (local vs cloud, estimated savings)

Only `httpx` is required; it is already a RouteLabs dependency.
"""

import json
import os
import sys
import time

import httpx

BASE_URL = os.environ.get("ROUTELABS_URL", "http://127.0.0.1:8000")
HEADERS = {}
if os.environ.get("ROUTELABS_API_KEY"):
    HEADERS["authorization"] = f"Bearer {os.environ['ROUTELABS_API_KEY']}"

BOLD, DIM, GREEN, CYAN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[33m", "\033[0m"
)


def say(text: str = "", delay: float = 0.0) -> None:
    print(text, flush=True)
    if delay:
        time.sleep(delay)


def show_route(client: httpx.Client, task: str) -> None:
    decision = client.post("/v1/route", json={"task": task}).json()
    say(f"  {DIM}prompt{RESET}  {task}")
    say(
        f"  {GREEN}→ {decision['target']}{RESET} · {decision['provider']}/{decision['model']}"
        f"  {DIM}({decision['reason']}){RESET}",
        delay=1.2,
    )


def stream_answer(client: httpx.Client, prompt: str) -> None:
    body = {"stream": True, "messages": [{"role": "user", "content": prompt}]}
    started = time.time()
    first_token = None
    thinking_chars = 0
    answering = False
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        h = response.headers
        say(
            f"  {DIM}x-routelabs-target:{RESET} {h.get('x-routelabs-target')}   "
            f"{DIM}provider:{RESET} {h.get('x-routelabs-provider')}/{h.get('x-routelabs-model')}   "
            f"{DIM}streaming:{RESET} {h.get('x-routelabs-streaming')}"
        )
        for line in response.iter_lines():
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                say(f"\n  error: {chunk['error'].get('message')}")
                return
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0]["delta"]
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            if (reasoning or content) and first_token is None:
                first_token = time.time() - started
                say(f"  {YELLOW}first token after {first_token:.2f}s{RESET}")
            if reasoning:
                thinking_chars += len(reasoning)
                sys.stdout.write(
                    f"\r  {DIM}thinking… {thinking_chars:,} chars of reasoning streamed{RESET}"
                )
                sys.stdout.flush()
            if content:
                if not answering:
                    answering = True
                    sys.stdout.write("\n\n" if thinking_chars else "\n")
                sys.stdout.write(content)
                sys.stdout.flush()
    say(f"\n\n  {DIM}done in {time.time() - started:.1f}s{RESET}\n", delay=1.0)


def main() -> None:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=300) as client:
        health = client.get("/healthz").json()
        say(f"{BOLD}RouteLabs Router{RESET} {DIM}· {BASE_URL} · status {health['status']}{RESET}\n", 0.8)

        say(f"{CYAN}1. Where does each request go?{RESET}")
        show_route(client, "Summarize this contract for jane.doe@acme.com, card 4111 1111 1111 1111")
        show_route(client, "Explain the CAP theorem in one sentence")
        say()

        say(f"{CYAN}2. Streaming, with model reasoning shown live{RESET}")
        stream_answer(client, "Write a Python function that checks whether a string is a palindrome. Code only.")

        stats = client.get("/v1/stats").json()
        say(f"{CYAN}3. Router stats{RESET}")
        say(
            f"  local responses: {stats['local_responses']}   cloud responses: {stats['cloud_responses']}   "
            f"estimated cloud cost avoided: ${stats['estimated_cost_saved_usd']:.4f}",
            delay=2.5,
        )


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        sys.exit(f"RouteLabs Router is not reachable at {BASE_URL}. Start it with `router start`.")
