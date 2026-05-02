from routelabs_router import RouteLabsClient


client = RouteLabsClient("http://127.0.0.1:8000")

print("health")
print(client.health())

print("\nroute")
print(client.route("Summarize a short product description", private=False))

print("\nchat")
print(
    client.chat(
        [
            {
                "role": "user",
                "content": (
                    "Summarize this in one sentence: RouteLabs Router chooses "
                    "between local and cloud models based on privacy, cost, latency, "
                    "and task complexity."
                ),
            }
        ]
    )
)

print("\nstats")
print(client.stats())

print("\nlogs")
print(client.logs())
