from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="route-auto",
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed-for-local-dev",
)


response = llm.invoke("Summarize RouteLabs Router in one sentence.")
print(response.content)
