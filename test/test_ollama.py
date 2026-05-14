from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)

response = llm.invoke("What is RAG?")

print(response.content)