from langchain_ollama import ChatOllama


MODEL_MAP = {
    "llama": "llama3.1:8b",
    "qwen": "qwen3:8b",
    "mistral": "mistral:7b",
    "deepseek": "deepseek-llm:7b"
}


def load_llm(model_name: str):

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model: {model_name}")

    llm = ChatOllama(
        model=MODEL_MAP[model_name],
        temperature=0
    )

    return llm