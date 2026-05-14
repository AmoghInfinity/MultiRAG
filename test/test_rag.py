from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# -------------------------
# Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# -------------------------
# Load Vector DB
# -------------------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Select Model
# -------------------------

MODEL_MAP = {
    "llama": "llama3.1:8b",
    "qwen": "qwen3:8b",
    "mistral": "mistral:7b",
    "deepseek": "deepseek-llm:7b"
}

selected_model = "qwen"

llm = ChatOllama(
    model=MODEL_MAP[selected_model],
    temperature=0
)

# -------------------------
# Query
# -------------------------

query = "What is machine learning and its types?"

retrieved_docs = retriever.invoke(query)

context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)

# -------------------------
# Prompt
# -------------------------

prompt = ChatPromptTemplate.from_template(
    """
Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}
"""
)

chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": query
})

# -------------------------
# Output
# -------------------------

print("\n================ RAG RESPONSE ================\n")

print(response.content)