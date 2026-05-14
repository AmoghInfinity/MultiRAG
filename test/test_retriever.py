from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# Load existing vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

query = "What is machine learning and its types?"

results = retriever.invoke(query)

print(f"\nRetrieved {len(results)} chunks\n")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Chunk {i} ---\n")
    print(doc.page_content[:1000])