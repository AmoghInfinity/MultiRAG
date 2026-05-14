from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

text = "Retrieval Augmented Generation improves LLM responses."

vector = embeddings.embed_query(text)

print(f"\nEmbedding vector length: {len(vector)}")

print("\nFirst 10 values:\n")
print(vector[:10])