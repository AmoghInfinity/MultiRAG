from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    return embeddings