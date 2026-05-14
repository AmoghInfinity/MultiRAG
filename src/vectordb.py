import os

from langchain_chroma import Chroma


CHROMA_DB_DIR = "chroma_db"


def vectorstore_exists():

    return os.path.exists(CHROMA_DB_DIR)


def create_vectorstore(chunks, embeddings):

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    return vectorstore


def load_vectorstore(embeddings):

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    return vectorstore