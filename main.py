import argparse

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import load_embeddings

from src.vectordb import (
    create_vectorstore,
    load_vectorstore,
    vectorstore_exists
)

from src.retriever import create_retriever
from src.llm_factory import load_llm
from src.rag_pipeline import run_rag


def main():

    parser = argparse.ArgumentParser(
        description="Multi-LLM Local RAG System"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: llama, qwen, mistral, deepseek"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to ask the RAG system"
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default="data/sample.pdf",
        help="Path to PDF document"
    )

    args = parser.parse_args()

    embeddings = load_embeddings()

    if vectorstore_exists():

        print("\nLoading existing Chroma DB...\n")

        vectorstore = load_vectorstore(embeddings)

    else:

        print("\nCreating new Chroma DB...\n")

        documents = load_pdf(args.pdf)

        chunks = split_documents(documents)

        vectorstore = create_vectorstore(
            chunks,
            embeddings
        )

    retriever = create_retriever(vectorstore)

    llm = load_llm(args.model)

    result = run_rag(
        query=args.query,
        retriever=retriever,
        llm=llm
    )

    print("\n================ QUERY ================\n")

    print(args.query)

    print("\n================ RETRIEVED CHUNKS ================\n")

    for i, doc in enumerate(result["context"], start=1):

        print(f"\n--- Chunk {i} ---\n")

        print(doc.page_content[:500])

    print("\n================ ANSWER ================\n")

    print(result["answer"])


if __name__ == "__main__":
    main()