from langchain_core.prompts import ChatPromptTemplate


def run_rag(query, retriever, llm):

    # -------------------------
    # Retrieve Documents
    # -------------------------

    retrieved_docs = retriever.invoke(query)

    # -------------------------
    # Build Context
    # -------------------------

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

    # -------------------------
    # Chain
    # -------------------------

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    # -------------------------
    # Return Everything
    # -------------------------

    return {
        "answer": response.content,
        "context": retrieved_docs
    }