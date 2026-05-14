import pandas as pd

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama

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

from evaluation_dataset import evaluation_data


# ==========================================
# Local Qwen Evaluator LLM
# ==========================================

evaluator_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


# ==========================================
# RAGAS Run Config
# ==========================================

run_config = RunConfig(
    timeout=180,
    max_workers=1,
    max_wait=10
)


# ==========================================
# Load Embeddings
# ==========================================

print("\nLoading embeddings...\n")

embeddings = load_embeddings()

ragas_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)


# ==========================================
# Load/Create Vector DB
# ==========================================

if vectorstore_exists():

    print("Loading existing Chroma DB...\n")

    vectorstore = load_vectorstore(embeddings)

else:

    print("Creating new Chroma DB...\n")

    documents = load_pdf("data/sample.pdf")

    chunks = split_documents(documents)

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )


# ==========================================
# Create Retriever
# ==========================================

retriever = create_retriever(vectorstore)


# ==========================================
# Models To Compare
# ==========================================

models = {
    "llama": "llama3.1:8b",
    "qwen": "qwen3:8b",
    "mistral": "mistral:7b",
    "deepseek": "deepseek-llm:7b"
}


all_results = []


# ==========================================
# Evaluate Each Model
# ==========================================

for model_key, model_name in models.items():

    print(f"\n==============================")
    print(f"Evaluating Model: {model_name}")
    print(f"==============================\n")

    llm = load_llm(model_key)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for sample in evaluation_data:

        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"\nQuestion: {question}")

        result = run_rag(
            query=question,
            retriever=retriever,
            llm=llm
        )

        answer = result["answer"]

        retrieved_contexts = [
            doc.page_content
            for doc in result["context"]
        ]

        print(f"\nAnswer:\n{answer}\n")

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    # ==========================================
    # Create Dataset
    # ==========================================

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print(f"\nRunning RAGAS Evaluation for {model_name}...\n")

    # ==========================================
    # Run RAGAS Evaluation
    # ==========================================

    scores = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=evaluator_llm,
        embeddings=ragas_embeddings,
        run_config=run_config
    )

    scores_df = scores.to_pandas()

    avg_scores = scores_df.mean(numeric_only=True)

    model_result = {
        "model": model_name,
        "faithfulness": round(float(avg_scores["faithfulness"]), 4),
        "answer_relevancy": round(float(avg_scores["answer_relevancy"]), 4),
        "context_precision": round(float(avg_scores["context_precision"]), 4),
        "context_recall": round(float(avg_scores["context_recall"]), 4)
    }

    all_results.append(model_result)

    print("\nAverage Scores:\n")

    print(model_result)


# ==========================================
# Final Comparison
# ==========================================

final_df = pd.DataFrame(all_results)

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================\n")

print(final_df)

final_df.to_csv("ragas_results.csv", index=False)

print("\nResults saved to ragas_results.csv\n")