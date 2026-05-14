# Multi-LLM RAG Evaluation Framework

## Overview

This project is a fully local Retrieval-Augmented Generation (RAG) framework built to compare the performance of multiple open-source Large Language Models (LLMs) on the same document-based question answering task.

The primary goal of this project was to explore:

- How different LLMs perform in a RAG pipeline
- Which model retrieves and answers more accurately
- How evaluation metrics differ across models
- How modern RAG evaluation frameworks like RAGAS can be used for benchmarking

Instead of testing only one model, this project benchmarks multiple models under the same retrieval setup and compares them using automated RAG evaluation metrics.

---

# Models Compared

The framework currently compares the following local models running through Ollama:

- Llama 3.1 8B
- Qwen3 8B
- Mistral 7B
- DeepSeek LLM 7B

Each model is evaluated on the same:

- PDF knowledge base
- retrieval pipeline
- chunking strategy
- embedding model
- evaluation dataset

This ensures a fair comparison between models.

---

# Features

- Local PDF-based RAG pipeline
- Multi-model benchmarking
- Chroma vector database integration
- Ollama-based local inference
- Automated RAG evaluation using RAGAS
- Retrieval context inspection
- Streamlit web interface
- CSV export of evaluation scores
- Modular project structure for scalability

---

# Tech Stack

## Core Frameworks

- Python
- LangChain
- RAGAS
- Streamlit

## LLM Runtime

- Ollama

## Models

- Llama 3.1 8B
- Qwen3 8B
- Mistral 7B
- DeepSeek LLM 7B

## Vector Database

- ChromaDB

## Embeddings

- nomic-embed-text

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
Retriever
      ↓
LLM Answer Generation
      ↓
RAGAS Evaluation
      ↓
Model Comparison
```

---

# Evaluation Metrics

The project uses RAGAS to evaluate model performance using:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

These metrics help measure:

- factual grounding
- retrieval quality
- answer correctness
- context utilization

---

# Folder Structure

```
MULTI_RAG/
│
├── chroma_db/
├── data/
├── results/
├── src/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── llm_factory.py
│   ├── loader.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectordb.py
│
├── test/
│   ├── test_chroma.py
│   ├── test_embeddings.py
│   ├── test_loader.py
│   ├── test_ollama.py
│   ├── test_rag.py
│   ├── test_retriever.py
│   └── test_splitter.py
│
├── evaluate.py
├── evaluation_dataset.py
├── main.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/AmoghInfinity/MultiRAG.git
cd MultiRAG
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Install Ollama Models

```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b
ollama pull mistral:7b
ollama pull deepseek-llm:7b
ollama pull nomic-embed-text
```

---

## 6. Start Ollama

```bash
ollama serve
```

---

# Running the Project

## Run Main RAG Pipeline

```bash
python main.py
```

---

## Run Evaluation Pipeline

```bash
python evaluate.py
```

---

# Key Learnings

Through this project, I explored:

- practical RAG pipeline development
- local LLM deployment using Ollama
- vector databases and embeddings
- automated evaluation of RAG systems
- benchmarking multiple LLMs
- modular GenAI system design
- tradeoffs between different open-source models

---

# Future Improvements

Potential future enhancements include:

- Hybrid search
- Reranking pipelines
- Multi-query retrieval
- Context compression
- Multimodal RAG
- Better evaluation datasets
- Advanced observability dashboards
- GPU optimized inference
- Quantized model benchmarking

---

# Author

Amogh Gupta

Built as a learning-focused GenAI project to explore and compare different open-source LLMs in Retrieval-Augmented Generation workflows.
