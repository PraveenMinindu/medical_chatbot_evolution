# Medical AI Chatbot — Final Integrated RAG System (V2 with Ollama)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Tagline / Short Description
A Retrieval-Augmented Generation (RAG) based medical chatbot that provides safe, context-aware responses using FAISS vector search and a local Ollama LLM.

---

## Problem Statement
Traditional chatbots often generate incorrect or hallucinated responses, especially in sensitive domains like healthcare. Users require reliable, context-based information without unsafe or misleading advice.

---

## Solution Overview
This project implements a RAG-based chatbot that retrieves relevant medical information from a curated knowledge base and generates responses using a local LLM (Ollama). A safety layer ensures that outputs remain educational and non-harmful.

---

## System Architecture

Offline Pipeline:
- Raw medical documents
- Chunking and preprocessing
- Embedding generation
- FAISS index creation

Online Pipeline:
- User query input
- Input safety validation
- Query embedding
- FAISS similarity search (Top-K retrieval)
- Prompt construction
- LLM response generation (Ollama)
- Output safety filtering
- Final response

---

## Tech Stack

- Language: Python
- Backend: FastAPI
- Vector Search: FAISS
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- LLM: Ollama (local inference)
- Data Handling: JSON

---

## Features

- Retrieval-Augmented Generation (RAG)
- FAISS-based semantic search
- Local LLM inference using Ollama
- Configurable LLM provider (mock, OpenAI, Ollama)
- Input and output safety guardrails
- Fallback handling for low-confidence queries
- API-based interaction via FastAPI
- CLI-based chatbot interface

---

## Sample Output

Example Query:
"Why do I feel hot and dizzy?"

Example Response:
"Feeling hot and dizzy can be related to conditions such as fever or dehydration. Fever occurs when the body temperature rises due to infection, while dehydration results from insufficient fluid levels..."

---

## Project Structure
05_final_integrated_rag_v2_ollama/
├── app/
├── data/
├── vector_store/
├── scripts/
├── run.py
├── requirements.txt
└── README.md

---

## Installation & Setup

```bash
git clone https://github.com/PraveenMinindu/medical_chatbot_evolution.git
cd 05_final_integrated_rag_v2_ollama
pip install -r requirements.txt
ollama run llama3
```
## Usage / How to Run
python scripts/ingest_data.py
python -m uvicorn app.main:app --reload
## Open
http://127.0.0.1:8000/docs

## How It Works (Workflow)

User → Embedding → FAISS → Context → LLM → Safety → Response
## Model Evaluation / Metrics

Similarity score analysis
Retrieved context validation
Manual answer verification

## Future:

Precision@K
Recall@K
Groundedness
Safety evaluation

## Limitations

Small dataset
Limited domain coverage
Retrieval-dependent accuracy
Not a clinical system

## Future Improvements

Larger dataset
Hybrid search (BM25 + embeddings)
UI interface
Deployment
Conversation memory

