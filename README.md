# Medical AI Chatbot Evolution — From Rule-Based Systems to RAG with Ollama (V2)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![FAISS](https://img.shields.io/badge/FAISS-VectorSearch-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## Overview

This project demonstrates the **complete evolution of chatbot architectures**, progressing from basic rule-based logic to a **fully integrated Retrieval-Augmented Generation (RAG) system** powered by a local Large Language Model (Ollama).

The final system combines:
- Semantic retrieval (FAISS + embeddings)
- Context-aware generation (LLM)
- Safety guardrails
- API-based deployment

The focus is on understanding **how modern AI systems are designed, not just used**.

---

## Problem Statement

Conventional chatbot systems suffer from key limitations:

- Rule-based systems lack flexibility and scalability  
- Keyword/NLP systems fail to generalize across contexts  
- ML-based systems require extensive labeled data  
- Pure LLM systems may produce hallucinated or unsafe responses  

In sensitive domains such as healthcare, these limitations can lead to **incorrect or unsafe outputs**.

---

## Solution Approach

This project adopts a **progressive system design approach**, where each version improves upon the previous one:

| Version | Approach | Capability |
|--------|--------|-----------|
| V1 | Rule-Based | Static responses |
| V2 | Keyword/NLP | Pattern matching |
| V3 | ML-Based | Intent classification |
| V4 | Retrieval | TF-IDF + similarity search |
| V5 | RAG | FAISS + embeddings + grounded generation |
| V5 V2 | RAG + Ollama | Fully functional AI system with local LLM |

The final system integrates **retrieval + generation + safety** into a single pipeline.

---

## System Architecture

### Evolution Pipeline

Rule-Based → NLP → ML → Retrieval → RAG → RAG + LLM

---

### Final System (V2 with Ollama)

#### Offline Processing
- Data collection
- Text chunking
- Embedding generation
- FAISS index creation

#### Online Inference Flow

User Query  
→ Input Safety Validation  
→ Embedding Generation  
→ FAISS Vector Search (Top-K Retrieval)  
→ Context Extraction  
→ Prompt Construction  
→ LLM Generation (Ollama)  
→ Output Safety Filtering  
→ Final Response  

---

## Tech Stack

- **Language:** Python  
- **Backend:** FastAPI  
- **Vector Search:** FAISS  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **LLM:** Ollama (local inference)  
- **Data Storage:** JSON  

---

## Key Features

- End-to-end RAG pipeline implementation  
- FAISS-based semantic retrieval  
- Local LLM execution using Ollama  
- Configurable LLM providers (mock, OpenAI, Ollama)  
- Input and output safety guardrails  
- Fallback mechanism for low-confidence retrieval  
- FastAPI backend with REST endpoints  
- CLI-based chatbot interaction  

---

## Project 

medical_chatbot_evolution/
│
├── report/
│
├── version_01_rule_based_chatbot/
├── version_02_keyword_nlp_chatbot/
├── version_03_intent_ml_chatbot/
├── version_04_context_retrieval_chatbot/
│
├── version_05_rag_medical_chatbot/
│   │
│   ├── 01_retrieval_baseline/
│   ├── 02_vector_retrieval/
│   ├── 03_data_ingestion_chunking/
│   ├── 04_generation_guardrails/
│   │
│   ├── 05_final_integrated_rag/
│   └── 05_final_integrated_rag_v2_ollama/
│
└── README.md


---

## Installation & Setup

Clone repository:
git clone <your-repo-link>
cd medical_chatbot_evolution


Install dependencies:
pip install -r requirements.txt


---

## Running the Final System (V2)

Navigate to final version:

cd version_05_rag_medical_chatbot/05_final_integrated_rag_v2_ollama


Start Ollama:

ollama run 


Build vector database:

python scripts/ingest_data.py

Run API server:

python -m uvicorn app.main:app --reload

Access API: http://127.0.0.1:8000/docs


---

## Workflow Summary

1. User submits a query  
2. Input is validated using safety rules  
3. Query is converted into embeddings  
4. FAISS retrieves top-K relevant chunks  
5. Context is injected into prompt  
6. LLM generates response  
7. Output is filtered for safety  
8. Final response is returned  

---

## Evaluation Strategy

Current evaluation:
- Similarity score inspection  
- Retrieved context validation  
- Manual verification of outputs  

Planned improvements:
- Precision@K  
- Recall@K  
- Answer relevance scoring  
- Groundedness validation  
- Safety compliance metrics  

---

## Limitations

- Limited dataset size  
- Domain restricted to predefined topics  
- Performance dependent on retrieval quality  
- Not suitable for real clinical use  
- No real-time knowledge updates  

---

## Future Improvements

- Integration of larger medical datasets (PubMed, WHO)  
- Hybrid search (BM25 + embeddings)  
- Multi-turn conversation memory  
- Web-based user interface  
- Deployment and scaling  
- Real-time data updates  
- User feedback loop  

---

## Learning Outcomes

This project enabled understanding of:

- Evolution of chatbot architectures  
- Semantic search and embeddings  
- Vector databases (FAISS)  
- Retrieval-Augmented Generation (RAG)  
- LLM integration using Ollama  
- Safety mechanisms in AI systems  
- Backend API development and system design  

---

## Author Perspective

This project was designed as a **learning-driven system evolution**, not just a final product.

### Before
- Limited understanding of AI system architecture  
- Focus on small, isolated programs  
- Basic knowledge of chatbots  

### After
- Ability to design full AI pipelines  
- Understanding of retrieval + generation systems  
- Experience integrating multiple technologies  
- Awareness of real-world constraints and safety  

This project represents a transition from **writing code → designing intelligent systems**.

---

## Contributing

Contributions are welcome.

1. Fork the repository  
2. Create a feature branch  
3. Commit changes  
4. Submit a pull request  

---

## License

This project is intended for educational and research purposes only.

