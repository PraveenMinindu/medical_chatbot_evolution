# MedBot A++++ -- Day 4: Full RAG Pipeline

---

## What Is RAG?

RAG stands for Retrieval-Augmented Generation. It is a technique that combines
two systems that are each insufficient on their own:

    A retrieval system alone (Days 1-3):
        Returns raw stored text. Cannot synthesise multiple sources.
        Cannot rephrase. Cannot handle questions that span topics.

    An LLM alone (without retrieval):
        Answers from training data, which may be outdated or wrong.
        Prone to hallucination -- generating plausible but false information.
        Cannot cite sources or explain where the answer came from.

    RAG combines both:
        Retrieval finds the right factual content.
        The LLM reads that content and generates a coherent, grounded answer.
        The answer is based on real retrieved text, not invented from memory.

---

## What Changed from Day 3 to Day 4

| Aspect | Day 3 | Day 4 |
|---|---|---|
| Answer source | Raw chunk text returned verbatim | LLM synthesises retrieved chunks |
| Multi-topic queries | Shows best chunk, lists others | LLM combines all relevant chunks |
| Project structure | Flat src/ folder | Layered app/ with rag/, llm/, safety/, utils/ |
| LLM provider | None | OpenAI, Ollama, or Mock (configurable) |
| Prompt building | None | Grounded system prompt + numbered context blocks |
| Output safety | Input check only | Input check + output scan after generation |
| Response schema | Simple dataclass | RAGResponse with full metadata for Day 5 API |

---

## How Retrieval and Generation Work Together

```
User query: "why do I feel hot and dizzy?"
                    |
                    v
    ┌─── Safety Guardrail (input) ───┐
    │  Check restricted keywords     │
    │  Check urgent phrases          │
    └──────────── pass ──────────────┘
                    |
                    v
    ┌─── FAISS Retrieval ────────────┐
    │  Embed query (384 dims)        │
    │  Search 46 chunk vectors       │
    │  Return top-5 by cosine sim    │
    └──────────── chunks ────────────┘
                    |
          [fever-chunk-002: 0.79]
          [dehyd-chunk-004: 0.74]
          [fever-chunk-000: 0.68]
                    |
                    v
    ┌─── Prompt Builder ─────────────┐
    │  System: role + rules          │
    │  User:   context + question    │
    └──────────── prompt ────────────┘
                    |
                    v
    ┌─── LLM Generation ─────────────┐
    │  OpenAI / Ollama / Mock        │
    │  Reads context, generates      │
    └──────────── answer ────────────┘
                    |
                    v
    ┌─── Output Safety Scan ─────────┐
    │  Check for policy violations   │
    └──────────── final ─────────────┘
                    |
                    v
    RAGResponse (answer + sources + scores + metadata)
```

---

## How the Prompt Reduces Hallucination

The system prompt contains two critical instructions:

    "Answer ONLY using the information provided in the context sections below."
    "If the context does not contain enough information to answer the question, say so clearly."

These two lines together implement grounding:
- The LLM is told not to use its own training knowledge.
- The LLM is told what to do when the retrieved context is insufficient.

Without these instructions, the LLM might:
- Add information from training data that contradicts the retrieved facts.
- Confabulate specific details that sound medical but are invented.
- Answer with false confidence instead of acknowledging uncertainty.

With these instructions, the LLM stays within the retrieved content.
Low temperature (0.2) further reduces creative variation and keeps answers consistent.

---

## Project Structure

```
medical_chatbot_A++++/
|
|-- app/
|   |-- config.py              All configuration (paths, LLM, retrieval, safety)
|   |-- schemas.py             Shared data structures (RAGResponse, RetrievedChunk, etc.)
|   |
|   |-- rag/
|   |   |-- chunking.py        Document -> overlapping text chunks
|   |   |-- embeddings.py      Sentence-transformer model + embed functions
|   |   |-- vector_store.py    FAISS build, save, load, search
|   |   |-- retrieval.py       KnowledgeRetriever: top-k semantic search
|   |   |-- prompt_builder.py  Build grounded system + user prompt
|   |   `-- pipeline.py        RAGPipeline: orchestrate all steps
|   |
|   |-- llm/
|   |   `-- generator.py       LLM provider abstraction (OpenAI/Ollama/mock)
|   |
|   |-- safety/
|   |   `-- guardrails.py      Input check + output scan
|   |
|   `-- utils/
|       `-- text.py            clean_text, contains_whole_word
|
|-- data/
|   |-- raw/medical_knowledge.md     Source document
|   `-- processed/chunks.json        Chunked output (auto-generated)
|
|-- vector_store/                    FAISS index (auto-generated)
|
|-- scripts/
|   `-- ingest_data.py               Build the FAISS index from the document
|
|-- run.py                           CLI chatbot loop
`-- requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt

# Step 1: build the vector index
python scripts/ingest_data.py

# Step 2: configure LLM provider (optional -- mock works without any key)
export LLM_PROVIDER=mock          # default -- no API needed, shows pipeline structure
export LLM_PROVIDER=openai        # requires: export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=ollama        # requires: ollama pull llama3.2 && ollama serve

# Step 3: run
python run.py
```

---

## LLM Provider Configuration

| Provider | Setup | Cost |
|---|---|---|
| mock | No setup | Free -- shows pipeline structure only |
| openai | export OPENAI_API_KEY=sk-... | Paid per token |
| ollama | ollama pull llama3.2 then ollama serve | Free -- runs on your machine |

---

## Remaining Risks

The system is much better than pure retrieval but not perfect:

- The LLM can still hallucinate if the retrieved chunks are ambiguous.
- Output scanning catches common violations but not all possible unsafe phrasings.
- The knowledge base is still small (46 chunks). Gaps in coverage mean the LLM
  sometimes receives insufficient context and should say so.
- There is no conversation memory. Each query is processed independently.
- The system has not been tested by medical professionals. It is for education only.

---

## Why Day 5 Adds FastAPI and Deployment Architecture

The RAGResponse schema was designed from Day 4 to be API-ready. It contains
all the fields a REST endpoint needs: answer, sources, scores, topics, metadata.

Day 5 will wrap the RAGPipeline in a FastAPI application that:
- Exposes a POST /query endpoint accepting JSON
- Returns structured RAGResponse as JSON
- Adds request validation with Pydantic
- Adds proper error handling and HTTP status codes
- Enables the chatbot to be called from a web frontend, mobile app, or another service

This transforms the CLI chatbot into a deployable web service.
