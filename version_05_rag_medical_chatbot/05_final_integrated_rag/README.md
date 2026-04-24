# Part 5: Final Integrated RAG Medical Chatbot

This folder contains the final integrated chatbot that combines:

- Semantic retrieval with sentence embeddings
- FAISS vector search with chunked document ingestion
- Prompt grounding for generation
- LLM provider abstraction (`mock`, `openai`, `ollama`)
- Input and output safety guardrails
- CLI chatbot and FastAPI endpoint

## Structure

```text
05_final_integrated_rag/
  app/
    config.py
    schemas.py
    main.py
    rag/
      chunking.py
      embeddings.py
      vector_store.py
      retrieval.py
      prompt_builder.py
      pipeline.py
    llm/
      generator.py
    safety/
      guardrails.py
    utils/
      text.py
  scripts/
    ingest_data.py
  data/
    raw/medical_knowledge.md
    processed/
  vector_store/
  run.py
  requirements.txt
```

## Run

```bash
pip install -r requirements.txt
python scripts/ingest_data.py
python run.py
```

## API

```bash
uvicorn app.main:app --reload
```

- `GET /health`
- `POST /query` with JSON: `{"query": "why do I feel hot and dizzy?"}`

## LLM Provider

Default:

```bash
set LLM_PROVIDER=mock
```

OpenAI:

```bash
set LLM_PROVIDER=openai
set OPENAI_API_KEY=sk-...
```

Ollama:

```bash
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=llama3.2
```

