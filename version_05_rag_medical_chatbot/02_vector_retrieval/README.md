# MedBot A++++ — Day 2: FAISS Vector Index + Top-K Retrieval

---

## What Changed from Day 1 to Day 2

| Aspect | Day 1 | Day 2 |
|---|---|---|
| Vector storage | numpy array in memory | FAISS IndexFlatIP on disk |
| Startup | re-embeds all entries every time | loads saved index in milliseconds |
| Results returned | 1 best match | top_k matches (default: 3) |
| Deduplication | none | keeps one result per topic |
| Multi-topic queries | only 1 answer | all relevant topics returned |
| LLM readiness | no context window | top_k chunks ready as context |

---

## What Is a Vector Index?

In Day 1, finding the best match meant computing similarity between the query
and every single KB entry one by one. This is fine for 30 entries.

For 1 million entries, looping through every one would take many seconds per query.

A vector index organises vectors into a data structure that allows fast search.
FAISS (Facebook AI Similarity Search) is the industry-standard library for this.
It is used in production at companies like Meta, Spotify, and Airbnb.

In this project, FAISS uses IndexFlatIP -- a flat index with Inner Product search.
After normalising all vectors to unit length, inner product equals cosine similarity.
The index is built once and saved to vector_store/ so it is reused on every run.

---

## What Is Top-K Retrieval?

Top-K retrieval means returning the K most similar entries from the index,
not just the single best one.

This is essential for RAG because:
- A query like "I feel hot and weak" is relevant to both fever AND dehydration.
- Returning only the single best match loses the dehydration information.
- A future LLM needs all relevant chunks in its context to give a complete answer.

In this system, top_k=3 by default. The chatbot returns the best single answer
for display, plus all top_k chunks as sources for future LLM use.

---

## How Metadata Is Stored

FAISS stores only the numeric vectors. It has no concept of text, topics, or answers.
Metadata (source_id, topic, question, answer) is stored separately in:

    vector_store/metadata.json

The integer positions in the FAISS index align exactly with the positions
in the metadata list. When FAISS returns index position 5, we look up
metadata[5] to get the text and topic for that result.

---

## Why This Is Closer to Real RAG

A complete RAG pipeline looks like this:

    User query
        |
        v
    Embed query
        |
        v
    Vector index search (FAISS) --> top_k chunks retrieved
        |
        v
    Assemble context window: [chunk_1, chunk_2, chunk_3]
        |
        v
    LLM generates answer grounded in retrieved context
        |
        v
    Final answer to user

This project now implements everything up to the LLM step.
The top_k chunks are assembled and ready. Day 3 adds the LLM.

---

## Project Structure

```
medical_chatbot_A++++/
|
|-- data/
|   `-- knowledge_base.json       30 Q/A entries, 5 topics
|
|-- vector_store/                 Created automatically on first run
|   |-- index.faiss               FAISS binary index (vectors)
|   `-- metadata.json             Aligned metadata for each vector
|
|-- src/
|   |-- config.py                 Paths, model, TOP_K, threshold, safety
|   |-- utils.py                  Text cleaning, KB loading, safety check
|   |-- embeddings.py             load_model(), embed_texts(), embed_query()
|   |-- vector_store.py           build_index(), load_index(), search()
|   |-- retrieval.py              KnowledgeRetriever, top-k, deduplication
|   |-- chatbot.py                MedicalChatbot, ChatResponse
|   `-- demo_queries.py           Multi-topic retrieval demonstration
|
|-- run.py                        CLI loop with source display
`-- requirements.txt              sentence-transformers, faiss-cpu, numpy
```

---

## How to Run

```bash
pip install -r requirements.txt

# First run: builds and saves the FAISS index
python run.py

# Force rebuild the index (if KB changes)
python run.py --rebuild

# Run the demo
python -m src.demo_queries
```

---

## Remaining Limitations

- Knowledge base is still 30 hand-written entries. Real RAG ingests documents.
- Each entry is one full Q/A pair. Real RAG chunks long documents into small pieces.
- The KB is small enough that IndexFlatIP (brute force) is fine. Larger KBs need
  approximate nearest-neighbour indexes (IndexIVFFlat, HNSW).
- There is still no LLM. The best retrieved answer is returned directly, not generated.

---

## Why Day 3 Focuses on Chunking and Knowledge Ingestion

Right now, each KB entry is a single manually-written question-answer pair.
In a real RAG system, knowledge comes from documents -- PDFs, articles, reports.

A long document cannot be indexed as one unit. A 10-page medical article would
produce one giant vector that loses all the specific details buried inside.

Chunking splits documents into small overlapping pieces -- typically 200 to 500
tokens each. Each chunk is embedded and stored as a separate vector.

Day 3 will:
1. Accept real text documents as input.
2. Split them into chunks with overlap.
3. Embed each chunk separately.
4. Store all chunks in the FAISS index.
5. Retrieve relevant chunks for any query.

This is how real RAG systems work.
