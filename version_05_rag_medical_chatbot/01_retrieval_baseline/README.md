# MedBot A++++ — Semantic Embedding Medical Chatbot
## Day 1: TF-IDF Retrieval → Semantic Embedding Retrieval

---

## What Changed from A+++ to A++++

| Aspect | A+++ | A++++ |
|---|---|---|
| Vector type | Sparse TF-IDF (mostly zeros) | Dense embeddings (all values active) |
| Vector size | 84 dimensions (vocabulary size) | 384 dimensions (fixed) |
| "coughing" vs "cough" | Different dimensions -- no match | Similar vectors -- correct match |
| "I feel feverish" | Zero score (word not in KB) | High score (meaning understood) |
| Requires word overlap | Yes | No |
| Understands synonyms | No | Yes |
| Model training | None (rule-based counts) | Pre-trained on billions of sentences |

---

## What Is Semantic Similarity?

In A+++, two sentences were similar if they shared the same words.

In A++++, two sentences are similar if they mean the same thing -- even if
they use completely different words. The sentence-transformer model was trained
on billions of sentence pairs and learned to map sentences with similar meanings
to nearby points in a 384-dimensional space.

When two vectors point in nearly the same direction, their cosine similarity
is close to 1.0. This means the sentences are semantically close, regardless
of the words used.

---

## How Text Becomes a Dense Vector

1. The sentence is tokenised into subword pieces.
2. Each token is looked up in the model's 30,000-token vocabulary.
3. A transformer processes all tokens together, considering context.
4. The model produces one 384-dimensional vector per sentence.
5. This vector encodes the sentence's meaning as a point in semantic space.

---

## File Structure

```
medical_chatbot_A++++/
|
|-- data/
|   `-- knowledge_base.json     30 Q/A entries with topic and source_id
|
|-- src/
|   |-- config.py               Model name, threshold, messages, safety keywords
|   |-- utils.py                Text cleaning, JSON loading, safety detection
|   |-- embeddings.py           Model loading, embed_texts(), embed_query()
|   |-- retrieval.py            KnowledgeRetriever: embed KB, get_best_match()
|   |-- chatbot.py              MedicalChatbot: safety + retrieval + threshold
|   `-- demo_queries.py         Demonstration of semantic vs TF-IDF retrieval
|
|-- run.py                      CLI chatbot loop
|-- requirements.txt            sentence-transformers, numpy, scikit-learn
`-- README.md                   This file
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the chatbot
```bash
python run.py
```

### Run the demo
```bash
python src/demo_queries.py
```

---

## Remaining Limitations

- The model embeds meaning but still has no conversation memory. Each question is processed independently.
- The system only retrieves from stored KB entries. It cannot generate new answers.
- The KB covers only five topics. Out-of-scope questions return a fallback.
- Embeddings are computed in memory on startup. With a large KB, this would be slow and memory-intensive.
- There is no persistent index. Every startup recomputes all embeddings from scratch.

---

## Why a Vector Database Is the Next Step (Day 2)

Computing all embeddings on every startup is acceptable for 30 entries.
For thousands or millions of entries, this would take minutes and consume
large amounts of memory.

A vector database solves this by:
1. Storing pre-computed embeddings persistently on disk.
2. Building an index structure for fast approximate nearest-neighbour search.
3. Allowing the KB to be updated without recomputing the entire index.
4. Scaling to millions of entries with sub-second query times.

Day 2 will replace the in-memory numpy matrix with a proper vector database.
