# MedBot A+ — Medical Information Chatbot
## Version A+ — Keyword-Based NLP

---

## What Changed from Version A?

| Feature | Version A | Version A+ |
|---|---|---|
| Input matching | Exact string only | Keywords inside sentences |
| "I have fever" | ❌ No match | ✅ Finds "fever" |
| "my head hurts" | ❌ No match | ✅ Finds "head" |
| Multiple symptoms | ❌ Only first match | ✅ Detects all |
| Text cleaning | Basic lowercase | Lowercase + punctuation removal + regex |
| NLP concepts | None | Tokenization, keyword extraction |

---

## What You Learn in This Version

- How to **preprocess raw text** (lowercase, remove punctuation)
- How to **search for keywords** inside full sentences
- How to **map multiple keywords** to the same topic
- How to handle **multiple symptoms in one message**
- Why **language is hard for machines** — this version still has limits
- Introduction to the concept of **NLP pipelines**

---

## File Structure

```
medical-chatbot-A-plus/
│
├── run.py          # Entry point — start the chatbot here
├── chatbot.py      # Core logic — keyword detection and response selection
├── keywords.py     # Data layer — keywords, responses, and word lists
├── utils.py        # NLP utilities — text cleaning and keyword extraction
└── README.md       # This file
```

---

## How to Run

```bash
python run.py
```

Requires Python 3.x only. No external libraries needed.

---

## Example Conversations

### Natural sentence input
```
You: I have fever today
Bot: Fever is a temporary increase in body temperature, often caused by infection.
```

### Synonym matching
```
You: my head hurts badly
Bot: Head pain may be related to a headache, stress, dehydration, or tension.
```

### Multiple symptoms detected
```
You: I have fever and headache
Bot: I found 2 topics in your message:

  [1] Fever is a temporary increase in body temperature...

  [2] A headache is pain in the head that may occur due to stress...
```

### Unsafe topic
```
You: what medicine should I take
Bot: I am not able to provide diagnosis, prescriptions, or treatment advice.
     Please consult a qualified healthcare professional.
```

### Exit
```
You: bye
Bot: Goodbye! Stay healthy. 🌿
```

---

## How Version A+ Works

```
User types sentence (natural language)
           ↓
   utils.clean_text()
   → lowercase
   → remove punctuation
   → strip whitespace
           ↓
   Check for exit words
   Check for greetings
   Check for unsafe topics
           ↓
   Search keywords in cleaned text
   (keyword in sentence → return response)
           ↓
   Multiple matches? → show all responses
   Single match?    → show one response
   No match?        → fallback message
```

---

## Keywords Covered

| Topic | Keywords Detected |
|---|---|
| Fever | fever, temperature, hot |
| Headache | headache, head, migraine |
| Cold | cold, runny, sneeze, sneezing |
| Cough | cough, coughing, throat |
| Dehydration | dehydration, dehydrated, thirsty, dizzy, dizziness |

---

## Limitations of Version A+

These limitations will motivate Version A++:

| Limitation | Example | Solution in A++ |
|---|---|---|
| No understanding of intent | "I don't have fever" still matches "fever" | ML intent classification |
| Keyword order doesn't matter | Misses context | Intent + context awareness |
| No negation handling | "no fever" still triggers fever response | Smarter classification |
| Cannot learn new patterns | Fixed keywords only | Trained ML model |

---

## Project Roadmap

| Version | Type | Status |
|---|---|---|
| Version A | Rule-Based | ✅ Complete |
| **Version A+** | Keyword NLP | ✅ Complete |
| Version A++ | ML Intent Classification | 🔜 Next |
| Version A+++ | Knowledge Retrieval | 🔜 Planned |
| Version A++++ | Generative AI (RAG) | 🔜 Planned |

---

*This project is for educational purposes only. MedBot does not provide medical advice.*
