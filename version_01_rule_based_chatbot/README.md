# MedBot — Medical Information Chatbot
## Version A — Rule-Based System

---

## Project Overview

MedBot Version A is a rule-based medical information chatbot built with pure Python.  
It answers basic health questions using predefined rules — no machine learning, no AI APIs.

This is **Version A** of a 5-stage project that evolves from rule-based logic → Generative AI.

---

## Features

| Feature | Description |
|---|---|
| Greeting | Responds to hello, hi, hey |
| Medical Info | Answers questions about 5 common health topics |
| Unsafe Topic Guard | Refuses diagnosis, prescriptions, treatment questions |
| Unknown Fallback | Handles unrecognized questions gracefully |
| Exit Command | Ends conversation cleanly |

---

## Topics Covered

- **Fever** — what it is and general information
- **Headache** — causes and types
- **Cold** — symptoms and nature
- **Cough** — what it is and causes
- **Dehydration** — definition and symptoms

---

## Topics NOT Covered (By Design)

This chatbot will NOT answer:

- Medical diagnosis
- Prescriptions or medication dosage
- Treatment plans
- Emergency advice

These restrictions are intentional and reflect responsible AI design.

---

## File Structure

```
medical-chatbot-A/
│
├── run.py          # Entry point — start the chatbot here
├── chatbot.py      # Core logic — rule checking and response selection
├── responses.py    # Data layer — all responses stored here
└── README.md       # This file
```

---

## How to Run

```bash
python run.py
```

No external libraries needed. Requires Python 3.x only.

---

## Example Conversation

```
You: hello
Bot: Hello! I am MedBot, your Medical Information Assistant.
     I can answer basic questions about: Fever, Headache, Cold, Cough, Dehydration

You: what is fever
Bot: Fever is a temporary increase in body temperature, often caused by infection.

You: how to treat cancer
Bot: I am not able to provide diagnosis, prescriptions, or treatment advice.
     Please consult a qualified healthcare professional.

You: quit
Bot: Goodbye! Stay healthy.
```

---

## Conversation Flow

```
User types input
      ↓
chatbot.py cleans and checks input
      ↓
  Is it a greeting?  → greeting response
  Is it an exit?     → goodbye and stop
  Is it unsafe?      → safety message
  Is it a known topic? → medical response
  Otherwise          → fallback message
      ↓
Response printed to user
```

---

## What This Version Teaches

- Chatbot conversation loop design
- Rule-based response systems
- Input cleaning and matching
- Separation of data and logic (responses.py vs chatbot.py)
- Safe response design principles

---

## Limitations of Version A

These limitations motivate the next versions:

| Limitation | Solution in Next Version |
|---|---|
| Only exact keyword matching | Version A+ — NLP keyword extraction |
| No understanding of sentence structure | Version A++ — ML intent classification |
| Cannot answer from documents | Version A+++ — Knowledge retrieval |
| Cannot generate new responses | Version A++++ — Generative AI (RAG) |

---

## Project Roadmap

| Version | Type | Status |
|---|---|---|
| **Version A** | Rule-Based | ** Complete** |
| Version A+ | Keyword NLP |--> Next |
| Version A++ | ML Intent Classification | --> Planned |
| Version A+++ | Knowledge Retrieval | -->Planned |
| Version A++++ | Generative AI (RAG) | -->Planned |

---

*This project is for educational purposes only. MedBot does not provide medical advice.*
