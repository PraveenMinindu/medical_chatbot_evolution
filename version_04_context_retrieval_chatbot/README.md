# Knowledge-Based Medical Information Chatbot (Retrieval-Based) - A+++

## Project Overview

This project is Version A+++ of a structured chatbot evolution:

- A: Rule-based chatbot
- A+: Keyword-based chatbot
- A++: Intent-based chatbot using machine learning
- A+++: Retrieval-based chatbot using Information Retrieval
- A++++: GenAI + RAG chatbot

This version is a clear upgrade from A++ because it does not mainly try to classify the user's intent. Instead, it searches a knowledge base, measures text similarity, and returns the most relevant stored answer.

## What Makes A+++ Different from A++

In A++, the chatbot tries to guess which intent label matches the user's input. That works for known training examples, but it can struggle when the user asks the same thing in a new way.

In A+++, the chatbot performs retrieval:

1. It stores a set of medical questions and answers.
2. It converts the stored questions into vectors.
3. It converts the user's question into a vector.
4. It compares similarity scores.
5. It returns the answer from the most similar knowledge-base entry.

This means the chatbot behaves more like a search system than a classifier.

## What Is a Retrieval-Based Chatbot?

A retrieval-based chatbot does not generate brand-new answers. Instead, it searches through existing knowledge and retrieves the best matching response.

That makes it useful for learning:

- information retrieval
- vectorization
- similarity search
- knowledge-based systems

It is also an important step before learning GenAI and RAG systems, because many modern chatbots still depend on retrieval to find useful documents before generating a response.

## How TF-IDF Works

TF-IDF stands for Term Frequency-Inverse Document Frequency.

- Term Frequency (TF): how often a word appears in a text
- Inverse Document Frequency (IDF): how important that word is across all texts

Words that appear often in one question but not everywhere else get more importance. This helps the system focus on meaningful terms like `fever`, `dehydration`, or `headache`.

In this project, TF-IDF converts each knowledge-base question into a numeric vector.

## How Similarity Works

After vectorization, the chatbot compares the user's vector with all knowledge-base vectors using cosine similarity.

- A high cosine similarity means two texts are close in meaning based on shared important terms.
- A low cosine similarity means the user's question does not match the available knowledge well.

If the similarity score is high enough, the chatbot returns the matched answer.
If the score is too low, it returns a fallback message.

## Supported Topics

This chatbot only provides basic educational information about:

- fever
- headache
- cough
- cold
- dehydration

It does not diagnose, prescribe medicine, or provide emergency advice.

## Safety Behavior

If the user asks for:

- treatment
- medicine
- diagnosis

the chatbot returns:

`I can only provide general educational information. Please consult a healthcare professional.`

## Project Structure

```text
medical_chatbot_A_plus_plus_plus/
|
|-- data/
|   `-- knowledge_base.json
|
|-- src/
|   |-- chatbot.py
|   |-- retrieval.py
|   |-- utils.py
|   `-- config.py
|
|-- run.py
|-- requirements.txt
`-- README.md
```

## Core System Logic

The chatbot follows these steps:

1. Load the knowledge base from JSON.
2. Extract all stored questions.
3. Preprocess the text by lowercasing, removing punctuation, and cleaning spaces.
4. Convert the questions into TF-IDF vectors.
5. Convert the user's question into a TF-IDF vector.
6. Compute cosine similarity with every stored question.
7. Return the answer with the highest score.

## How to Run

### 1. Move into the project folder

```bash
cd medical_chatbot_A_plus_plus_plus
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the chatbot

```bash
python run.py
```

## Example Inputs and Outputs

**Example 1**

```text
You: what is fever
Bot: Fever is a temporary increase in body temperature, often happening when the body is responding to an infection or illness.
```

**Example 2**

```text
You: my head hurts
Bot: Head pain can happen for many reasons such as stress, dehydration, tiredness, or illness. This chatbot can only give general educational information.
```

**Example 3**

```text
You: what medicine should I take
Bot: I can only provide general educational information. Please consult a healthcare professional.
```

**Example 4**

```text
You: tell me about stomach pain
Bot: Sorry, I can only provide basic medical information about fever, headache, cough, cold, and dehydration.
```

## Why This Matters Before GenAI

Before building a GenAI + RAG chatbot, it is important to understand retrieval itself.

This project teaches the core idea that a chatbot can answer questions by:

- storing knowledge
- converting text into vectors
- measuring similarity
- retrieving the closest answer

RAG systems use the same basic idea, but with larger knowledge sources and a language model on top.

## Limitations

- The chatbot only answers from a small predefined knowledge base.
- It does not truly understand language the way a large language model does.
- It may return weak matches if the user's wording is very different from the stored questions.
- It does not provide diagnosis, medicine recommendations, or emergency guidance.
- It only covers five basic topics.

## Educational Summary

This A+++ version demonstrates a real shift from classification to retrieval.

- A++ asks: "Which intent is this?"
- A+++ asks: "Which stored question is most similar?"

That change is what makes this project a strong introduction to information retrieval and knowledge-based chatbot design.
