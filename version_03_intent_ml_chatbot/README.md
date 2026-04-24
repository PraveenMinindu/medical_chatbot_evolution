# Intent-Based Medical Information Chatbot (A++)

## 1. Project Title

Intent-Based Medical Information Chatbot

## 2. Overview

This project is a beginner-friendly medical information chatbot built with Python and scikit-learn. It uses a traditional machine learning workflow to predict the user's intent and return a suitable response.

The chatbot is designed for educational medical information only. It does not diagnose diseases, prescribe medicines, or replace a qualified healthcare professional.

## 3. Why this version is called A++

This version is called **A++** because it improves on earlier chatbot versions by using a machine learning model to recognize intent instead of depending only on exact rules or simple keyword matching.

In this version, the chatbot:

- learns from training examples
- converts text into numeric features
- predicts the most likely intent
- chooses a response from that intent
- uses confidence-based fallback when uncertain

## 4. Difference between A, A+, and A++

### A = Rule-based chatbot

- Uses exact if/else rules
- Works only when user input matches expected patterns closely
- Hard to scale as topics grow

### A+ = Keyword-based chatbot

- Looks for words like `fever` or `headache`
- More flexible than exact rules
- Still weak when users phrase the same idea differently

### A++ = Intent-based chatbot

- Learns from example sentences for each intent
- Understands different phrasings better than keyword matching
- Uses ML prediction instead of only checking for words
- Can handle uncertainty with confidence thresholds and fallback behavior

## 5. Features

- Terminal-based chatbot
- Intent prediction using machine learning
- Beginner-friendly text preprocessing
- Multiple responses for each intent
- Confidence-based fallback for uncertain inputs
- Safe handling for risky or unsupported medical questions
- Clear medical disclaimer
- Simple tests for sample inputs

## 6. Technologies used

- Python
- scikit-learn
- joblib
- json
- random
- re
- pathlib
- unittest

## 7. Project structure

```text
medical_chatbot_A_plus_plus/
│
├── data/
│   └── intents.json
│
├── models/
│   ├── intent_model.joblib
│   └── vectorizer.joblib
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── config.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_samples.py
│
├── run.py
├── requirements.txt
└── README.md
```

## 8. How the chatbot works

The chatbot follows a simple machine learning pipeline:

1. **Training data**  
   The file `data/intents.json` contains example user messages grouped by intent tags such as `greeting`, `fever_info`, and `out_of_scope`.

2. **Preprocessing**  
   User text is converted to lowercase, punctuation is removed, and extra spaces are cleaned.

3. **Feature extraction**  
   `TfidfVectorizer` converts text into numeric features so the classifier can learn patterns from words and short phrases.

4. **Intent classification**  
   `LogisticRegression` is trained on these features to predict the user's intent.

5. **Confidence check**  
   If the model confidence is too low, the chatbot uses a fallback response instead of pretending to know the answer.

6. **Response selection**  
   After predicting the intent, the chatbot randomly selects one response from that intent's response list.

7. **Safety checks**  
   Risky and urgent inputs are handled with safe messages rather than diagnosis or treatment advice.

## 9. How to train the model

Open a terminal in the project folder:

```bash
cd medical_chatbot_A_plus_plus
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train
```

This creates:

- `models/intent_model.joblib`
- `models/vectorizer.joblib`

## 10. How to run the chatbot

Start the chatbot with:

```bash
python run.py
```

If model files do not exist yet, `run.py` will train them automatically first.

## 11. Sample conversation

```text
You: hello
Bot: Hello! I am a medical information chatbot. You can ask me about fever, headache, cough, cold, or dehydration.
(Predicted intent: greeting, confidence: 0.82)

You: my head hurts badly
Bot: Headaches can be linked to lack of sleep, stress, dehydration, illness, or eye strain.
(Predicted intent: headache_info, confidence: 0.61)

You: what medicine should I take
Bot: I can only provide general educational information. Please consult a qualified healthcare professional.
(Predicted intent: out_of_scope, confidence: 0.88)

You: bye
Bot: Goodbye! Stay healthy.
```

## 12. Safety limitations

This chatbot is intentionally limited.

- It does not diagnose medical conditions.
- It does not prescribe medicines.
- It does not give emergency treatment instructions.
- It is not a replacement for a doctor or other qualified healthcare professional.
- It only supports a small set of general health topics.

If a user asks a risky or urgent question, the chatbot responds with a safe message instead of specific medical advice.

## 13. Future improvements

Natural next steps for future versions:

- **A+++ Retrieval-based chatbot**  
  Search a trusted medical knowledge source and return the most relevant educational passages.

- **A++++ GenAI + RAG chatbot**  
  Combine retrieval with a generative model to create more natural and context-aware answers.

- Add more health topics
- Improve training data variety
- Add a web interface
- Add logging and evaluation tools

## Why A++ is better than A+

A+ depends mostly on seeing matching keywords. A++ is better because it learns from different example sentences for the same intent. That means inputs like `my head hurts`, `tell me about headache`, and `what causes head pain` can all map to the same intent even though the wording is different.

## Current limitations

- The chatbot only knows a small number of intents
- It is trained on a small handcrafted dataset
- It does not remember previous conversation context
- It cannot retrieve facts from external medical sources

## Why A+++ retrieval-based chatbot is the next natural upgrade

An intent-based chatbot is useful for mapping user questions to broad categories, but it can only answer from prewritten responses. A retrieval-based chatbot improves this by searching a trusted knowledge base and returning relevant information. That makes answers more flexible, more informative, and easier to scale across many medical topics.
