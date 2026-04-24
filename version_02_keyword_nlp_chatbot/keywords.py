# keywords.py
# Medical Chatbot - Version A+
# Stores all keywords and responses
#
# KEY DESIGN: keywords are grouped by TOPIC.
# This fixes the duplicate-response bug where both "headache" and "head"
# matched the same sentence and returned two responses for one topic.
#
# Now each topic has ONE response and MANY keywords that trigger it.
# Only one response per topic is ever returned — no duplicates.

# topic_keywords maps: topic_name → (list of keywords, response text)
topic_keywords = {
    "fever": {
        "keywords": ["fever", "temperature", "hot"],
        "response": (
            "Fever is a temporary increase in body temperature, often caused by infection.\n"
            "Normal body temperature is around 37°C (98.6°F).\n"
            "Note: This is general information only. Please consult a doctor if symptoms persist."
        )
    },
    "headache": {
        "keywords": ["headache", "head", "migraine"],
        "response": (
            "A headache is pain in the head that may occur due to stress, dehydration, or illness.\n"
            "Common types include tension headaches and migraines.\n"
            "Note: This is general information only. Please consult a doctor if symptoms persist."
        )
    },
    "cold": {
        "keywords": ["cold", "runny", "sneeze", "sneezing"],
        "response": (
            "The common cold is a viral infection affecting the nose and throat.\n"
            "Symptoms include runny nose, sneezing, sore throat, and mild fever.\n"
            "Note: This is general information only. Please consult a doctor if symptoms persist."
        )
    },
    "cough": {
        "keywords": ["cough", "coughing", "throat"],
        "response": (
            "Coughing is a reflex that helps clear the airways of irritants or mucus.\n"
            "It can be caused by a cold, infection, or throat irritation.\n"
            "Note: This is general information only. Please consult a doctor if symptoms persist."
        )
    },
    "dehydration": {
        "keywords": ["dehydration", "dehydrated", "thirsty", "dizzy", "dizziness"],
        "response": (
            "Dehydration occurs when the body loses more fluids than it takes in.\n"
            "Symptoms include thirst, dry mouth, dizziness, and dark urine.\n"
            "Note: This is general information only. Please consult a doctor if symptoms persist."
        )
    },
}

greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]

exit_words = ["quit", "exit", "bye", "goodbye", "stop"]

unsafe_keywords = [
    "diagnose", "diagnosis", "prescription", "prescribe",
    "treatment", "medicine", "medication", "dose", "dosage",
    "surgery", "emergency", "cancer", "diabetes",
    "heart attack", "stroke", "drug"
]

greeting_response = (
    "Hello! I am MedBot A+, your Keyword-Based Medical Information Assistant.\n"
    "I understand natural sentences — just speak normally!\n"
    "You can ask me about:\n"
    "  - Fever / high temperature / feeling hot\n"
    "  - Headache / head pain / migraine\n"
    "  - Cold / runny nose / sneezing\n"
    "  - Cough / sore throat\n"
    "  - Dehydration / feeling dizzy / thirsty\n"
    "Type 'quit' to exit."
)

unsafe_response = (
    "I am not able to provide diagnosis, prescriptions, or treatment advice.\n"
    "Please consult a qualified healthcare professional for medical guidance."
)

fallback_response = (
    "Sorry, I could not find a match for your question.\n"
    "I can help with: fever, headache, cold, cough, or dehydration.\n"
    "Try rephrasing — for example: 'I have a headache' or 'tell me about fever'."
)
