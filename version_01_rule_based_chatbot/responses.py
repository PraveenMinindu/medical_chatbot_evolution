# responses.py
# Medical Chatbot - Version A
# Stores all responses separately from chatbot logic

responses = {
    "fever": (
        "Fever is a temporary increase in body temperature, often caused by infection.\n"
        "Normal body temperature is around 37°C (98.6°F).\n"
        "Note: This is general information only. Please consult a doctor if symptoms persist."
    ),
    "headache": (
        "A headache is pain in the head that may occur due to stress, dehydration, or illness.\n"
        "Common types include tension headaches and migraines.\n"
        "Note: This is general information only. Please consult a doctor if symptoms persist."
    ),
    "cold": (
        "The common cold is a viral infection affecting the nose and throat.\n"
        "Symptoms include runny nose, sneezing, sore throat, and mild fever.\n"
        "Note: This is general information only. Please consult a doctor if symptoms persist."
    ),
    "cough": (
        "Coughing is a reflex that helps clear the airways of irritants or mucus.\n"
        "It can be caused by a cold, infection, or throat irritation.\n"
        "Note: This is general information only. Please consult a doctor if symptoms persist."
    ),
    "dehydration": (
        "Dehydration occurs when the body loses more fluids than it takes in.\n"
        "Symptoms include thirst, dry mouth, dizziness, and dark urine.\n"
        "Note: This is general information only. Please consult a doctor if symptoms persist."
    ),
}

greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]

exit_commands = ["quit", "exit", "bye", "goodbye", "stop"]

unsafe_topics = [
    "diagnose", "diagnosis", "prescription", "prescribe", "treatment",
    "medicine", "medication", "dose", "dosage", "surgery", "emergency",
    "cancer", "diabetes", "heart attack", "stroke", "drug"
]

unsafe_response = (
    "I am not able to provide diagnosis, prescriptions, or treatment advice.\n"
    "Please consult a qualified healthcare professional for medical guidance."
)

fallback_response = (
    "Sorry, I can only provide basic information about fever, headache, cold, cough, and dehydration.\n"
    "For anything else, please consult a healthcare professional."
)

greeting_response = (
    "Hello! I am MedBot, your Medical Information Assistant.\n"
    "I can answer basic questions about:\n"
    "  - Fever\n"
    "  - Headache\n"
    "  - Cold\n"
    "  - Cough\n"
    "  - Dehydration\n"
    "Type 'quit' at any time to exit."
)
