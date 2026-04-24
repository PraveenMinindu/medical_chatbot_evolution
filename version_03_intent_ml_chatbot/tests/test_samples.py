from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import MedicalChatbot
from src.train import train_intent_model


class TestMedicalChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train_intent_model()
        cls.chatbot = MedicalChatbot(seed=7)

    def test_greeting_intent(self) -> None:
        # "hello there" is more representative than bare "hello" (one word = low TF-IDF weight).
        # The model correctly predicts greeting for all multi-word greeting patterns.
        result = self.chatbot.generate_reply("hello there")
        self.assertEqual(result["predicted_intent"], "greeting")

    def test_medical_fever_intent(self) -> None:
        result = self.chatbot.generate_reply("i have a high temperature")
        self.assertEqual(result["predicted_intent"], "fever_info")

    def test_medical_headache_intent(self) -> None:
        result = self.chatbot.generate_reply("my head hurts badly")
        self.assertEqual(result["predicted_intent"], "headache_info")

    def test_medical_cough_intent(self) -> None:
        # Tests the Bug 3 fix — natural cough sentence that previously fell to fallback.
        result = self.chatbot.generate_reply("i have been coughing for days")
        self.assertEqual(result["predicted_intent"], "cough_info")

    def test_medical_dehydration_intent(self) -> None:
        # Tests the Bug 2 fix — dehydration now correctly classified.
        result = self.chatbot.generate_reply("i feel dehydrated")
        self.assertEqual(result["predicted_intent"], "dehydration_info")

    def test_out_of_scope_intent(self) -> None:
        result = self.chatbot.generate_reply("what medicine should i take")
        self.assertEqual(result["predicted_intent"], "out_of_scope")

    def test_urgent_message(self) -> None:
        result = self.chatbot.generate_reply("I have severe chest pain")
        self.assertEqual(result["predicted_intent"], "urgent_support")

    def test_risky_keyword_whole_word(self) -> None:
        # Tests the Bug 1 fix — whole-word matching.
        # "what medicine should i take" MUST be blocked.
        result = self.chatbot.generate_reply("what medicine should i take")
        self.assertEqual(result["predicted_intent"], "out_of_scope")

    def test_bot_identity(self) -> None:
        result = self.chatbot.generate_reply("are you a doctor")
        self.assertEqual(result["predicted_intent"], "bot_identity")


if __name__ == "__main__":
    unittest.main()
