import pytest
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM

# ---- Ollama Local Model Wrapper ----
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        return response.json()["response"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name

# ---- Initialize model ----
ollama_model = OllamaModel(model_name="mistral")

# ---- Simulated chatbot responses ----
def get_response(keyword):
    responses = {
        "paracetamol": "The recommended Paracetamol dosage for adults is 500mg to 1000mg every 4 to 6 hours as needed.",
        "diabetes":    "Symptoms include excessive thirst, frequent urination, fatigue",
        "cancer":      "Paracetamol cures cancer and costs $5"
    }
    return responses.get(keyword, "I don't know")

# ---- Test 1: Paracetamol relevancy ----
def test_answer_relevancy():
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="What is the Paracetamol dosage for adults?",
        actual_output=get_response("paracetamol")
    )
    assert_test(test_case, [metric])

# ---- Test 2: Diabetes relevancy ----
def test_diabetes_relevancy():
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="What are the symptoms of diabetes?",
        actual_output=get_response("diabetes")
    )
    assert_test(test_case, [metric])
