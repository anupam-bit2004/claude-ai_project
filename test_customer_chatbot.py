<<<<<<< HEAD
import pytest
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.models import DeepEvalBaseLLM

# Copy your OllamaModel class from test_medical_chatbot.py
# Then write 3 test functions below it

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
        "order_status": "Your order #12345 is currently out for delivery and will arrive by 6 PM today.",
        "order_return": "Yes, you can return this product within 30 days of delivery.",
        "customer_care": "Our customer care number is 1800-123-4567 and we also offer support via WhatsApp at 9876543210"
    }
    return responses.get(keyword, "I don't know")

def test_answer_relevancy():
    metric = AnswerRelevancyMetric(
        threshold=0.6,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="Where is my order #12345?",
        actual_output=get_response("order_status")
    )
    assert_test(test_case, [metric])


def test_faithfulness():
    metric = FaithfulnessMetric(
        threshold=0.6,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="Can I return this product?",
        actual_output=get_response("order_return"),
        retrieval_context=[                              # ✅ Added
            "Our return policy allows returns within 30 days of delivery",
            "Products must be unused and in original packaging",
            "Refunds are processed within 5-7 business days"
        ]
    )
    assert_test(test_case, [metric])

def test_hallucination():
    metric = HallucinationMetric(
        threshold=0.2,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="What is your customer care number?",
        actual_output=get_response("customer_care"),
        context=[                                        # ✅ Added
            "Customer care number is 1800-123-4567",
            "Support available Monday to Saturday 9AM to 6PM"
        ]
    )
=======
import pytest
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.models import DeepEvalBaseLLM

# Copy your OllamaModel class from test_medical_chatbot.py
# Then write 3 test functions below it

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
        "order_status": "Your order #12345 is currently out for delivery and will arrive by 6 PM today.",
        "order_return": "Yes, you can return this product within 30 days of delivery.",
        "customer_care": "Our customer care number is 1800-123-4567 and we also offer support via WhatsApp at 9876543210"
    }
    return responses.get(keyword, "I don't know")

def test_answer_relevancy():
    metric = AnswerRelevancyMetric(
        threshold=0.6,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="Where is my order #12345?",
        actual_output=get_response("order_status")
    )
    assert_test(test_case, [metric])


def test_faithfulness():
    metric = FaithfulnessMetric(
        threshold=0.6,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="Can I return this product?",
        actual_output=get_response("order_return"),
        retrieval_context=[                              # ✅ Added
            "Our return policy allows returns within 30 days of delivery",
            "Products must be unused and in original packaging",
            "Refunds are processed within 5-7 business days"
        ]
    )
    assert_test(test_case, [metric])

def test_hallucination():
    metric = HallucinationMetric(
        threshold=0.2,
        model=ollama_model
    )
    test_case = LLMTestCase(
        input="What is your customer care number?",
        actual_output=get_response("customer_care"),
        context=[                                        # ✅ Added
            "Customer care number is 1800-123-4567",
            "Support available Monday to Saturday 9AM to 6PM"
        ]
    )
>>>>>>> fe85f3779a09903a807811f8f0b1b9b5aa78bde7
    assert_test(test_case, [metric])