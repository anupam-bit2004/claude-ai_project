import json
import pytest
import requests
import mlflow
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM

# ---- Ollama Model (same as before) ----
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name="mistral"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=300
        )
        return response.json()["response"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name

# ---- Setup ----
ollama_model = OllamaModel(model_name="mistral")
mlflow.set_experiment("Dell-Medical-Chatbot-Eval-V2")

# ---- Load golden dataset ----
with open("golden_dataset.json") as f:
    GOLDEN_DATASET = json.load(f)

# ---- Reusable fixture ----
@pytest.fixture
def log_result():
    def metric_logging(runval, test_case, metric):
        with mlflow.start_run(run_name=runval):
            try:
                metric.measure(test_case)
                score = metric.score
                mlflow.log_metric("answer_relevancy", score if score is not None else 0.0)
                mlflow.log_param("status", "completed" if score is not None else "TIMEOUT_CRITICAL")
                mlflow.log_param("domain", "medical")
                mlflow.log_param("model", "mistral")
                if score is not None and score < metric.threshold:
                    raise AssertionError(
                        f"Score {score:.2f} below threshold {metric.threshold}"
                    )
            except AssertionError as e:
                print(f"Test FAILED: {e}")
                raise
            except Exception as e:
                print(f"Test ERROR: {type(e).__name__}: {e}")
                mlflow.log_metric("answer_relevancy", 0.0)
                mlflow.log_param("status", "TIMEOUT_CRITICAL")
                raise
    return metric_logging

# ---- Generate tests dynamically from golden dataset ----
@pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c["id"] for c in GOLDEN_DATASET])
def test_medical_chatbot(case, log_result):
    metric = AnswerRelevancyMetric(
        threshold=case["threshold"],
        model=ollama_model
    )
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=case["actual_output"]
    )
    log_result(case["run_name"], test_case, metric)