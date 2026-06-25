import mlflow

# ---- Create/Set Experiment ----
mlflow.set_experiment("Dell-Medical-Chatbot-Eval")

# ---- Run 1: Mistral Model ----
with mlflow.start_run(run_name="mistral-run-1"):
    mlflow.log_param("model", "mistral")
    mlflow.log_param("threshold", 0.7)
    mlflow.log_metric("faithfulness", 0.85)
    mlflow.log_metric("answer_relevancy", 0.90)
    mlflow.log_metric("hallucination", 0.08)

# ---- Run 2: Simulating a worse model version ----
with mlflow.start_run(run_name="mistral-run-2-degraded"):
    mlflow.log_param("model", "mistral")
    mlflow.log_param("threshold", 0.7)
    mlflow.log_metric("faithfulness", 0.65)  # dropped!
    mlflow.log_metric("answer_relevancy", 0.72)  # dropped!
    mlflow.log_metric("hallucination", 0.25)  # increased!

print("✅ Done! Run 'mlflow ui' to view results")

