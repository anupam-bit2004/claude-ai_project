import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# ---- Reference Data (Training 2023) ----
np.random.seed(42)
reference_df = pd.DataFrame({
    "query_length": np.random.normal(50, 10, 1000),
    "response_time": np.random.normal(2.5, 0.5, 1000),
    "faithfulness_score": np.random.normal(0.85, 0.05, 1000),
    "user_satisfaction": np.random.normal(4.2, 0.3, 1000)
})

# ---- Current Data (Production 2026 — drifted!) ----
current_df = pd.DataFrame({
    "query_length": np.random.normal(80, 15, 500),
    "response_time": np.random.normal(4.0, 1.0, 500),
    "faithfulness_score": np.random.normal(0.70, 0.08, 500),
    "user_satisfaction": np.random.normal(3.5, 0.5, 500)
})

# ---- Run Report ----
report = Report([DataDriftPreset()])
my_eval = report.run(reference_df, current_df)

# ---- Save HTML ----
my_eval.save_html("drift_report.html")
print("✅ Done! Open drift_report.html to see results")