import pandas as pd
import wandb
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="evidently-drift")

ref  = pd.read_parquet("data/X_train.parquet").sample(5000, random_state=42)
curr = pd.read_parquet("data/X_test.parquet").sample(5000, random_state=42)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=curr)
report.save_html("monitoring/drift_report.html")

result    = report.as_dict()
n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
n_total   = result["metrics"][0]["result"]["number_of_columns"]

print(f"Drifted columns: {n_drifted} / {n_total}")
wandb.log({"drifted_columns": n_drifted, "total_columns": n_total})
wandb.finish()
print("Saved: monitoring/drift_report.html")
