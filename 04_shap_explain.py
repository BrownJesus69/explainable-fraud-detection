import shap
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="shap-explainability")

X_val = pd.read_parquet("data/X_val.parquet")
with open("models/xgb_baseline.pkl", "rb") as f:
    model = pickle.load(f)

explainer   = shap.TreeExplainer(model)
sample      = X_val.sample(n=5000, random_state=42)
shap_values = explainer.shap_values(sample)

shap.summary_plot(shap_values, sample, show=False)
plt.savefig("shap_outputs/global_importance.png", bbox_inches="tight", dpi=150)
plt.close()

shap.force_plot(
    explainer.expected_value, shap_values[0],
    sample.iloc[0], matplotlib=True, show=False
)
plt.savefig("shap_outputs/local_force_plot.png", bbox_inches="tight", dpi=150)
plt.close()

wandb.log({
    "shap_global_importance": wandb.Image("shap_outputs/global_importance.png"),
    "shap_local_force_plot":  wandb.Image("shap_outputs/local_force_plot.png")
})

def audit_trail(shap_row, feat_row, n=5):
    pairs = sorted(zip(shap_row, feat_row.index), reverse=True)[:n]
    return [f"{feat}: SHAP={val:+.3f} (raw={feat_row[feat]:.2f})" for val, feat in pairs]

print("Audit trail — transaction 0:")
for line in audit_trail(shap_values[0], sample.iloc[0]):
    print(" •", line)

wandb.finish()
print("Saved: shap_outputs/")
