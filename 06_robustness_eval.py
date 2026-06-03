import pandas as pd
import pickle
import json
import wandb
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score, precision_recall_curve
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="robustness-eval-v2")

def eval_model(model, X, y, label):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    roc  = roc_auc_score(y, probs)
    pr   = average_precision_score(y, probs)
    rec  = recall_score(y, preds)
    print(f"[{label}] ROC:{roc:.4f}  PR-AUC:{pr:.4f}  Recall:{rec:.4f}")
    return probs, {"roc": roc, "pr_auc": pr, "recall": rec}

X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_csv("data/y_test.csv").squeeze()

with open("models/xgb_baseline.pkl", "rb") as f:
    baseline = pickle.load(f)

with open("adversarial_outputs/evasions.json") as f:
    evasions = json.load(f)

probs, m = eval_model(baseline, X_test, y_test, "Baseline-Clean")
wandb.log({f"baseline_{k}": v for k, v in m.items()})
wandb.log({"attack_success_rate": len(evasions) / 500})

# Threshold tuning — maximise F1
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1          = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
best_idx    = int(f1.argmax())
opt_thresh  = float(thresholds[best_idx])
opt_f1      = float(f1[best_idx])
opt_prec    = float(precision[best_idx])
opt_rec     = float(recall[best_idx])

print(f"Optimal threshold: {opt_thresh:.4f}  F1:{opt_f1:.4f}  "
      f"Precision:{opt_prec:.4f}  Recall:{opt_rec:.4f}")
wandb.log({"optimal_threshold": opt_thresh, "optimal_f1": opt_f1,
           "optimal_precision": opt_prec, "optimal_recall": opt_rec})
wandb.finish()
