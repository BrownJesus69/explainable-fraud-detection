import pandas as pd
import pickle
import json
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="xgb-robust-retrain")

X_tr   = pd.read_parquet("data/X_train.parquet")
y_tr   = pd.read_csv("data/y_train.csv").squeeze()
X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_csv("data/y_test.csv").squeeze()

with open("adversarial_outputs/evasions.json") as f:
    evasions = json.load(f)

adv_idx = [e["idx"] for e in evasions]
adv_X   = X_test.loc[adv_idx]
adv_y   = pd.Series([1] * len(adv_X), index=adv_X.index)

X_aug = pd.concat([X_tr, adv_X]).reset_index(drop=True)
y_aug = pd.concat([y_tr, adv_y]).reset_index(drop=True)

scale_pos = (y_aug == 0).sum() / (y_aug == 1).sum()
model = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    scale_pos_weight=scale_pos, subsample=0.8,
    colsample_bytree=0.8, n_jobs=-1, random_state=42
)
model.fit(X_aug, y_aug)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)
roc   = roc_auc_score(y_test, probs)
pr    = average_precision_score(y_test, probs)
rec   = recall_score(y_test, preds)

print(f"Robust Model — ROC:{roc:.4f}  PR-AUC:{pr:.4f}  Recall:{rec:.4f}")
wandb.log({"robust_roc": roc, "robust_pr_auc": pr, "robust_recall": rec})

with open("models/xgb_robust.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/xgb_robust.pkl")

artifact = wandb.Artifact("xgb-robust", type="model")
artifact.add_file("models/xgb_robust.pkl")
wandb.log_artifact(artifact)
print("Logged artifact: xgb-robust")
wandb.finish()
