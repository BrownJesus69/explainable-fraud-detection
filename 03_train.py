import pandas as pd
import pickle
import wandb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="xgb-baseline", config={
    "n_estimators": 500, "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8
})

X_tr  = pd.read_parquet("data/X_train.parquet")
y_tr  = pd.read_csv("data/y_train.csv").squeeze()
X_val = pd.read_parquet("data/X_val.parquet")
y_val = pd.read_csv("data/y_val.csv").squeeze()

scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

model = XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    scale_pos_weight=scale_pos, subsample=0.8,
    colsample_bytree=0.8, n_jobs=-1, random_state=42,
    eval_metric="aucpr", early_stopping_rounds=30
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)

probs = model.predict_proba(X_val)[:, 1]
roc   = roc_auc_score(y_val, probs)
pr    = average_precision_score(y_val, probs)

print(f"ROC-AUC : {roc:.4f}")
print(f"PR-AUC  : {pr:.4f}")
print(classification_report(y_val, (probs > 0.5).astype(int)))

wandb.log({"val_roc_auc": roc, "val_pr_auc": pr})
wandb.finish()

with open("models/xgb_baseline.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/xgb_baseline.pkl")
