import pandas as pd
import pickle
import wandb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="lgbm-baseline", config={
    "model": "LightGBM",
    "n_estimators": 500, "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8
})

X_tr  = pd.read_parquet("data/X_train.parquet")
y_tr  = pd.read_csv("data/y_train.csv").squeeze()
X_val = pd.read_parquet("data/X_val.parquet")
y_val = pd.read_csv("data/y_val.csv").squeeze()
X_te  = pd.read_parquet("data/X_test.parquet")
y_te  = pd.read_csv("data/y_test.csv").squeeze()

scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()

model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.05,
    scale_pos_weight=scale_pos, subsample=0.8,
    colsample_bytree=0.8, n_jobs=-1, random_state=42,
    metric="average_precision",   # early stopping on PR-AUC, matching XGBoost
    verbose=-1
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=True), lgb.log_evaluation(50)]
)

# Val metrics
val_probs = model.predict_proba(X_val)[:, 1]
val_roc   = roc_auc_score(y_val, val_probs)
val_pr    = average_precision_score(y_val, val_probs)
print(f"[Val]  ROC-AUC: {val_roc:.4f}  PR-AUC: {val_pr:.4f}")

# Test metrics
te_probs = model.predict_proba(X_te)[:, 1]
te_roc   = roc_auc_score(y_te, te_probs)
te_pr    = average_precision_score(y_te, te_probs)
print(f"[Test] ROC-AUC: {te_roc:.4f}  PR-AUC: {te_pr:.4f}")
print(classification_report(y_te, (te_probs > 0.5).astype(int)))

wandb.log({
    "val_roc_auc":  val_roc,  "val_pr_auc":  val_pr,
    "test_roc_auc": te_roc,   "test_pr_auc": te_pr
})
wandb.finish()

with open("models/lgbm_baseline.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/lgbm_baseline.pkl")
