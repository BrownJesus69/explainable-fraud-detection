import os
import pickle
from typing import Any, Dict

import shap
import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = os.environ.get("MODEL_PATH", "../models/xgb_robust.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
explainer = shap.TreeExplainer(model)
FEATURES  = model.get_booster().feature_names

THRESHOLD = 0.828

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(features: Dict[str, Any]):
    row       = pd.DataFrame([features]).reindex(columns=FEATURES, fill_value=-999)
    prob      = float(model.predict_proba(row)[0][1])
    shap_vals = explainer.shap_values(row)[0]
    top5      = sorted(zip(shap_vals, FEATURES), reverse=True)[:5]
    audit     = [f"{feat}: {val:+.3f}" for val, feat in top5]
    return {
        "fraud_probability": round(prob, 4),
        "decision":          "REVIEW" if prob > THRESHOLD else "PASS",
        "top_shap_factors":  audit
    }
