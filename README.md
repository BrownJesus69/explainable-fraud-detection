# Explainable Fraud Detection + Adversarial Robustness

![Python](https://img.shields.io/badge/Python-3.14-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange) ![Docker](https://img.shields.io/badge/Docker-ready-blue) ![W&B](https://img.shields.io/badge/W%26B-tracked-yellow)

## Abstract

> Dataset: 590,540 transactions, 3.5% fraud rate (IEEE-CIS)
> Features: 360 engineered features after null-column pruning
> Model: XGBoost (500 trees, depth 7, scale_pos_weight=27.57)
> ROC-AUC: 0.957 (vs LightGBM 0.944)
> PR-AUC: 0.742 (vs LightGBM 0.684, +5.8pp advantage)
> Optimal F1: 0.696 at threshold 0.828 (Precision 0.797, Recall 0.617)
> Robustness: 3.01% adversarial evasion rate (Nelder-Mead, 20 features, n=2562)
> Drift: 2.2% feature drift (8/360 columns, Evidently DataDriftPreset)
> API latency: <100ms per prediction with SHAP audit trail

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            IEEE-CIS Fraud Detection Dataset                 │
│         590k transactions · 434 raw features · 3.5% fraud  │
└──────────────────────────┬──────────────────────────────────┘
                           │
               ┌───────────▼────────────┐
               │     01_ingest.py       │  merge CSVs → parquet
               └───────────┬────────────┘
                           │
               ┌───────────▼────────────┐
               │    02_features.py      │  360 features · 70/15/15 split
               └───────────┬────────────┘
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
 ┌────────▼────────┐  ┌────▼────────┐  ┌────▼────────────┐
 │  03_train.py    │  │09_lgbm_     │  │ 04_shap_        │
 │  XGBoost        │  │compare.py   │  │ explain.py      │
 │  ROC  0.957     │  │LightGBM     │  │ Waterfall plots │
 │  PR   0.742  ◄──┼──│ROC  0.944   │  │ Audit trail     │
 └────────┬────────┘  │PR   0.684   │  └─────────────────┘
          │           └─────────────┘
 ┌────────▼────────┐
 │ 05_adversarial  │  Nelder-Mead · 20 SHAP features · 3.01% evasion
 └────────┬────────┘
          │
 ┌────────▼────────┐
 │ 06_robustness   │  Threshold tuning → optimal 0.828
 └────────┬────────┘
          │
 ┌────────▼────────┐
 │ 07_retrain_     │  Adversarial augmentation → xgb-robust:v1
 │ robust.py       │  W&B Artifact Registry
 └────────┬────────┘
          │
 ┌────────▼────────┐
 │ 08_drift_check  │  Evidently · 8/360 columns drifted (2.2%)
 └────────┬────────┘
          │
 ┌────────▼──────────────┐     ┌──────────────────────────────┐
 │   api/main.py         │◄────│   streamlit_app.py           │
 │   FastAPI · port 8000 │     │   Gauge · Waterfall · Drift  │
 │   xgb-robust:v1       │     │   port 8501                  │
 └───────────────────────┘     └──────────────────────────────┘
```

---

## Results

| Metric | XGBoost | LightGBM | XGB Advantage |
|--------|---------|----------|---------------|
| ROC-AUC (test) | **0.9568** | 0.9440 | +0.013 |
| **PR-AUC (test)** | **0.7422** | 0.6841 | **+0.058** |
| Recall @ 0.5 | 0.827 | 0.830 | ≈ tied |
| Precision @ 0.5 | **0.350** | 0.270 | +0.080 |
| Optimal threshold | **0.828** | — | — |
| F1 @ optimal | **0.696** | — | — |

> **Why PR-AUC?** ROC-AUC flatters all classifiers on skewed data. With 3.5% fraud, a model that flags everything scores 0.50 ROC-AUC but 0.035 PR-AUC. PR-AUC is the honest metric.

---

## Adversarial Robustness

Under a Nelder-Mead optimizer targeting 20 SHAP-dominant continuous features, **3.01% of confirmed fraud transactions** (77/2562) could be perturbed below the 0.5 detection threshold. Both low-confidence evasions (original probability 0.52–0.56) were borderline detections — high-confidence fraud was not evaded.

---

## Quickstart — Docker

```bash
# Build
docker build -t fraud-api -f api/Dockerfile .

# Run
docker run -d -p 8000:8000 fraud-api

# Health check
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": {"TransactionAmt": 500, "hour": 2, "C13": 5, "M5": 1}}'
```

**Response:**
```json
{
  "fraud_probability": 0.3461,
  "decision": "PASS",
  "top_shap_factors": ["C14: +0.692", "id_31: +0.293", "card6: +0.179", "V70: +0.163", "C13: +0.141"]
}
```

Decision boundary is at **0.828** (F1-optimal threshold, not 0.5).

---

## Quickstart — Streamlit Dashboard

```bash
pip install streamlit plotly
streamlit run streamlit_app.py
```

Open `http://localhost:8501` — interactive fraud gauge, SHAP waterfall, drift monitor.

---

## W&B Dashboard

All experiments tracked at:
**https://wandb.ai/adityabidappam-dayananda-sagar-university/fraud-detection**

| Run | Key Metrics |
|-----|-------------|
| `xgb-baseline` | ROC 0.952, PR-AUC 0.721 |
| `lgbm-baseline` | ROC 0.944, PR-AUC 0.684 |
| `shap-explainability` | beeswarm + waterfall plots |
| `adversarial-attack-v3-full` | evasion rate 3.01%, n=2562 |
| `robustness-eval-v2` | optimal threshold 0.828 |
| `xgb-robust-retrain` | artifact `xgb-robust:v1` |
| `evidently-drift` | 8/360 columns drifted |

---

## Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/BrownJesus69/explainable-fraud-detection
cd explainable-fraud-detection
python -m venv venv && venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add credentials to .env
# WANDB_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY, EVIDENTLY_API_KEY, EVIDENTLY_PROJECT_ID

# 3. Download data
kaggle competitions download -c ieee-fraud-detection -p data
Expand-Archive data\ieee-fraud-detection.zip -DestinationPath data

# 4. Run full pipeline
.\run_all.ps1

# 5. Launch API
cd api && uvicorn main:app --reload --port 8000

# 6. Launch dashboard
streamlit run streamlit_app.py
```

---

## Project Structure

```
├── 01_ingest.py          # CSV merge → parquet
├── 02_features.py        # Feature engineering, stratified split
├── 03_train.py           # XGBoost baseline + W&B logging
├── 04_shap_explain.py    # SHAP beeswarm + waterfall plots
├── 05_adversarial.py     # Nelder-Mead adversarial attack
├── 06_robustness_eval.py # Robustness eval + threshold tuning
├── 07_retrain_robust.py  # Adversarial augmentation + W&B artifact
├── 08_drift_check.py     # Evidently drift report
├── 09_lgbm_compare.py    # LightGBM comparison model
├── streamlit_app.py      # Interactive dashboard
├── api/
│   ├── main.py           # FastAPI serving xgb-robust:v1
│   ├── Dockerfile        # python:3.12-slim
│   └── requirements.txt  # Slim API deps
└── run_all.ps1           # One-command pipeline runner
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | XGBoost 3.2, LightGBM 4.6, scikit-learn 1.9 |
| Explainability | SHAP 0.52 (TreeExplainer, waterfall plots) |
| Adversarial | SciPy Nelder-Mead optimizer |
| Experiment tracking | Weights & Biases 0.27 |
| Drift monitoring | Evidently 0.7 |
| Serving | FastAPI 0.136, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Containerisation | Docker (python:3.12-slim) |

