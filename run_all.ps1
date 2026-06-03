.\venv\Scripts\Activate.ps1
python 01_ingest.py
python 02_features.py
python 03_train.py
python 04_shap_explain.py
python 05_adversarial.py
python 06_robustness_eval.py
python 07_retrain_robust.py
python 08_drift_check.py
Write-Host "Pipeline complete. Launch API with: cd api && uvicorn main:app --reload --port 8000"
