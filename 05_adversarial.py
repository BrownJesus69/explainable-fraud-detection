import pandas as pd
import numpy as np
import pickle
import json
import wandb
from scipy.optimize import minimize
from dotenv import load_dotenv

load_dotenv()

wandb.init(project="fraud-detection", name="adversarial-attack-v3-full")

with open("models/xgb_baseline.pkl", "rb") as f:
    model = pickle.load(f)

X_test = pd.read_parquet("data/X_test.parquet")
y_test = pd.read_csv("data/y_test.csv").squeeze()

fraud_X   = X_test[y_test == 1]   # full fraud set — no cap
THRESHOLD = 0.5

PERTURBABLE = [c for c in [
    "TransactionAmt", "amt_log", "hour", "day",
    "C1", "C2", "C6", "C11", "C13", "C14",
    "V70", "V83", "V87", "V130", "V131", "V258", "V307", "V308", "V310", "V317"
] if c in fraud_X.columns]

print(f"Fraud samples: {len(fraud_X)} | Perturbable features: {len(PERTURBABLE)}")

evasions  = []
n_attacked = 0

for idx, row in fraud_X.iterrows():
    base_row  = row.copy()
    orig_prob = model.predict_proba(base_row.values.reshape(1, -1))[0][1]
    if orig_prob < THRESHOLD:
        continue
    n_attacked += 1

    x0 = base_row[PERTURBABLE].values.astype(float)

    def evasion_objective(x):
        r              = base_row.copy()
        r[PERTURBABLE] = x
        return model.predict_proba(r.values.reshape(1, -1))[0][1]

    result   = minimize(evasion_objective, x0=x0, method="Nelder-Mead",
                        options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-3})
    new_prob = result.fun

    if new_prob < THRESHOLD:
        evasions.append({
            "idx": int(idx), "orig_prob": round(float(orig_prob), 4),
            "adv_prob": round(float(new_prob), 4), "n_iters": result.nit
        })

    if n_attacked % 100 == 0:
        print(f"  Attacked {n_attacked} | Evaded {len(evasions)}")

rate = len(evasions) / n_attacked if n_attacked else 0
print(f"Attacks: {n_attacked} | Evasions: {len(evasions)} | Rate: {rate:.2%}")

wandb.log({"attack_success_rate": rate, "total_evasions": len(evasions),
           "n_perturbable_features": len(PERTURBABLE)})
wandb.finish()

with open("adversarial_outputs/evasions.json", "w") as f:
    json.dump(evasions, f, indent=2)
print("Saved: adversarial_outputs/evasions.json")
