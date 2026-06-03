import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
train_txn = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
train_id  = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
df = train_txn.merge(train_id, on="TransactionID", how="left")

assert "isFraud" in df.columns
assert df.duplicated(subset="TransactionID").sum() == 0, "Duplicate TransactionIDs found"

print(f"Shape       : {df.shape}")
print(f"Fraud rate  : {df['isFraud'].mean():.4f}")
print(f"Top nulls   :\n{df.isnull().sum().sort_values(ascending=False).head(10)}")

df.to_parquet(f"{DATA_DIR}/merged_raw.parquet", index=False)
print("Saved: data/merged_raw.parquet")
