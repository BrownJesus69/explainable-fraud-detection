import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet("data/merged_raw.parquet")

df["hour"]    = (df["TransactionDT"] / 3600).astype(int) % 24
df["day"]     = (df["TransactionDT"] / 86400).astype(int) % 7
df["amt_log"] = np.log1p(df["TransactionAmt"])

n_before  = df.shape[1]
df        = df[df.columns[df.isnull().mean() < 0.8]]
n_dropped = n_before - df.shape[1]
print(f"Dropped {n_dropped} columns with >80% null values")

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude="number").columns
df[num_cols] = df[num_cols].fillna(-999)
df[cat_cols] = df[cat_cols].fillna("missing")

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
print(f"Label-encoded {len(cat_cols)} categorical columns")

X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
y = df["isFraud"]

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

for df_ in [X_tr, X_val, X_te]: df_.reset_index(drop=True, inplace=True)
for s_ in [y_tr, y_val, y_te]:  s_.reset_index(drop=True, inplace=True)

X_tr.to_parquet("data/X_train.parquet");  y_tr.to_csv("data/y_train.csv", index=False)
X_val.to_parquet("data/X_val.parquet");   y_val.to_csv("data/y_val.csv",  index=False)
X_te.to_parquet("data/X_test.parquet");   y_te.to_csv("data/y_test.csv",  index=False)

print(f"Train {X_tr.shape} | Val {X_val.shape} | Test {X_te.shape}")
print(f"Fraud rates — Train:{y_tr.mean():.4f} Val:{y_val.mean():.4f} Test:{y_te.mean():.4f}")
