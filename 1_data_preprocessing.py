"""
1_data_preprocessing.py
────────────────────────
Step 1 of the pipeline: load raw CSV, clean, engineer features,
scale numeric columns, and produce train/test splits saved to disk.

Run:
    python 1_data_preprocessing.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load creditcard.csv and perform basic sanity checks."""
    if not os.path.exists(path):
        print(f"\n[ERROR] Dataset not found at: {path}")
        print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Place creditcard.csv inside the data/ folder.\n")
        sys.exit(1)

    print(f"Loading data from {path} …")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inspect & clean
# ─────────────────────────────────────────────────────────────────────────────

def inspect(df: pd.DataFrame) -> None:
    print("\n── Class distribution ──")
    counts = df[config.TARGET_COLUMN].value_counts()
    for cls, cnt in counts.items():
        label = "Legitimate" if cls == 0 else "Fraudulent"
        pct = cnt / len(df) * 100
        print(f"  {label} ({cls}): {cnt:,}  ({pct:.3f}%)")

    print("\n── Missing values ──")
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("  No missing values found.")
    else:
        print(nulls[nulls > 0])

    print("\n── Duplicate rows ──")
    dupes = df.duplicated().sum()
    print(f"  {dupes:,} duplicate rows found.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and any configured columns."""
    initial = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {initial - len(df):,} duplicate rows.")

    if config.DROP_COLUMNS:
        df = df.drop(columns=config.DROP_COLUMNS, errors="ignore")
        print(f"Dropped columns: {config.DROP_COLUMNS}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that capture fraud patterns.
    V1–V28 are already PCA-transformed; we engineer on Amount and Time.
    """
    print("\n── Feature engineering ──")

    # Time-based features
    df["Hour"]          = (df["Time"] // 3600) % 24
    df["IsNightHour"]   = df["Hour"].between(1, 4).astype(int)
    df["DayOfWeek"]     = ((df["Time"] // 86400) % 7).astype(int)

    # Amount-based features
    df["LogAmount"]     = np.log1p(df["Amount"])
    df["AmountBin"]     = pd.cut(
        df["Amount"],
        bins=[0, 10, 50, 100, 500, 1000, 5000, np.inf],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).fillna(0).astype(int)
    
    # High-value flag (fraud often targets specific ranges)
    df["IsHighAmount"]  = (df["Amount"] > 1000).astype(int)
    df["IsLowAmount"]   = (df["Amount"] < 1).astype(int)

    # Interaction features between top PCA components and Amount
    for v in ["V1", "V3", "V4", "V12", "V14", "V17"]:
        if v in df.columns:
            df[f"{v}_x_Amount"] = df[v] * df["LogAmount"]

    new_cols = [c for c in df.columns if c not in [
        "Time", "Amount", config.TARGET_COLUMN,
        *[f"V{i}" for i in range(1, 29)]
    ]]
    print(f"  New features added: {new_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scale numeric columns
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on train set only to prevent data leakage,
    then transform both train and test.
    """
    scale_cols = [c for c in config.SCALE_COLUMNS if c in X_train.columns]
    # Also scale the engineered numeric columns
    extra = ["LogAmount", "Hour", "DayOfWeek"]
    scale_cols += [c for c in extra if c in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    print(f"\n  Scaled columns: {scale_cols}")
    return X_train, X_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 5. Split & save
# ─────────────────────────────────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> None:
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.TEST_SIZE,
        stratify     = y,               # preserve class ratio in both splits
        random_state = config.RANDOM_STATE,
    )

    X_train, X_test, scaler = scale_features(X_train.copy(), X_test.copy())

    # Reassemble and save
    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train.values
    test_df  = X_test.copy()
    test_df[config.TARGET_COLUMN]  = y_test.values

    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    test_df.to_csv(config.TEST_PATH,   index=False)

    # Save scaler for inference
    import joblib
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "scaler.pkl"))

    print(f"\n── Split summary ──")
    print(f"  Train: {len(train_df):,} rows  "
          f"(fraud: {y_train.sum():,} / {y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Test:  {len(test_df):,} rows   "
          f"(fraud: {y_test.sum():,} / {y_test.sum()/len(y_test)*100:.2f}%)")
    print(f"\n  Saved → {config.TRAIN_PATH}")
    print(f"  Saved → {config.TEST_PATH}")
    print(f"  Saved → models/scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 1 — Data Preprocessing")
    print("=" * 60)

    df = load_data(config.RAW_DATA_PATH)
    inspect(df)
    df = clean(df)
    df = engineer_features(df)

    # Save processed full dataset
    df.to_csv(config.PROCESSED_PATH, index=False)
    print(f"\n  Processed data saved → {config.PROCESSED_PATH}")

    split_and_save(df)
    print("\n✓ Preprocessing complete. Run: python 2_eda.py\n")


if __name__ == "__main__":
    main()