"""
3_train_models.py
─────────────────
Step 3: Train Logistic Regression, Random Forest, XGBoost, and Neural Network.

Pipeline:
  1. Load train split
  2. Apply SMOTE to balance classes (on train only — no leakage)
  3. Train each model with cross-validation
  4. Save trained models to models/

Run:
    python 3_train_models.py
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline       import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

import config

os.makedirs(config.MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_train() -> tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(config.TRAIN_PATH):
        print("[ERROR] Run 1_data_preprocessing.py first.")
        raise SystemExit(1)
    df = pd.read_csv(config.TRAIN_PATH)
    X  = df.drop(columns=[config.TARGET_COLUMN])
    y  = df[config.TARGET_COLUMN]
    print(f"  Train set: {len(X):,} rows, {X.shape[1]} features")
    print(f"  Fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y


def apply_smote(X: pd.DataFrame, y: pd.Series):
    """
    Apply SMOTE to oversample the minority fraud class.
    Falls back to class_weight='balanced' if imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(
            sampling_strategy = config.SMOTE_STRATEGY,
            k_neighbors       = config.SMOTE_K,
            random_state      = config.RANDOM_STATE,
        )
        X_res, y_res = sm.fit_resample(X, y)
        print(f"\n  SMOTE applied:")
        print(f"    Before → Legit: {(y==0).sum():,}  Fraud: {(y==1).sum():,}")
        print(f"    After  → Legit: {(y_res==0).sum():,}  Fraud: {(y_res==1).sum():,}")
        return X_res, y_res
    except ImportError:
        print("\n  [INFO] imbalanced-learn not installed — skipping SMOTE.")
        print("         Install with: pip install imbalanced-learn")
        print("         Using class_weight='balanced' in classifiers instead.\n")
        return X.values, y.values


def cv_score(model, X, y, label: str) -> float:
    """Quick stratified 3-fold CV on F1 (faster than full 5-fold for large sets)."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    print(f"    CV F1: {mean:.4f} ± {std:.4f}")
    return mean


def save_model(model, name: str) -> None:
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("\n[1/4] Logistic Regression")
    t0 = time.time()
    model = LogisticRegression(**config.LR_PARAMS)
    cv_score(model, X_train, y_train, "LR")
    model.fit(X_train, y_train)
    print(f"    Train time: {time.time()-t0:.1f}s")
    save_model(model, "logistic_regression")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\n[2/4] Random Forest")
    t0 = time.time()
    model = RandomForestClassifier(**config.RF_PARAMS)
    cv_score(model, X_train, y_train, "RF")
    model.fit(X_train, y_train)
    print(f"    Train time: {time.time()-t0:.1f}s")
    save_model(model, "random_forest")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    print("\n[3/4] XGBoost")
    try:
        from xgboost import XGBClassifier
        t0 = time.time()
        params = {k: v for k, v in config.XGB_PARAMS.items()
                  if k != "use_label_encoder"}
        model = XGBClassifier(**params, verbosity=0)
        cv_score(model, X_train, y_train, "XGB")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        print(f"    Train time: {time.time()-t0:.1f}s")
        save_model(model, "xgboost")
        return model
    except ImportError:
        print("    [SKIP] XGBoost not installed. Install: pip install xgboost")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Model 4: Neural Network (sklearn MLP or Keras)
# ─────────────────────────────────────────────────────────────────────────────

def train_neural_network(X_train, y_train):
    print("\n[4/4] Neural Network")
    t0 = time.time()

    try:
        # Prefer Keras / TensorFlow for deeper architecture
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(config.RANDOM_STATE)
        layers_cfg = config.NN_PARAMS["hidden_layers"]
        dropout    = config.NN_PARAMS["dropout_rate"]
        lr         = config.NN_PARAMS["learning_rate"]

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(X_train.shape[1],)))
        for units in layers_cfg:
            model.add(keras.layers.Dense(units, activation="relu"))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer = keras.optimizers.Adam(lr),
            loss      = "binary_crossentropy",
            metrics   = ["AUC"],
        )

        # Class weights instead of SMOTE for Keras
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        class_weight = {0: 1.0, 1: neg / pos}

        model.fit(
            X_train, y_train,
            epochs        = config.NN_PARAMS["epochs"],
            batch_size    = config.NN_PARAMS["batch_size"],
            class_weight  = class_weight,
            validation_split = 0.1,
            verbose       = 1,
        )
        model.save(os.path.join(config.MODELS_DIR, "neural_network.keras"))
        print(f"    Train time: {time.time()-t0:.1f}s")
        print(f"    Saved → models/neural_network.keras")
        return model

    except ImportError:
        # Fall back to sklearn's MLPClassifier
        print("    [INFO] TensorFlow not installed — using sklearn MLPClassifier")
        layers = tuple(config.NN_PARAMS["hidden_layers"])
        model = MLPClassifier(
            hidden_layer_sizes = layers,
            activation         = "relu",
            solver             = "adam",
            alpha              = 1e-4,
            max_iter           = 200,
            early_stopping     = True,
            validation_fraction= 0.1,
            random_state       = config.RANDOM_STATE,
            verbose            = False,
        )
        cv_score(model, X_train, y_train, "MLP")
        model.fit(X_train, y_train)
        print(f"    Train time: {time.time()-t0:.1f}s")
        save_model(model, "neural_network")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance report
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_importance(rf_model, feature_names, top_n=10) -> None:
    if rf_model is None:
        return
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)
    print("\n── Top feature importances (Random Forest) ──")
    for feat, imp in top.items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<18} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 3 — Model Training")
    print("=" * 60)

    X, y = load_train()
    X_sm, y_sm = apply_smote(X, y)

    lr_model  = train_logistic_regression(X_sm, y_sm)
    rf_model  = train_random_forest(X_sm, y_sm)
    xgb_model = train_xgboost(X_sm, y_sm)
    nn_model  = train_neural_network(X_sm, y_sm)

    print_feature_importance(rf_model, X.columns.tolist())

    # Save feature column order for inference
    joblib.dump(X.columns.tolist(), os.path.join(config.MODELS_DIR, "feature_names.pkl"))

    print("\n✓ All models trained and saved.")
    print("  Run: python 4_evaluate_models.py\n")


if __name__ == "__main__":
    main()