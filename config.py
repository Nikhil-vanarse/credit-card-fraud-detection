"""
config.py — Central configuration for the Fraud Detection pipeline.
Edit paths and hyperparameters here; all other scripts import from this file.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

RAW_DATA_PATH    = os.path.join(DATA_DIR,   "creditcard.csv")
PROCESSED_PATH   = os.path.join(DATA_DIR,   "processed.csv")
TRAIN_PATH       = os.path.join(DATA_DIR,   "train.csv")
TEST_PATH        = os.path.join(DATA_DIR,   "test.csv")

# ── Dataset ────────────────────────────────────────────────────────────────
TARGET_COLUMN  = "Class"           # 1 = fraud, 0 = legitimate
TEST_SIZE      = 0.20              # 80/20 train-test split
RANDOM_STATE   = 42

# ── Preprocessing ──────────────────────────────────────────────────────────
SCALE_COLUMNS  = ["Amount", "Time"]   # columns that need StandardScaler
DROP_COLUMNS   = []                   # columns to drop before training

# ── SMOTE ──────────────────────────────────────────────────────────────────
SMOTE_STRATEGY = "minority"        # oversample minority to match majority
SMOTE_K        = 5                 # number of nearest neighbours

# ── Model hyperparameters ──────────────────────────────────────────────────
LR_PARAMS = {
    "max_iter": 1000,
    "C": 0.1,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 577,       # ratio of negatives to positives
    "use_label_encoder": False,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}

NN_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 2048,
}

# ── Evaluation ─────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.30          # lower threshold catches more fraud
CV_FOLDS           = 5             # stratified k-fold cross-validation

# ── Plots ──────────────────────────────────────────────────────────────────
PLOT_DPI    = 150
PLOT_STYLE  = "seaborn-v0_8-whitegrid"
PALETTE     = {"legit": "#378add", "fraud": "#e24b4a"}