"""
5_predict.py
────────────
Predict fraud probability for a single transaction from the command line.
Uses the best saved model (XGBoost → Random Forest → Logistic Regression).

Usage:
    python 5_predict.py
    python 5_predict.py --amount 4500 --hour 3 --location foreign
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

import config


# ─────────────────────────────────────────────────────────────────────────────
# Load model + scaler
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model():
    """Load the highest-priority available model."""
    priority = ["xgboost", "random_forest", "neural_network", "logistic_regression"]
    for name in priority:
        path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"  Using model: {name.replace('_', ' ').title()}")
            return model, name
    print("[ERROR] No trained models found. Run 3_train_models.py first.")
    sys.exit(1)


def load_scaler():
    path = os.path.join(config.MODELS_DIR, "scaler.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_feature_names() -> list:
    path = os.path.join(config.MODELS_DIR, "feature_names.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Build feature vector from transaction inputs
# ─────────────────────────────────────────────────────────────────────────────

# V1-V28 typical means by class (approximated from published dataset stats)
# In production these would come from the actual transaction's PCA-transformed values.
LEGIT_V_MEANS = {
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
    "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
    "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
    "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
    "V26": 0.0, "V27": 0.0, "V28": 0.0,
}
# Fraud signature — key components shift significantly
FRAUD_V_SIGNATURE = {
    "V1": -3.0, "V3": -3.3, "V4": 4.0, "V7": -6.0, "V9": -2.5,
    "V10": -4.5, "V11": 2.0, "V12": -5.5, "V14": -7.5, "V16": -3.5,
    "V17": -8.5, "V18": -2.5, "V19": 1.0, "V21": 0.8, "V27": 1.5,
}


def build_feature_vector(
    amount: float,
    hour: int,
    location: str,   # "home" | "near" | "different" | "foreign"
    merchant: str,   # "grocery" | "electronics" | "online" | "atm" | "gas"
    velocity: str,   # "normal" | "high" | "very_high"
    card_present: bool,
    risk_level: str = "auto",  # "auto" | "low" | "medium" | "high"
) -> pd.DataFrame:
    """
    Construct a synthetic feature row that approximates a real transaction.
    For a production system, V1–V28 come from the bank's PCA pipeline.
    """
    # Build V components
    v_vals = dict(LEGIT_V_MEANS)

    # Inject fraud-like signature based on risk factors
    fraud_score = 0.0
    reasons     = []

    if hour in range(1, 5):
        fraud_score += 0.35
        reasons.append(f"Late-night transaction ({hour}:00 AM)")
        for k, v in FRAUD_V_SIGNATURE.items():
            v_vals[k] = v_vals.get(k, 0) + v * 0.4

    if location == "foreign":
        fraud_score += 0.30
        reasons.append("Foreign country (geolocation mismatch)")
        for k, v in FRAUD_V_SIGNATURE.items():
            v_vals[k] = v_vals.get(k, 0) + v * 0.35
    elif location == "different":
        fraud_score += 0.12
        reasons.append("Different state from home location")

    if merchant == "atm":
        fraud_score += 0.20
        reasons.append("ATM withdrawal (high-risk category)")
    elif merchant in ("electronics", "online"):
        fraud_score += 0.10
        reasons.append(f"High-risk merchant: {merchant}")

    if velocity == "very_high":
        fraud_score += 0.20
        reasons.append("Very high transaction velocity (>10/day)")
    elif velocity == "high":
        fraud_score += 0.10
        reasons.append("Elevated transaction velocity (5-10/day)")

    if not card_present:
        fraud_score += 0.15
        reasons.append("Card-not-present (CNP) transaction")

    if amount > 2000:
        fraud_score += 0.12
        reasons.append(f"High transaction amount (${amount:,.2f})")
    elif amount < 1:
        fraud_score += 0.05
        reasons.append("Micro-transaction (<$1) — possible testing")

    # Build row
    row = {**v_vals}
    row["Amount"]        = amount
    row["Time"]          = hour * 3600
    row["Hour"]          = hour
    row["IsNightHour"]   = int(hour in range(1, 5))
    row["DayOfWeek"]     = 3
    row["LogAmount"]     = np.log1p(amount)
    row["AmountBin"]     = min(6, int(np.log1p(amount) * 0.8))
    row["IsHighAmount"]  = int(amount > 1000)
    row["IsLowAmount"]   = int(amount < 1)

    for v in ["V1", "V3", "V4", "V12", "V14", "V17"]:
        row[f"{v}_x_Amount"] = row[v] * row["LogAmount"]

    return pd.DataFrame([row]), fraud_score, reasons


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(model, scaler, feature_names, X: pd.DataFrame) -> float:
    """Align columns, scale, and return fraud probability."""
    if feature_names:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]

    if scaler:
        scale_cols = [c for c in ["Amount", "Time", "LogAmount", "Hour", "DayOfWeek"]
                      if c in X.columns]
        X = X.copy()
        X[scale_cols] = scaler.transform(X[scale_cols])

    try:
        prob = model.predict_proba(X)[0, 1]
    except AttributeError:
        prob = float(model.predict(X, verbose=0)[0][0])
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def display_result(prob: float, reasons: list, threshold: float) -> None:
    bar_len   = 40
    filled    = int(prob * bar_len)
    bar       = "█" * filled + "░" * (bar_len - filled)
    is_fraud  = prob >= threshold

    print("\n" + "=" * 55)
    print("  FRAUD DETECTION RESULT")
    print("=" * 55)
    print(f"\n  Fraud probability:  {prob*100:.1f}%")
    print(f"  [{bar}]")
    print(f"  Decision threshold: {threshold*100:.0f}%")

    if is_fraud:
        print(f"\n  ⛔  DECISION: FRAUD DETECTED — BLOCK TRANSACTION")
        print(f"      Alert sent to fraud investigation team.")
    else:
        print(f"\n  ✅  DECISION: LEGITIMATE — APPROVE TRANSACTION")

    if reasons:
        print(f"\n  Risk factors identified:")
        for r in reasons:
            print(f"    • {r}")
    else:
        print("\n  No significant risk factors detected.")

    print("\n  Risk level: ", end="")
    if prob < 0.2:
        print("LOW RISK")
    elif prob < 0.5:
        print("MEDIUM RISK")
    elif prob < 0.75:
        print("HIGH RISK")
    else:
        print("CRITICAL RISK")

    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def interactive_mode(model, scaler, feature_names):
    print("\n── Interactive Transaction Predictor ──")
    print("Enter transaction details (or Ctrl+C to quit)\n")

    while True:
        try:
            amount       = float(input("  Amount ($): "))
            hour         = int(input("  Hour (0-23): "))
            location     = input("  Location [home/near/different/foreign]: ").strip().lower()
            merchant     = input("  Merchant [grocery/electronics/online/atm/gas/other]: ").strip().lower()
            velocity     = input("  Velocity [normal/high/very_high]: ").strip().lower()
            card_present = input("  Card present? [yes/no]: ").strip().lower() == "yes"

            X, _, reasons = build_feature_vector(
                amount, hour, location, merchant, velocity, card_present
            )
            prob = predict(model, scaler, feature_names, X)
            display_result(prob, reasons, config.DECISION_THRESHOLD)

            again = input("  Predict another? [y/n]: ").strip().lower()
            if again != "y":
                break
            print()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except ValueError as e:
            print(f"  Invalid input: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Credit Card Fraud Predictor")
    p.add_argument("--amount",   type=float, default=None)
    p.add_argument("--hour",     type=int,   default=None)
    p.add_argument("--location", type=str,   default="home",
                   choices=["home", "near", "different", "foreign"])
    p.add_argument("--merchant", type=str,   default="grocery",
                   choices=["grocery", "electronics", "online", "atm", "gas", "other"])
    p.add_argument("--velocity", type=str,   default="normal",
                   choices=["normal", "high", "very_high"])
    p.add_argument("--no-card",  action="store_true", help="Card not present")
    return p.parse_args()


def main():
    print("=" * 60)
    print("  STEP 5 — Transaction Fraud Predictor")
    print("=" * 60)

    model, _    = load_best_model()
    scaler       = load_scaler()
    feature_names = load_feature_names()

    args = parse_args()

    if args.amount is not None and args.hour is not None:
        # CLI mode
        X, _, reasons = build_feature_vector(
            args.amount, args.hour, args.location,
            args.merchant, args.velocity, not args.no_card,
        )
        prob = predict(model, scaler, feature_names, X)
        display_result(prob, reasons, config.DECISION_THRESHOLD)
    else:
        # Interactive demo with sample transactions
        print("\n── Sample Predictions ──")
        samples = [
            dict(amount=45,    hour=14, location="home",    merchant="grocery",     velocity="normal",    card_present=True,  label="Normal grocery purchase"),
            dict(amount=4299,  hour=2,  location="foreign", merchant="electronics", velocity="very_high", card_present=False, label="Suspicious late-night electronics"),
            dict(amount=9500,  hour=1,  location="foreign", merchant="atm",         velocity="very_high", card_present=False, label="High-risk ATM abroad at 1 AM"),
            dict(amount=87,    hour=11, location="near",    merchant="gas",         velocity="normal",    card_present=True,  label="Gas station nearby"),
            dict(amount=1850,  hour=3,  location="foreign", merchant="online",      velocity="high",      card_present=False, label="Online purchase from unknown country"),
        ]

        for s in samples:
            label = s.pop("label")
            X, _, reasons = build_feature_vector(**s)
            prob = predict(model, scaler, feature_names, X)
            decision = "⛔ FRAUD" if prob >= config.DECISION_THRESHOLD else "✅ Legit"
            print(f"  {decision} ({prob*100:5.1f}%) — {label}")

        print()
        interactive_mode(model, scaler, feature_names)


if __name__ == "__main__":
    main()