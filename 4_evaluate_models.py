"""
4_evaluate_models.py
────────────────────
Step 4: Load saved models, evaluate on the held-out test set,
and generate:
  • Classification report (precision, recall, F1)
  • Confusion matrices
  • ROC curves
  • Precision-Recall curves
  • Threshold sensitivity analysis
  • Summary comparison table

Run:
    python 4_evaluate_models.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
)

warnings.filterwarnings("ignore")
import config

os.makedirs(config.PLOTS_DIR, exist_ok=True)

BLUE = config.PALETTE["legit"]
RED  = config.PALETTE["fraud"]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_test() -> tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(config.TEST_PATH):
        print("[ERROR] Run 1_data_preprocessing.py first.")
        raise SystemExit(1)
    df = pd.read_csv(config.TEST_PATH)
    X  = df.drop(columns=[config.TARGET_COLUMN])
    y  = df[config.TARGET_COLUMN]
    print(f"  Test set: {len(X):,} rows | Fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y


def load_model(name: str):
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_keras_model():
    try:
        from tensorflow import keras
        path = os.path.join(config.MODELS_DIR, "neural_network.keras")
        if os.path.exists(path):
            return keras.models.load_model(path)
    except ImportError:
        pass
    return load_model("neural_network")


def get_proba(model, X) -> np.ndarray:
    """Return probability of class=1 (fraud). Handles sklearn & Keras."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        # Keras model
        return model.predict(X, verbose=0).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model, name: str, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict:
    y_prob = get_proba(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fpr_, tpr_, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr_, tpr_)
    ap      = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy:  {acc*100:.3f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {roc_auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraudulent"],
                                digits=4))

    return {
        "name": name,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "auc_roc": roc_auc, "avg_precision": ap,
        "y_prob": y_prob, "y_pred": y_pred,
        "fpr": fpr_, "tpr": tpr_, "cm": cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = ["#e24b4a", "#378add", "#639922", "#ef9f27"]

def plot_roc_curves(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    for res, color in zip(results, COLORS):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['name']} (AUC = {res['auc_roc']:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "07_roc_curves.png")
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_precision_recall(results: list[dict], X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for res, color in zip(results, COLORS):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{res['name']} (AP = {res['avg_precision']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "08_precision_recall_curves.png")
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion_matrices(results: list[dict]) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices (test set)", fontsize=13, fontweight="bold")

    for ax, res in zip(axes, results):
        cm = res["cm"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(res["name"], fontsize=11, fontweight="bold")
        labels = ["Legit", "Fraud"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, rotation=90, va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "09_confusion_matrices.png")
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_threshold_analysis(best_result: dict, y_test: pd.Series) -> None:
    """Show how precision, recall, and F1 change with different thresholds."""
    thresholds = np.linspace(0.05, 0.95, 100)
    precisions, recalls, f1s = [], [], []
    y_prob = best_result["y_prob"]

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    best_idx = np.argmax(f1s)
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, color="#378add", lw=2, label="Precision")
    ax.plot(thresholds, recalls,   color="#e24b4a", lw=2, label="Recall")
    ax.plot(thresholds, f1s,       color="#639922", lw=2.5, label="F1 Score", zorder=5)
    ax.axvline(best_t, color="black", linestyle="--", lw=1.5,
               label=f"Best threshold = {best_t:.2f} (F1={f1s[best_idx]:.4f})")
    ax.axvline(config.DECISION_THRESHOLD, color="gray", linestyle=":",
               label=f"Config threshold = {config.DECISION_THRESHOLD}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sensitivity — {best_result['name']}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "10_threshold_analysis.png")
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    print(f"\n  Optimal threshold: {best_t:.3f}  →  F1 = {f1s[best_idx]:.4f}")


def plot_model_comparison(results: list[dict]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    names   = [r["name"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (res, color) in enumerate(zip(results, COLORS)):
        vals = [res[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=res["name"],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0.8, 1.02])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "11_model_comparison.png")
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    rows = []
    for r in results:
        rows.append({
            "Model":     r["name"],
            "Accuracy":  f"{r['accuracy']*100:.3f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall":    f"{r['recall']*100:.2f}%",
            "F1 Score":  f"{r['f1']:.4f}",
            "AUC-ROC":   f"{r['auc_roc']:.4f}",
            "Avg Prec":  f"{r['avg_precision']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 75)
    print("  MODEL EVALUATION SUMMARY")
    print("=" * 75)
    print(df.to_string(index=False))
    print("=" * 75)

    best = max(results, key=lambda r: r["f1"])
    print(f"\n  🏆 Best model by F1: {best['name']} (F1 = {best['f1']:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 4 — Model Evaluation")
    print("=" * 60)

    X_test, y_test = load_test()
    threshold = config.DECISION_THRESHOLD

    model_map = {
        "Logistic Regression": load_model("logistic_regression"),
        "Random Forest":       load_model("random_forest"),
        "XGBoost":             load_model("xgboost"),
        "Neural Network":      load_keras_model(),
    }

    results = []
    for name, model in model_map.items():
        if model is None:
            print(f"\n  [SKIP] {name} — model file not found.")
            continue
        res = evaluate_model(model, name, X_test, y_test, threshold)
        results.append(res)

    if not results:
        print("[ERROR] No trained models found. Run 3_train_models.py first.")
        return

    print("\nGenerating evaluation plots …")
    plot_roc_curves(results)
    plot_precision_recall(results, X_test, y_test)
    plot_confusion_matrices(results)
    plot_model_comparison(results)

    best = max(results, key=lambda r: r["auc_roc"])
    plot_threshold_analysis(best, y_test)

    print_summary_table(results)
    print(f"\n✓ Evaluation complete. Plots saved to {config.PLOTS_DIR}/")
    print("  Run: streamlit run app.py\n")


if __name__ == "__main__":
    main()