"""
2_eda.py
────────
Step 2: Exploratory Data Analysis.
Generates and saves charts to plots/ covering:
  • Class distribution
  • Transaction amount distributions by class
  • Fraud by hour of day
  • Correlation heatmap
  • PCA component box plots (V1–V28 fraud vs legit)
  • Feature pair scatter (V14 vs V17)

Run:
    python 2_eda.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import config

warnings.filterwarnings("ignore")
os.makedirs(config.PLOTS_DIR, exist_ok=True)

try:
    plt.style.use(config.PLOT_STYLE)
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

BLUE  = config.PALETTE["legit"]
RED   = config.PALETTE["fraud"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    if not os.path.exists(config.PROCESSED_PATH):
        print("[ERROR] Run 1_data_preprocessing.py first.")
        raise SystemExit(1)
    return pd.read_csv(config.PROCESSED_PATH)


def save(fig, name: str) -> None:
    path = os.path.join(config.PLOTS_DIR, name)
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[config.TARGET_COLUMN].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    colors = [BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Class Distribution — Highly Imbalanced Dataset", fontsize=13, fontweight="bold")

    # Bar chart
    ax = axes[0]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Number of transactions")
    ax.set_title("Transaction count by class")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 800,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.3f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Class proportion")

    plt.tight_layout()
    save(fig, "01_class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Amount distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_amount_distribution(df: pd.DataFrame) -> None:
    legit = df[df[config.TARGET_COLUMN] == 0]["Amount"]
    fraud = df[df[config.TARGET_COLUMN] == 1]["Amount"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Transaction Amount Analysis", fontsize=13, fontweight="bold")

    # Box plots
    ax = axes[0]
    ax.boxplot([legit, fraud], labels=["Legitimate", "Fraudulent"],
               patch_artist=True,
               boxprops={"facecolor": BLUE, "alpha": 0.7},
               medianprops={"color": "white", "linewidth": 2})
    ax.set_ylabel("Amount ($)")
    ax.set_title("Amount distribution (full range)")
    ax.set_yscale("log")

    # Histogram — legitimate
    ax = axes[1]
    ax.hist(legit.clip(upper=1000), bins=60, color=BLUE, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Amount ($) — clipped at $1,000")
    ax.set_ylabel("Count")
    ax.set_title("Legitimate transactions")

    # Histogram — fraudulent
    ax = axes[2]
    ax.hist(fraud, bins=40, color=RED, alpha=0.8, edgecolor="white")
    ax.set_xlabel("Amount ($)")
    ax.set_title("Fraudulent transactions")
    ax.text(0.97, 0.95,
            f"Mean: ${fraud.mean():.2f}\nMax: ${fraud.max():.2f}\nMedian: ${fraud.median():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    save(fig, "02_amount_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fraud by hour of day
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_by_hour(df: pd.DataFrame) -> None:
    if "Hour" not in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24

    hourly = df.groupby(["Hour", config.TARGET_COLUMN]).size().unstack(fill_value=0)
    hourly.columns = ["Legitimate", "Fraudulent"]
    hourly["FraudRate"] = hourly["Fraudulent"] / (hourly["Legitimate"] + hourly["Fraudulent"]) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Fraud Patterns by Hour of Day", fontsize=13, fontweight="bold")

    bar_colors = [RED if r > hourly["FraudRate"].mean() * 1.5 else "#ef9f27"
                  if r > hourly["FraudRate"].mean() else BLUE
                  for r in hourly["FraudRate"]]

    axes[0].bar(hourly.index, hourly["Fraudulent"], color=bar_colors, edgecolor="white")
    axes[0].set_ylabel("Fraud count")
    axes[0].set_title("Number of fraudulent transactions per hour")
    axes[0].axhline(hourly["Fraudulent"].mean(), color="gray", linestyle="--",
                    linewidth=1, label=f"Mean ({hourly['Fraudulent'].mean():.1f})")
    axes[0].legend(fontsize=9)

    axes[1].plot(hourly.index, hourly["FraudRate"], color=RED, marker="o",
                 linewidth=2, markersize=5)
    axes[1].fill_between(hourly.index, hourly["FraudRate"], alpha=0.15, color=RED)
    axes[1].set_xlabel("Hour of day (0 = midnight)")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_title("Fraud rate (%) by hour")
    axes[1].set_xticks(range(0, 24))

    plt.tight_layout()
    save(fig, "03_fraud_by_hour.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    # Focus on PCA features + Amount + engineered cols vs Class
    feat_cols = [c for c in df.columns if c.startswith("V") or c in
                 ["Amount", "LogAmount", "Hour", "IsNightHour", "IsHighAmount", config.TARGET_COLUMN]]
    corr = df[feat_cols].corr()[[config.TARGET_COLUMN]].drop(config.TARGET_COLUMN).sort_values(config.TARGET_COLUMN)

    fig, ax = plt.subplots(figsize=(5, 12))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Feature correlation with Class (Fraud=1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    plt.tight_layout()
    save(fig, "04_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PCA component distributions (top fraud-correlated)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_distributions(df: pd.DataFrame) -> None:
    top_features = ["V17", "V14", "V12", "V10", "V11", "V4", "V3", "V7", "V16", "V1"]
    top_features = [f for f in top_features if f in df.columns]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("PCA Component Distributions — Legitimate vs Fraudulent",
                 fontsize=13, fontweight="bold")

    legit = df[df[config.TARGET_COLUMN] == 0]
    fraud = df[df[config.TARGET_COLUMN] == 1]

    for ax, feat in zip(axes.ravel(), top_features):
        ax.hist(legit[feat], bins=60, alpha=0.6, color=BLUE, density=True,
                label="Legit", edgecolor="white")
        ax.hist(fraud[feat], bins=40, alpha=0.8, color=RED, density=True,
                label="Fraud", edgecolor="white")
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    save(fig, "05_pca_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Scatter — V14 vs V17 (top two separating components)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_v14_v17(df: pd.DataFrame) -> None:
    if "V14" not in df.columns or "V17" not in df.columns:
        return

    legit = df[df[config.TARGET_COLUMN] == 0].sample(n=min(5000, len(df)), random_state=42)
    fraud = df[df[config.TARGET_COLUMN] == 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(legit["V14"], legit["V17"], alpha=0.15, s=8,
               color=BLUE, label=f"Legitimate ({len(legit):,} sampled)")
    ax.scatter(fraud["V14"], fraud["V17"], alpha=0.7, s=25,
               color=RED, label=f"Fraudulent ({len(fraud):,})", zorder=5)
    ax.set_xlabel("PCA Component V14", fontsize=11)
    ax.set_ylabel("PCA Component V17", fontsize=11)
    ax.set_title("Feature Space Separation — V14 vs V17", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save(fig, "06_scatter_v14_v17.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Statistical summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_statistical_summary(df: pd.DataFrame) -> None:
    print("\n── Statistical Summary by Class ──")
    for col in ["Amount", "LogAmount", "Hour", "IsNightHour"]:
        if col not in df.columns:
            continue
        grp = df.groupby(config.TARGET_COLUMN)[col].agg(["mean", "median", "std"])
        grp.index = ["Legitimate", "Fraudulent"]
        print(f"\n  {col}:")
        print(grp.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 2 — Exploratory Data Analysis")
    print("=" * 60)

    df = load()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns\n")

    print("Generating plots …")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_fraud_by_hour(df)
    plot_correlation_heatmap(df)
    plot_pca_distributions(df)
    plot_scatter_v14_v17(df)

    print_statistical_summary(df)
    print(f"\n✓ EDA complete. All plots saved to {config.PLOTS_DIR}/")
    print("  Run: python 3_train_models.py\n")


if __name__ == "__main__":
    main()