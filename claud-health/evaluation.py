"""Metrics, tables, CSV export, and epsilon trade-off plots."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Extended binary classification metrics for baseline vs DP comparison."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    out["specificity"] = float(specificity)
    out["npv"] = float(npv)
    out["fpr"] = float(fpr)

    if y_prob is not None:
        p = np.asarray(y_prob, dtype=np.float64).ravel()
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        try:
            out["auc_roc"] = float(roc_auc_score(y_true, p))
        except ValueError:
            out["auc_roc"] = float("nan")
        try:
            out["avg_precision"] = float(average_precision_score(y_true, p))
        except ValueError:
            out["avg_precision"] = float("nan")
        try:
            y01 = y_true.astype(int)
            out["log_loss"] = float(
                log_loss(y01, p, labels=[0, 1])
            )
        except ValueError:
            out["log_loss"] = float("nan")
    else:
        out["auc_roc"] = float("nan")
        out["avg_precision"] = float("nan")
        out["log_loss"] = float("nan")

    return out


# Column order for console (wide); CSV always has full frame
DISPLAY_COLS = [
    "method",
    "stage",
    "epsilon",
    "delta",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "mcc",
    "cohen_kappa",
    "specificity",
    "auc_roc",
    "avg_precision",
    "log_loss",
    "accounted_epsilon",
]


def print_comparison_table(df: pd.DataFrame) -> None:
    show = [c for c in DISPLAY_COLS if c in df.columns]
    disp = df[show].copy()
    if "delta" in disp.columns:
        disp["delta"] = disp["delta"].apply(
            lambda v: "" if pd.isna(v) else f"{float(v):.2e}"
        )
    fmt_cols = [
        c
        for c in show
        if c not in ("method", "stage", "delta") and c != "epsilon"
    ]
    for c in fmt_cols:
        if c in disp.columns:
            disp[c] = disp[c].apply(
                lambda v: "" if pd.isna(v) else round(float(v), 4)
            )
    if "epsilon" in disp.columns:
        disp["epsilon"] = disp["epsilon"].apply(
            lambda v: "" if pd.isna(v) else round(float(v), 4)
        )
    print("\n" + "=" * 120)
    print(" COMPARISON — Baseline (standard ML) vs DPML")
    print("=" * 120)
    print(disp.to_string(index=False))


def save_metrics_csv(df: pd.DataFrame, output_dir: str, filename: str = "metrics_all_runs.csv") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved metrics CSV: {path}")
    return path


def plot_epsilon_tradeoffs(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sub = df[df["epsilon"].notna() & (df["epsilon"] > 0)].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for stage, grp in sub.groupby("stage"):
        g = grp.sort_values("epsilon")
        axes[0].plot(
            g["epsilon"], g["accuracy"], marker="o", linewidth=2, label=stage
        )
        axes[1].plot(g["epsilon"], g["f1"], marker="s", linewidth=2, label=stage)
    baselines = df[df["stage"] == "BASELINE"]
    if not baselines.empty:
        styles = ["--", "-.", ":"]
        for i, (_, row) in enumerate(baselines.iterrows()):
            ls = styles[i % len(styles)]
            label = f"Baseline: {row['method']}"
            axes[0].axhline(row["accuracy"], color="gray", linestyle=ls, linewidth=1.5, label=label)
            axes[1].axhline(row["f1"], color="gray", linestyle=ls, linewidth=1.5, label=label)
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Epsilon vs Accuracy (DP stages vs standard ML baselines)")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("F1 (binary)")
    axes[1].set_title("Epsilon vs F1 (DP stages vs standard ML baselines)")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "epsilon_vs_accuracy_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_baseline_vs_dp_bar_summary(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart: standard ML baselines vs mean metric across DP-SGD epsilons."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["accuracy", "balanced_accuracy", "f1", "auc_roc"]
    names: List[str] = []
    vals: Dict[str, List[float]] = {m: [] for m in metrics}

    lr = df[(df["stage"] == "BASELINE") & (df["method"].str.contains("Logistic", na=False))]
    rf = df[(df["stage"] == "BASELINE") & (df["method"].str.contains("Forest", na=False))]
    dpsgd = df[df["stage"] == "DP_SGD"]

    if not lr.empty:
        names.append("LR (baseline)")
        for m in metrics:
            vals[m].append(float(lr.iloc[0][m]))
    if not rf.empty:
        names.append("RF (baseline)")
        for m in metrics:
            vals[m].append(float(rf.iloc[0][m]))
    if not dpsgd.empty:
        names.append("DP-SGD (mean over ε)")
        for m in metrics:
            vals[m].append(float(dpsgd[m].mean()))

    if len(names) < 2:
        return

    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, vals[m], width, label=m.replace("_", " "))
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Standard ML vs DP-SGD (mean utility over privacy sweeps)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "baseline_vs_dp_sgd_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
