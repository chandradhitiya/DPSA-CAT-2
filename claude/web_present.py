"""
Build structured context for Flask results page (grouped tables + copy).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

import experiment_runner as er

# Columns shown in grouped UI tables (readable order)
WEB_METRIC_COLS = [
    "method",
    "stage",
    "epsilon",
    "accounted_epsilon",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "cohen_kappa",
    "auc_roc",
    "avg_precision",
    "specificity",
    "npv",
    "fpr",
    "log_loss",
]


def _fmt_df_html(sub: pd.DataFrame) -> str:
    if sub is None or len(sub) == 0:
        return "<p class='empty-note'>No rows.</p>"
    disp = sub.copy()
    num_cols = disp.select_dtypes(include=["float64", "float32", "int64"]).columns
    for c in num_cols:
        disp[c] = disp[c].round(4)
    use_cols = [c for c in WEB_METRIC_COLS if c in disp.columns]
    if "delta" in disp.columns and "delta" not in use_cols:
        use_cols.insert(2, "delta")
    disp = disp[[c for c in use_cols if c in disp.columns]]
    return disp.to_html(classes="metrics-table", border=0, index=False, na_rep="—")


def load_run_metadata(output_dir: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    path = os.path.join(output_dir, er.RUN_METADATA_FILENAME)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    epsilons = list(er.DEFAULT_EPSILONS)
    if df is not None and "epsilon" in df.columns:
        found = sorted({float(x) for x in df["epsilon"].dropna().unique().tolist()})
        if found:
            epsilons = found
    return {
        "epsilons": epsilons,
        "delta": er.DEFAULT_DELTA,
        "random_state": 42,
    }


def build_results_sections(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Ordered sections for the results template."""
    sections: List[Dict[str, Any]] = []

    base = df[df["stage"] == "BASELINE"].copy()
    sections.append(
        {
            "id": "baseline",
            "title": "1. Standard ML baselines (no privacy noise)",
            "tag": "BASELINE",
            "intro": (
                "Trained on clean scaled + one-hot features. No ε: these are your "
                "<strong>utility ceiling</strong> for comparison. Logistic regression is a "
                "linear baseline; random forest is a stronger non-private tabular model."
            ),
            "noise": None,
            "blocks": [{"subtitle": None, "html": _fmt_df_html(base)}],
        }
    )

    pre_lap = df[df["method"].astype(str).str.contains("PreDP_Laplace", na=False)].copy()
    pre_gau = df[df["method"].astype(str).str.contains("PreDP_Gaussian", na=False)].copy()
    sections.append(
        {
            "id": "preprocessing",
            "title": "2. Pre-processing DP (input perturbation)",
            "tag": "PRE",
            "intro": (
                "Noise is added to <strong>training features</strong> after scaling/clipping, "
                "then a logistic regression is fit on the noisy matrix. Test evaluation uses "
                "the <strong>clean</strong> test set. Each row is one ε run."
            ),
            "noise": (
                "Two noise modes are compared side by side: "
                "<strong>Laplace</strong> (ε-DP style scale on bounded features) and "
                "<strong>Gaussian</strong> (σ tied to ε and δ on bounded features)."
            ),
            "blocks": [
                {
                    "subtitle": "Laplace noise on inputs → train LR",
                    "html": _fmt_df_html(pre_lap),
                },
                {
                    "subtitle": "Gaussian noise on inputs → train LR",
                    "html": _fmt_df_html(pre_gau),
                },
            ],
        }
    )

    sgd = df[df["stage"] == "DP_SGD"].copy()
    sections.append(
        {
            "id": "dp-sgd",
            "title": "3. Training-time DP (DP-SGD, Opacus)",
            "tag": "DP-SGD",
            "intro": (
                "A <strong>linear model in PyTorch</strong> is trained with "
                "<strong>per-sample gradient clipping</strong> and "
                "<strong>Gaussian noise</strong> on gradients (DP-SGD). "
                "<code>accounted_epsilon</code> is the privacy budget reported by Opacus "
                "after training (target ε is the ε column)."
            ),
            "noise": (
                "Mechanism: <strong>Gaussian</strong> on clipped gradients. "
                "δ is fixed for the accountant (see experiment setup)."
            ),
            "blocks": [{"subtitle": None, "html": _fmt_df_html(sgd)}],
        }
    )

    post_lap = df[df["method"].astype(str).str.contains("PostDP_Laplace", na=False)].copy()
    post_gau = df[df["method"].astype(str).str.contains("PostDP_Gaussian", na=False)].copy()
    sections.append(
        {
            "id": "postprocessing",
            "title": "4. Post-processing DP (output perturbation)",
            "tag": "POST",
            "intro": (
                "The <strong>non-private</strong> baseline model (LR or RF) produces class "
                "probabilities; we add noise to the positive-class score, clip to [0,1], "
                "then threshold at 0.5. This shows release-time privacy separate from training."
            ),
            "noise": (
                "<strong>Laplace</strong> and <strong>Gaussian</strong> noise on probabilities "
                "are shown separately. <em>LR</em> vs <em>RF</em> rows show which baseline "
                "probabilities were noised."
            ),
            "blocks": [
                {
                    "subtitle": "Laplace noise on released probabilities",
                    "html": _fmt_df_html(post_lap),
                },
                {
                    "subtitle": "Gaussian noise on released probabilities",
                    "html": _fmt_df_html(post_gau),
                },
            ],
        }
    )

    return sections


def per_epsilon_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Mini summary per ε across non-baseline rows."""
    sub = df[df["stage"] != "BASELINE"].copy()
    if sub.empty or "epsilon" not in sub.columns:
        return []
    rows_out = []
    for eps, grp in sub.groupby("epsilon", sort=True):
        if pd.isna(eps):
            continue
        best = grp.loc[grp["accuracy"].idxmax()]
        rows_out.append(
            {
                "epsilon": float(eps),
                "best_method": str(best["method"]),
                "best_stage": str(best["stage"]),
                "best_accuracy": round(float(best["accuracy"]), 4),
                "best_f1": round(float(best["f1"]), 4),
                "n_runs": int(len(grp)),
            }
        )
    return rows_out


def build_page_context(output_dir: str, df: pd.DataFrame) -> Dict[str, Any]:
    meta = load_run_metadata(output_dir, df=df)
    epsilons = meta.get("epsilons", er.DEFAULT_EPSILONS)
    delta = float(meta.get("delta", er.DEFAULT_DELTA))

    return {
        "meta": meta,
        "epsilons": epsilons,
        "delta": delta,
        "delta_fmt": f"{delta:.2e}",
        "sections": build_results_sections(df),
        "epsilon_why": (
            "We sweep several values of ε (epsilon) because it is the usual "
            "<strong>privacy budget</strong> knob: "
            "<strong>smaller ε ⇒ stronger privacy</strong> (typically more noise or a "
            "tighter DP-SGD budget), which often <strong>reduces accuracy / F1</strong>. "
            "The exact trade-off depends on the mechanism, model, and dataset. "
            "δ (delta) is the small failure probability paired with Gaussian mechanisms."
        ),
        "per_epsilon": per_epsilon_summary(df),
        "full_table_html": _fmt_df_html(df),
    }
