"""
Run the full baseline vs DPML experiment; used by main.py and the Flask UI.
"""
from __future__ import annotations

import io
import os
import sys
import warnings
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import evaluation as ev
import preprocess as pp
import train_baseline as tb
import train_dp as td

DEFAULT_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dp_results_ppml")
)
PLOT_FILENAMES = [
    "epsilon_vs_accuracy_f1.png",
    "baseline_vs_dp_sgd_summary.png",
]

DEFAULT_EPSILONS = [0.1, 0.5, 1.0, 5.0]
DEFAULT_DELTA = 1e-5
RUN_METADATA_FILENAME = "run_metadata.json"


def default_output_dir() -> str:
    return DEFAULT_OUTPUT_DIR


def run_experiment(
    *,
    epsilons: Optional[List[float]] = None,
    delta: float = 1e-5,
    random_state: int = 42,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    log_print: Optional[io.StringIO] = None,
) -> Dict[str, Any]:
    """
    Train baselines + DP variants, save CSV/plots under output_dir.

    Returns dict with keys: metrics_df, output_dir, plot_files, best_row, baselines_df, log_text
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    epsilon_values = epsilons if epsilons is not None else DEFAULT_EPSILONS

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)
        if log_print is not None:
            log_print.write(msg + "\n")

    np.random.seed(random_state)

    df = pp.load_tabular_dataframe()
    X_train, X_test, y_train, y_test, _pre, _nc, _cc = pp.train_test_arrays(
        df, random_state=random_state
    )

    rows = []

    _log("Training baseline: LogisticRegression...")
    base_lr = tb.train_logistic_baseline(X_train, y_train)
    bp, bpro = tb.predict_labels_probs(base_lr, X_test)
    m = ev.classification_metrics(y_test, bp, bpro)
    rows.append(
        {
            "method": "LogisticRegression",
            "stage": "BASELINE",
            "epsilon": np.nan,
            "delta": np.nan,
            "accounted_epsilon": np.nan,
            **m,
        }
    )

    _log("Training baseline: RandomForestClassifier...")
    base_rf = tb.train_random_forest_baseline(
        X_train, y_train, random_state=random_state
    )
    rfp, rfpro = tb.predict_rf_labels_probs(base_rf, X_test)
    m = ev.classification_metrics(y_test, rfp, rfpro)
    rows.append(
        {
            "method": "RandomForestClassifier",
            "stage": "BASELINE",
            "epsilon": np.nan,
            "delta": np.nan,
            "accounted_epsilon": np.nan,
            **m,
        }
    )

    for eps_val in epsilon_values:
        _log(f"\n--- epsilon = {eps_val} ---")

        for mech, label in (("laplace", "Laplace"), ("gaussian", "Gaussian")):
            _log(f"  PREPROCESSING_DP ({label})...")
            _, pout = td.train_preprocessing_dp(
                X_train, y_train, X_test, y_test, eps_val, mech, delta
            )
            m = ev.classification_metrics(y_test, pout["pred"], pout["proba"])
            rows.append(
                {
                    "method": f"PreDP_{label}_LR",
                    "stage": "PREPROCESSING_DP",
                    "epsilon": eps_val,
                    "delta": delta,
                    "accounted_epsilon": eps_val,
                    **m,
                }
            )

        _log("  DP_SGD (Opacus)...")
        _, out, spent = td.train_dp_sgd_opacus(
            X_train,
            y_train,
            X_test,
            y_test,
            eps_val,
            delta,
            random_state=random_state,
        )
        m = ev.classification_metrics(y_test, out["pred"], out["proba"])
        rows.append(
            {
                "method": "DP_SGD_Linear_Opacus",
                "stage": "DP_SGD",
                "epsilon": eps_val,
                "delta": delta,
                "accounted_epsilon": spent,
                **m,
            }
        )

        for mech, label in (("laplace", "Laplace"), ("gaussian", "Gaussian")):
            _log(f"  POSTPROCESSING_DP LR ({label})...")
            out = td.predict_postprocessing_dp(base_lr, X_test, eps_val, mech, delta)
            m = ev.classification_metrics(y_test, out["pred"], out["proba"])
            rows.append(
                {
                    "method": f"PostDP_{label}_LR_proba",
                    "stage": "POSTPROCESSING_DP",
                    "epsilon": eps_val,
                    "delta": delta,
                    "accounted_epsilon": eps_val,
                    **m,
                }
            )
            _log(f"  POSTPROCESSING_DP RF ({label})...")
            out = td.predict_postprocessing_dp(base_rf, X_test, eps_val, mech, delta)
            m = ev.classification_metrics(y_test, out["pred"], out["proba"])
            rows.append(
                {
                    "method": f"PostDP_{label}_RF_proba",
                    "stage": "POSTPROCESSING_DP",
                    "epsilon": eps_val,
                    "delta": delta,
                    "accounted_epsilon": eps_val,
                    **m,
                }
            )

    res = pd.DataFrame(rows)
    if verbose:
        ev.print_comparison_table(res)
    ev.save_metrics_csv(res, out_dir)
    ev.plot_epsilon_tradeoffs(res, out_dir)
    ev.plot_baseline_vs_dp_bar_summary(res, out_dir)

    meta_path = os.path.join(out_dir, RUN_METADATA_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "epsilons": [float(x) for x in epsilon_values],
                "delta": float(delta),
                "random_state": int(random_state),
            },
            mf,
            indent=2,
        )

    best_idx = int(res["accuracy"].values.argmax())
    best = res.iloc[best_idx]
    base_only = res[res["stage"] == "BASELINE"].copy()

    if verbose:
        print("\n" + "=" * 72)
        print("Best test accuracy (any row):")
        print("=" * 72)
        show_cols = [
            "method",
            "stage",
            "epsilon",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "mcc",
            "auc_roc",
            "avg_precision",
        ]
        show_cols = [c for c in show_cols if c in best.index]
        print(best[show_cols].to_string())
        if len(base_only) >= 2:
            print("\nStandard ML baselines (no DP):")
            cols_b = [
                "method",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "mcc",
                "auc_roc",
                "avg_precision",
            ]
            cols_b = [c for c in cols_b if c in base_only.columns]
            print(base_only[cols_b].round(4).to_string(index=False))
        print(f"\nOutputs: {out_dir}/")

    log_text = log_print.getvalue() if log_print is not None else ""

    return {
        "metrics_df": res,
        "output_dir": out_dir,
        "plot_files": PLOT_FILENAMES,
        "best_row": best.to_dict(),
        "baselines_df": base_only,
        "log_text": log_text,
    }
