"""
Healthcare treatments CSV — load, binary outcome, drop identifiers, sample, OHE + scale.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_DEFAULT = "y"

FAVORABLE_OUTCOMES = frozenset(
    {
        "successful",
        "discharged",
        "stable",
        "partially successful",
    }
)

READ_COLS = [
    "treatment_outcome_status",
    "treatment_duration",
    "treatment_cost",
    "treatment_type",
    "speciality_id_x",
    "affiliated_hospital",
    "location_id",
    "country",
    "state",
    "city",
    "gender",
    "age",
    "disease_id",
    "speciality_id_y",
    "disease_type",
    "severity",
    "transmission_mode",
    "mortality_rate",
]


def get_max_sample_rows() -> int:
    return int(os.environ.get("HEALTHCARE_MAX_ROWS", "75000"))


def find_healthcare_csv() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.path.join(script_dir, "healthcare_treatments__csv.csv"),
        os.path.join(repo_root, "healthcare_treatments__csv.csv"),
        os.path.join(os.getcwd(), "healthcare_treatments__csv.csv"),
    ]
    env_path = os.environ.get("HEALTHCARE_CSV_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find healthcare_treatments__csv.csv. Place it in the repo root "
        "or set HEALTHCARE_CSV_PATH."
    )


def load_tabular_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or find_healthcare_csv()
    df = pd.read_csv(path, usecols=lambda c: c in READ_COLS, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    status_col = "treatment_outcome_status"
    if status_col not in df.columns:
        raise ValueError(f"Expected column {status_col!r} in {path}")

    s = df[status_col].astype(str).str.strip().str.lower()
    df[TARGET_DEFAULT] = s.isin({x.lower() for x in FAVORABLE_OUTCOMES}).astype(np.int64)
    df = df.drop(columns=[status_col])

    max_n = get_max_sample_rows()
    if len(df) > max_n:
        df, _ = train_test_split(
            df,
            train_size=max_n,
            random_state=42,
            stratify=df[TARGET_DEFAULT],
        )

    return df


def build_feature_preprocessor(X_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=40,
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No columns to preprocess.")
    pre = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    pre.set_output(transform="pandas")
    return pre, num_cols, cat_cols


def train_test_arrays(
    df: pd.DataFrame,
    target: str = TARGET_DEFAULT,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Missing target column {target!r}")
    y = df[target].to_numpy(dtype=np.int64)
    X_df = df.drop(columns=[target])
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pre, num_cols, cat_cols = build_feature_preprocessor(X_train_df)
    X_train = pre.fit_transform(X_train_df).to_numpy(dtype=np.float64)
    X_test = pre.transform(X_test_df).to_numpy(dtype=np.float64)
    return X_train, X_test, y_train, y_test, pre, num_cols, cat_cols
