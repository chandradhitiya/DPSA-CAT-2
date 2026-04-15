"""
Load tabular CSV, OneHotEncoder + StandardScaler, stratified train/test split.
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


def find_bank_csv() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    candidates = [
        os.path.join(script_dir, "bank-additional-full.csv"),
        os.path.join(repo_root, "dpsa-data-bank ", "bank-additional-full.csv"),
        os.path.join(repo_root, "dpsa-data-bank", "bank-additional-full.csv"),
        os.path.join(os.getcwd(), "bank-additional-full.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find bank-additional-full.csv. Place it next to this script "
        "or under dpsa-data-bank/ in the repo root."
    )


def load_tabular_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or find_bank_csv()
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 10:
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                return df
        except Exception:
            continue
    raise ValueError(f"Could not read CSV: {path}")


def encode_target_series(y: pd.Series) -> np.ndarray:
    s = y.astype(str).str.strip().str.lower()
    return (s == "yes").astype(np.int64).values


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
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
    y = encode_target_series(df[target])
    X_df = df.drop(columns=[target])
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pre, num_cols, cat_cols = build_feature_preprocessor(X_train_df)
    X_train = pre.fit_transform(X_train_df).to_numpy(dtype=np.float64)
    X_test = pre.transform(X_test_df).to_numpy(dtype=np.float64)
    return X_train, X_test, y_train, y_test, pre, num_cols, cat_cols
