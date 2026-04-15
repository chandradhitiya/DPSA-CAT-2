# BASELINE — non-private models (same preprocessing as DP runs)
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def predict_labels_probs(model: LogisticRegression, X: np.ndarray):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pred, proba


def train_random_forest_baseline(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> RandomForestClassifier:
    """Stronger tabular baseline — comparable non-DP ceiling for the same features."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def predict_rf_labels_probs(model: RandomForestClassifier, X: np.ndarray):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pred, proba
