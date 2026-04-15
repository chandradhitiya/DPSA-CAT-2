"""
# PREPROCESSING_DP — Laplace/Gaussian noise on training features, then sklearn LR.

# DP_SGD — PyTorch linear classifier + Opacus (gradient clipping + Gaussian noise).

# POSTPROCESSING_DP — train LR on clean data; add noise to positive-class probability.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.base import ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

import dp_mechanisms as dp


def train_preprocessing_dp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    mechanism: str,
    delta: float,
    clip_bound: float = 3.0,
    l1_sensitivity: float = 6.0,
    l2_sensitivity: float = 6.0,
) -> Tuple[ClassifierMixin, Dict[str, np.ndarray]]:
    Xn = dp.clip_features(X_train.copy(), clip_bound)
    if mechanism == "laplace":
        Xn = dp.add_laplace_noise_vector(Xn, epsilon, l1_sensitivity)
    elif mechanism == "gaussian":
        Xn = dp.add_gaussian_noise_vector(Xn, epsilon, delta, l2_sensitivity)
    else:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=2000, random_state=42, class_weight="balanced", solver="lbfgs"
    )
    model.fit(Xn, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return model, {"pred": pred, "proba": proba}


class TabularLogit(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def train_dp_sgd_opacus(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_epsilon: float,
    delta: float,
    *,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.08,
    max_grad_norm: float = 1.0,
    random_state: int = 42,
) -> Tuple[nn.Module, Dict[str, np.ndarray], float]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cpu")
    d = X_train.shape[1]
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TabularLogit(d).to(device)
    model = PrivacyEngine.get_compatible_module(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    engine = PrivacyEngine()
    model, optimizer, loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
        criterion=criterion,
        poisson_sampling=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        )
        proba = torch.sigmoid(logits_te).cpu().numpy()
    pred = (proba >= 0.5).astype(np.int64)

    spent_eps = float(engine.get_epsilon(delta))
    return model, {"pred": pred, "proba": proba}, spent_eps


def predict_postprocessing_dp(
    model: ClassifierMixin,
    X_test: np.ndarray,
    epsilon: float,
    mechanism: str,
    delta: float,
    l1_sensitivity: float = 1.0,
    l2_sensitivity: float = 1.0,
) -> Dict[str, np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1].astype(np.float64)
    if mechanism == "laplace":
        noisy = dp.add_laplace_noise_scalar(proba, epsilon, l1_sensitivity)
    elif mechanism == "gaussian":
        noisy = dp.add_gaussian_noise_scalar(proba, epsilon, delta, l2_sensitivity)
    else:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")
    noisy = np.clip(noisy, 0.0, 1.0)
    pred = (noisy >= 0.5).astype(np.int64)
    return {"pred": pred, "proba": noisy}
