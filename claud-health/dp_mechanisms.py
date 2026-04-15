"""
Laplace / Gaussian noise helpers for input perturbation and output release.
"""
from __future__ import annotations

import numpy as np


def laplace_scale_from_l1_sensitivity(epsilon: float, l1_sensitivity: float) -> float:
    """Laplace mechanism: noise ~ Lap(0, b) with b = Δ₁ / ε."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return l1_sensitivity / epsilon


def gaussian_sigma_from_l2_sensitivity(
    epsilon: float, delta: float, l2_sensitivity: float
) -> float:
    """Gaussian mechanism (approx DP): σ = √(2 ln(1.25/δ)) · Δ₂ / ε."""
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("invalid epsilon or delta")
    return float(np.sqrt(2 * np.log(1.25 / delta)) * l2_sensitivity / epsilon)


def add_laplace_noise_vector(x: np.ndarray, epsilon: float, l1_sensitivity: float) -> np.ndarray:
    b = laplace_scale_from_l1_sensitivity(epsilon, l1_sensitivity)
    return x + np.random.laplace(0.0, b, size=x.shape)


def add_gaussian_noise_vector(
    x: np.ndarray, epsilon: float, delta: float, l2_sensitivity: float
) -> np.ndarray:
    sigma = gaussian_sigma_from_l2_sensitivity(epsilon, delta, l2_sensitivity)
    return x + np.random.normal(0.0, sigma, size=x.shape)


def add_laplace_noise_scalar(value: np.ndarray, epsilon: float, l1_sensitivity: float) -> np.ndarray:
    b = laplace_scale_from_l1_sensitivity(epsilon, l1_sensitivity)
    return value + np.random.laplace(0.0, b, size=value.shape)


def add_gaussian_noise_scalar(
    value: np.ndarray, epsilon: float, delta: float, l2_sensitivity: float
) -> np.ndarray:
    sigma = gaussian_sigma_from_l2_sensitivity(epsilon, delta, l2_sensitivity)
    return value + np.random.normal(0.0, sigma, size=value.shape)


def clip_features(x: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(x, -bound, bound)
