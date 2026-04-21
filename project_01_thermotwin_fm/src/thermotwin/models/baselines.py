"""Simple forecasting baselines for honest comparison.

We include two baselines:

- NaiveLast: repeat the last observed target for every horizon step.
- LinearAR:  a per-target ridge regression on recent lags of the target and
             the full last-step feature vector.

Both conform to the same `predict(windows)` interface so train/evaluate can
treat them uniformly with the neural model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge

from thermotwin.features.resample_and_window import WindowArrays


@dataclass
class NaiveLastForecaster:
    horizon: int
    n_targets: int

    def fit(self, windows: WindowArrays) -> "NaiveLastForecaster":
        return self

    def predict(self, windows: WindowArrays) -> np.ndarray:
        # Context contains features; target values are not in context by definition,
        # so we reconstruct naive-last from the last known context feature that matches each target.
        # We assume targets are numerical and their last observed values are passed via `target[:, 0]`
        # at call time only when the window itself stored context of targets too.
        # For simplicity, we reuse the first target step's value in the context window if present.
        # Shape: (N, H, T) broadcast of the last context observation of each target by name.
        last_obs = _last_context_target_values(windows)
        return np.repeat(last_obs[:, None, :], self.horizon, axis=1)


def _last_context_target_values(windows: WindowArrays) -> np.ndarray:
    """Read the last context-step values for each target feature, if present in the feature list."""
    last = windows.context[:, -1, :]                # (N, F)
    target_cols = []
    for t in windows.target_order:
        if t in windows.feature_order:
            target_cols.append(windows.feature_order.index(t))
    if not target_cols:
        # Fall back to zeros — callers may scale-invert this; naive baseline on unknown targets is a no-op.
        return np.zeros((windows.context.shape[0], len(windows.target_order)), dtype=np.float32)
    return last[:, target_cols]


@dataclass
class LinearARForecaster:
    horizon: int
    n_targets: int
    lags: int = 6
    alpha: float = 1.0
    models: list[Ridge] = None  # type: ignore[assignment]
    target_feature_idx: list[int] = None  # type: ignore[assignment]

    def _design_matrix(self, windows: WindowArrays) -> np.ndarray:
        # Use the last `lags` context steps (flattened) + the last feature vector.
        lags = min(self.lags, windows.context.shape[1])
        recent = windows.context[:, -lags:, :].reshape(windows.context.shape[0], -1)
        return recent

    def fit(self, windows: WindowArrays) -> "LinearARForecaster":
        X = self._design_matrix(windows)
        self.models = []
        self.target_feature_idx = []
        # Fit one ridge per (target, horizon_step).
        for t_idx in range(self.n_targets):
            self.target_feature_idx.append(t_idx)
            per_horizon = []
            for h in range(self.horizon):
                y = windows.target[:, h, t_idx]
                m = Ridge(alpha=self.alpha)
                m.fit(X, y)
                per_horizon.append(m)
            self.models.append(per_horizon)
        return self

    def predict(self, windows: WindowArrays) -> np.ndarray:
        if self.models is None:
            raise RuntimeError("LinearARForecaster must be fit before predict")
        X = self._design_matrix(windows)
        N = X.shape[0]
        out = np.zeros((N, self.horizon, self.n_targets), dtype=np.float32)
        for t_idx, per_horizon in enumerate(self.models):
            for h, m in enumerate(per_horizon):
                out[:, h, t_idx] = m.predict(X).astype(np.float32)
        return out
