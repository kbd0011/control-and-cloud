"""Forecast-based anomaly detector ensemble.

Three interpretable streams:

1. **Robust z-score**: (value − rolling_median) / rolling_MAD.
   Fast, non-parametric, resistant to short outliers polluting the baseline.

2. **Forecast residual**: one-step-ahead AR(k) forecast from the last
   `residual_window` observations. The absolute standardized residual is the
   score. This is explicitly interpretable and computes cheaply per series.

3. **Changepoint score**: absolute difference of the two rolling means on
   either side of each timestamp — a CUSUM-style approximation that fires on
   shifts in level.

Fusion:
- Default (no labels): fixed-weight convex combination from config.
- With labels: a calibrated logistic meta-model over the three streams.
- An optional isotonic regression is applied to the fused probability to
  sharpen calibration.

Thresholding respects a configured daily alert budget: we sort events by
probability and take the top-k consistent with budget × active_days.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def rolling_zscore(
    values: np.ndarray, window: int = 48, clip: float = 10.0
) -> np.ndarray:
    """Robust rolling z-score using median and MAD.

    Values within the first `window` positions get computed from whatever data
    is available, falling back to z=0 when fewer than 3 points are available.
    """
    s = pd.Series(values, dtype=float)
    med = s.rolling(window=window, min_periods=3).median()
    mad = (s - med).abs().rolling(window=window, min_periods=3).median() * 1.4826
    z = (s - med) / (mad.replace(0.0, np.nan))
    z = z.fillna(0.0).clip(-clip, clip).to_numpy()
    return np.abs(z)


def forecast_residual_score(
    values: np.ndarray, window: int = 24, order: int = 3
) -> np.ndarray:
    """Rolling AR(order) one-step-ahead absolute standardized residual.

    Simple and interpretable: for each timestep, fit coefficients on the prior
    `window` observations and score the current step's residual in units of
    the fitting-window standard deviation.
    """
    n = len(values)
    out = np.zeros(n, dtype=float)
    for t in range(window + order, n):
        y = values[t - window : t]
        if np.isnan(y).any():
            continue
        # Build lag design matrix.
        X = np.stack([y[order - k : order - k + (window - order)] for k in range(1, order + 1)], axis=1)
        y_lag = y[order:]
        # Solve least squares.
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_lag, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Predict the current step.
        lag_vec = np.array([values[t - k] for k in range(1, order + 1)])
        if np.isnan(lag_vec).any():
            continue
        pred = float(lag_vec @ coef)
        residual = float(values[t] - pred)
        sigma = float(np.std(y_lag - X @ coef) + 1e-6)
        out[t] = abs(residual) / sigma
    return out


def changepoint_score(values: np.ndarray, window: int = 24) -> np.ndarray:
    """Absolute difference of rolling means on either side of each step.

    Peaks when the mean level changes abruptly.
    """
    s = pd.Series(values, dtype=float)
    left = s.rolling(window=window, min_periods=3).mean().shift(1)
    right = s[::-1].rolling(window=window, min_periods=3).mean().shift(1)[::-1]
    diff = (right - left).abs().fillna(0.0)
    # Normalize by rolling MAD so the scale is comparable across series.
    mad = (s - s.rolling(window=window, min_periods=3).median()).abs().rolling(window=window, min_periods=3).median() * 1.4826
    mad = mad.replace(0, np.nan).fillna(diff.std() + 1e-6)
    return (diff / mad).fillna(0.0).to_numpy()


def compute_detector_streams(
    values: np.ndarray,
    *,
    rolling_window: int,
    residual_window: int,
    changepoint_window: int,
    zscore_clip: float = 10.0,
) -> dict[str, np.ndarray]:
    """Compute all three detector streams for one (entity, metric) series."""
    return {
        "zscore": rolling_zscore(values, window=rolling_window, clip=zscore_clip),
        "residual": forecast_residual_score(values, window=residual_window),
        "changepoint": changepoint_score(values, window=changepoint_window),
    }


def _sigmoid_normalize(x: np.ndarray, slope: float = 1.0) -> np.ndarray:
    """Map raw, non-negative score to (0, 1) via 1 - exp(-slope * x)."""
    return 1.0 - np.exp(-slope * np.clip(x, 0.0, None))


@dataclass
class FusedDetector:
    """Calibrated fusion of detector streams into anomaly probabilities."""

    rolling_window: int = 48
    residual_window: int = 24
    changepoint_window: int = 24
    zscore_clip: float = 10.0
    default_weights: dict[str, float] = field(
        default_factory=lambda: {"zscore": 0.4, "residual": 0.4, "changepoint": 0.2}
    )
    logistic: LogisticRegression | None = None
    isotonic: IsotonicRegression | None = None
    stream_names: tuple[str, ...] = ("zscore", "residual", "changepoint")

    def streams_for_series(self, values: np.ndarray) -> dict[str, np.ndarray]:
        return compute_detector_streams(
            values,
            rolling_window=self.rolling_window,
            residual_window=self.residual_window,
            changepoint_window=self.changepoint_window,
            zscore_clip=self.zscore_clip,
        )

    def _stack_streams(self, streams: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([streams[n] for n in self.stream_names], axis=-1)

    def fit_fusion(self, streams: dict[str, np.ndarray], labels: np.ndarray) -> "FusedDetector":
        """Fit logistic + isotonic on (score_matrix, binary_labels)."""
        X = self._stack_streams(streams)
        y = np.asarray(labels).astype(int)
        if len(np.unique(y)) < 2:
            logger.warning("Fusion labels have a single class; falling back to default weights")
            return self
        X_norm = np.stack([_sigmoid_normalize(X[:, i]) for i in range(X.shape[1])], axis=-1)
        self.logistic = LogisticRegression(max_iter=1000).fit(X_norm, y)
        prob = self.logistic.predict_proba(X_norm)[:, 1]
        # Isotonic calibration on top of logistic probabilities.
        try:
            self.isotonic = IsotonicRegression(out_of_bounds="clip").fit(prob, y)
        except ValueError:
            self.isotonic = None
        return self

    def score_series(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Compute streams, fused probability, and normalized per-stream scores."""
        streams = self.streams_for_series(values)
        stream_norm = {k: _sigmoid_normalize(v) for k, v in streams.items()}
        if self.logistic is not None:
            X = self._stack_streams(stream_norm)
            prob = self.logistic.predict_proba(X)[:, 1]
            if self.isotonic is not None:
                prob = self.isotonic.transform(prob)
        else:
            # Fixed-weight convex combination.
            weights = np.array([self.default_weights[n] for n in self.stream_names], dtype=float)
            weights = weights / weights.sum()
            X = self._stack_streams(stream_norm)
            prob = X @ weights
        prob = np.clip(prob, 0.0, 1.0)
        return {**streams, "prob": prob, **{f"{k}_norm": v for k, v in stream_norm.items()}}

    # Serialization helpers ------------------------------------------------

    def to_dict(self) -> dict:
        payload = {
            "rolling_window": self.rolling_window,
            "residual_window": self.residual_window,
            "changepoint_window": self.changepoint_window,
            "zscore_clip": self.zscore_clip,
            "default_weights": self.default_weights,
            "stream_names": list(self.stream_names),
        }
        if self.logistic is not None:
            payload["logistic_coef"] = self.logistic.coef_.tolist()
            payload["logistic_intercept"] = float(self.logistic.intercept_[0])
        if self.isotonic is not None:
            payload["isotonic_X"] = self.isotonic.X_thresholds_.tolist()
            payload["isotonic_y"] = self.isotonic.y_thresholds_.tolist()
        return payload

    @classmethod
    def from_dict(cls, d: dict) -> "FusedDetector":
        obj = cls(
            rolling_window=int(d["rolling_window"]),
            residual_window=int(d["residual_window"]),
            changepoint_window=int(d["changepoint_window"]),
            zscore_clip=float(d["zscore_clip"]),
            default_weights={k: float(v) for k, v in d["default_weights"].items()},
            stream_names=tuple(d.get("stream_names", ("zscore", "residual", "changepoint"))),
        )
        if "logistic_coef" in d:
            obj.logistic = LogisticRegression()
            obj.logistic.coef_ = np.asarray(d["logistic_coef"], dtype=float)
            obj.logistic.intercept_ = np.asarray([d["logistic_intercept"]], dtype=float)
            obj.logistic.classes_ = np.array([0, 1])
        if "isotonic_X" in d:
            obj.isotonic = IsotonicRegression(out_of_bounds="clip")
            obj.isotonic.X_thresholds_ = np.asarray(d["isotonic_X"], dtype=float)
            obj.isotonic.y_thresholds_ = np.asarray(d["isotonic_y"], dtype=float)
            obj.isotonic.X_min_ = float(obj.isotonic.X_thresholds_.min())
            obj.isotonic.X_max_ = float(obj.isotonic.X_thresholds_.max())
            obj.isotonic.increasing_ = True
            obj.isotonic.f_ = None  # Will be rebuilt on first transform via _build_f.
            obj.isotonic._build_f(obj.isotonic.X_thresholds_, obj.isotonic.y_thresholds_)
        return obj


def apply_alert_budget(
    events_df: pd.DataFrame,
    *,
    daily_budget: int,
    time_col: str = "timestamp",
    prob_col: str = "prob",
    probability_threshold: float = 0.5,
) -> pd.DataFrame:
    """Given a per-event probability table, keep at most `daily_budget` events per day.

    Events below `probability_threshold` are dropped first. Remaining events are
    sorted by probability (descending) within each UTC day and clipped to the budget.
    """
    if events_df.empty:
        return events_df.copy()
    df = events_df.copy()
    df = df[df[prob_col] >= probability_threshold].copy()
    if df.empty:
        return df
    df[time_col] = pd.to_datetime(df[time_col])
    df["__day"] = df[time_col].dt.date
    df = df.sort_values([prob_col], ascending=False)
    kept = df.groupby("__day", as_index=False).head(daily_budget)
    return kept.drop(columns="__day").reset_index(drop=True)
