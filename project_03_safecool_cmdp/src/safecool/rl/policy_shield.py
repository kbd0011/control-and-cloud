"""Policy shield: safety-aware action wrapper.

Given a candidate action from any policy, the shield:

1. Checks fast surrogate safety conditions (box bounds, rate limits).
2. Optionally forecasts one step ahead with the world model and rejects
   actions whose predicted next-step inlet temperature exceeds
   `max_inlet_temp_c - margin`.
3. If the candidate is unsafe, projects it into the feasible set via
   `project_to_safe_action`.
4. If projection cannot eliminate the violation, falls back to a
   conservative policy's action.
5. Counts every intervention so deployment metrics surface the shield's
   contribution.

The shield is deliberately deterministic and inspectable; there is no
learned component here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from safecool.envs.safety_constraints import (
    SafetyConfig,
    compute_constraint_cost,
    project_to_safe_action,
)
from safecool.envs.lcopt_env import STATE_FIELDS
from safecool.models.world_model import WorldModel, predict_next_inlet_temp


@dataclass
class ShieldStats:
    total_calls: int = 0
    projected: int = 0
    fallback_used: int = 0
    lookahead_rejected: int = 0

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "projected": self.projected,
            "fallback_used": self.fallback_used,
            "lookahead_rejected": self.lookahead_rejected,
            "projection_rate": self.projected / max(self.total_calls, 1),
            "fallback_rate": self.fallback_used / max(self.total_calls, 1),
        }


@dataclass
class PolicyShield:
    safety_cfg: SafetyConfig
    coolant_delta_range: tuple[float, float] = (-1.0, 1.0)
    fan_delta_range: tuple[float, float] = (-10.0, 10.0)
    fallback_policy: Callable[[np.ndarray], np.ndarray] | None = None
    world_model: WorldModel | None = None
    wm_state_mean: np.ndarray | None = None
    wm_state_std: np.ndarray | None = None
    use_lookahead: bool = True
    lookahead_margin_c: float = 0.2
    stats: ShieldStats = field(default_factory=ShieldStats)

    # ------------------------------------------------------------------ checks

    def is_safe(self, action: np.ndarray, state: np.ndarray) -> bool:
        a = np.asarray(action, dtype=np.float32).flatten()
        if a.shape != (2,):
            return False
        # Range checks (pre-projection bounds).
        if not (self.coolant_delta_range[0] - 1e-6 <= a[0] <= self.coolant_delta_range[1] + 1e-6):
            return False
        if not (self.fan_delta_range[0] - 1e-6 <= a[1] <= self.fan_delta_range[1] + 1e-6):
            return False
        # Rate limits.
        if abs(a[0]) > self.safety_cfg.max_rate_change_coolant_c + 1e-6:
            return False
        if abs(a[1]) > self.safety_cfg.max_rate_change_fan_pct + 1e-6:
            return False
        # Operating-window clips: applying the delta must not push controls out of bounds.
        inlet_idx = STATE_FIELDS.index("inlet_temp_c")
        coolant_idx = STATE_FIELDS.index("coolant_supply_c")
        fan_idx = STATE_FIELDS.index("fan_speed_pct")
        cs_low, cs_high = self.safety_cfg.coolant_supply_bounds
        f_low, f_high = self.safety_cfg.fan_speed_bounds
        next_coolant = float(state[coolant_idx]) + float(a[0])
        next_fan = float(state[fan_idx]) + float(a[1])
        if next_coolant < cs_low - 1e-3 or next_coolant > cs_high + 1e-3:
            return False
        if next_fan < f_low - 1e-3 or next_fan > f_high + 1e-3:
            return False
        # Current state should not already be above the inlet hard limit.
        if float(state[inlet_idx]) > self.safety_cfg.max_inlet_temp_c + 1e-3:
            return False
        # Optional world-model look-ahead.
        if self.use_lookahead and self.world_model is not None:
            predicted = predict_next_inlet_temp(
                self.world_model, state, a, self.wm_state_mean, self.wm_state_std, inlet_idx=inlet_idx
            )
            if predicted > self.safety_cfg.max_inlet_temp_c - self.lookahead_margin_c:
                return False
        return True

    # ------------------------------------------------------------------ shield

    def __call__(self, action: np.ndarray, state: np.ndarray) -> dict:
        """Return dict with keys: action (np.ndarray), intervened (bool), reason (str | None)."""
        self.stats.total_calls += 1
        candidate = np.asarray(action, dtype=np.float32).flatten()
        reason = None

        if self.is_safe(candidate, state):
            return {"action": candidate, "intervened": False, "reason": None}

        # Try projection first.
        projected = project_to_safe_action(
            candidate, state, self.safety_cfg,
            coolant_delta_range=self.coolant_delta_range,
            fan_delta_range=self.fan_delta_range,
        )
        if self.is_safe(projected, state):
            self.stats.projected += 1
            reason = "projected"
            return {"action": projected, "intervened": True, "reason": reason}

        # Fall back to a conservative policy, then project its output for safety.
        if self.fallback_policy is not None:
            fb = np.asarray(self.fallback_policy(state), dtype=np.float32).flatten()
            fb_proj = project_to_safe_action(
                fb, state, self.safety_cfg,
                coolant_delta_range=self.coolant_delta_range,
                fan_delta_range=self.fan_delta_range,
            )
            self.stats.fallback_used += 1
            # If look-ahead still flags the fallback, record it but emit the projected fallback anyway,
            # because the fallback is by construction the safest thing we know.
            if self.use_lookahead and self.world_model is not None:
                pred = predict_next_inlet_temp(
                    self.world_model, state, fb_proj, self.wm_state_mean, self.wm_state_std
                )
                if pred > self.safety_cfg.max_inlet_temp_c - self.lookahead_margin_c:
                    self.stats.lookahead_rejected += 1
            return {"action": fb_proj, "intervened": True, "reason": "fallback"}

        # Last resort: the projected candidate even if look-ahead still flags it.
        self.stats.lookahead_rejected += 1
        return {"action": projected, "intervened": True, "reason": "projected_warn"}


def conservative_fallback(state: np.ndarray) -> np.ndarray:
    """A small, deterministic cool+boost-fans action used as the shield fallback."""
    return np.array([-0.5, 3.0], dtype=np.float32)
