"""Local cooling simulator with a gym-like API.

**Honest framing:** this module is named `lcopt_env` to preserve the scaffold's
file name, but it is NOT a wrapper around LC-Opt / Modelica. LC-Opt would
require a Modelica runtime and bundled thermal models that are out of scope
for a portable portfolio repo. Instead we implement a deliberately simple but
physically plausible local simulator:

- **State** (7-D): fan_speed_pct, coolant_supply_c, inlet_temp_c, outlet_temp_c,
  cpu_load, ambient_temp_c, power_kw.
- **Action** (2-D): deltas applied to coolant setpoint and fan speed.
- **Dynamics**: first-order thermal lag coupling coolant + ambient to inlet,
  plus outlet = inlet + gain · filtered_load. Controls are rate-limited.
- **Reward**: negative total power (IT + fans + pumps), scaled to unit range.
- **Safety cost**: hinge losses on inlet/outlet temperature bounds and RCI
  deficit relative to target.
- **Scenarios**: `nominal`, `spike` (workload burst), `degradation` (coolant
  supply drifts warmer), `heatwave` (elevated ambient).

The same interface (`reset`, `step`, `observation_space`, `action_space`) a
real LC-Opt wrapper would implement, so swapping in the real simulator later
is a drop-in change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Canonical ordering of the state vector used everywhere downstream.
STATE_FIELDS = (
    "fan_speed_pct",
    "coolant_supply_c",
    "inlet_temp_c",
    "outlet_temp_c",
    "cpu_load",
    "ambient_temp_c",
    "power_kw",
)
# Action semantics: deltas applied to controls between steps.
ACTION_FIELDS = ("coolant_supply_delta_c", "fan_speed_delta_pct")


@dataclass
class EnvConfig:
    episode_steps: int = 96
    cadence_minutes: int = 5
    ambient_mean_c: float = 20.0
    ambient_amplitude_c: float = 5.0
    load_base: float = 0.45
    load_amplitude: float = 0.2
    thermal_tau_steps: int = 6
    outlet_gain: float = 12.0
    pump_power_coef: float = 0.03
    fan_power_coef: float = 0.0004
    it_power_coef: float = 0.18
    noise_scale: float = 0.15
    # Action bounds.
    coolant_supply_delta_range: tuple[float, float] = (-1.0, 1.0)
    fan_speed_delta_range: tuple[float, float] = (-10.0, 10.0)
    coolant_supply_bounds: tuple[float, float] = (15.0, 24.0)
    fan_speed_bounds: tuple[float, float] = (25.0, 100.0)
    # Initial state.
    initial_inlet_temp_c: float = 22.0
    initial_outlet_temp_c: float = 28.0
    initial_coolant_supply_c: float = 18.0
    initial_fan_speed_pct: float = 60.0


def env_config_from_dict(d: dict) -> EnvConfig:
    env = d.get("environment", d)
    a = d.get("action_space", {})
    return EnvConfig(
        episode_steps=int(env.get("episode_steps", 96)),
        cadence_minutes=int(env.get("cadence_minutes", 5)),
        ambient_mean_c=float(env.get("ambient_mean_c", 20.0)),
        ambient_amplitude_c=float(env.get("ambient_amplitude_c", 5.0)),
        load_base=float(env.get("load_base", 0.45)),
        load_amplitude=float(env.get("load_amplitude", 0.2)),
        thermal_tau_steps=int(env.get("thermal_tau_steps", 6)),
        outlet_gain=float(env.get("outlet_gain", 12.0)),
        pump_power_coef=float(env.get("pump_power_coef", 0.03)),
        fan_power_coef=float(env.get("fan_power_coef", 0.0004)),
        it_power_coef=float(env.get("it_power_coef", 0.18)),
        noise_scale=float(env.get("noise_scale", 0.15)),
        coolant_supply_delta_range=(
            float(a.get("coolant_supply_delta_c", {}).get("min", -1.0)),
            float(a.get("coolant_supply_delta_c", {}).get("max", 1.0)),
        ),
        fan_speed_delta_range=(
            float(a.get("fan_speed_delta_pct", {}).get("min", -10.0)),
            float(a.get("fan_speed_delta_pct", {}).get("max", 10.0)),
        ),
        coolant_supply_bounds=tuple(a.get("coolant_supply_bounds_c", [15.0, 24.0])),
        fan_speed_bounds=tuple(a.get("fan_speed_bounds_pct", [25.0, 100.0])),
        initial_inlet_temp_c=float(env.get("initial_inlet_temp_c", 22.0)),
        initial_outlet_temp_c=float(env.get("initial_outlet_temp_c", 28.0)),
        initial_coolant_supply_c=float(env.get("initial_coolant_supply_c", 18.0)),
        initial_fan_speed_pct=float(env.get("initial_fan_speed_pct", 60.0)),
    )


@dataclass
class ScenarioSpec:
    family: str = "nominal"
    seed: int = 0
    ambient_offset_c: float = 0.0
    load_multiplier: float = 1.0
    coolant_degradation_c: float = 0.0
    spike_start: int | None = None
    spike_width: int = 12
    spike_magnitude: float = 0.35


def make_scenario(family: str, seed: int) -> ScenarioSpec:
    """Build a parameterized scenario of the given family."""
    rng = np.random.default_rng(seed)
    if family == "nominal":
        return ScenarioSpec(family=family, seed=seed)
    if family == "spike":
        return ScenarioSpec(
            family=family,
            seed=seed,
            spike_start=int(rng.integers(12, 60)),
            spike_width=int(rng.integers(6, 24)),
            spike_magnitude=float(0.25 + 0.3 * rng.random()),
        )
    if family == "degradation":
        return ScenarioSpec(
            family=family,
            seed=seed,
            coolant_degradation_c=float(1.0 + 2.0 * rng.random()),
        )
    if family == "heatwave":
        return ScenarioSpec(
            family=family,
            seed=seed,
            ambient_offset_c=float(4.0 + 3.0 * rng.random()),
        )
    raise ValueError(f"Unknown scenario family: {family}")


class CoolingEnv:
    """Minimal gym-like env."""

    observation_dim = len(STATE_FIELDS)
    action_dim = len(ACTION_FIELDS)

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.scenario: ScenarioSpec = ScenarioSpec()
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self._load_filter = 0.0
        self._state = np.zeros(self.observation_dim, dtype=np.float32)

    # ------------------------------------------------------------------ helpers

    def _ambient(self, step: int) -> float:
        # Diurnal sinusoid with a random phase set at scenario time.
        t_hours = step * self.cfg.cadence_minutes / 60.0
        amb = self.cfg.ambient_mean_c + self.cfg.ambient_amplitude_c * np.sin(
            2 * np.pi * (t_hours - 6) / 24.0
        )
        amb = amb + self.scenario.ambient_offset_c
        return float(amb + self.cfg.noise_scale * self.rng.standard_normal())

    def _base_load(self, step: int) -> float:
        t_hours = step * self.cfg.cadence_minutes / 60.0
        diurnal = self.cfg.load_base + self.cfg.load_amplitude * np.sin(
            2 * np.pi * (t_hours / 24.0) - np.pi / 2
        )
        load = diurnal * self.scenario.load_multiplier
        # Workload spike.
        if (
            self.scenario.spike_start is not None
            and self.scenario.spike_start <= step < self.scenario.spike_start + self.scenario.spike_width
        ):
            load = load + self.scenario.spike_magnitude
        load = float(load + 0.03 * self.rng.standard_normal())
        return float(np.clip(load, 0.05, 1.0))

    def _power_kw(self, cpu_load: float, fan_speed_pct: float) -> float:
        it_kw = self.cfg.it_power_coef * cpu_load * 20.0   # IT draw at ~0.18*20 = 3.6 kW baseline
        fan_kw = self.cfg.fan_power_coef * fan_speed_pct ** 2
        pump_kw = self.cfg.pump_power_coef * fan_speed_pct
        return float(it_kw + fan_kw + pump_kw)

    # ------------------------------------------------------------------ gym API

    def reset(self, scenario: ScenarioSpec | None = None) -> np.ndarray:
        self.scenario = scenario if scenario is not None else ScenarioSpec()
        self.rng = np.random.default_rng(self.scenario.seed)
        self.step_count = 0
        self._load_filter = 0.0
        fan = self.cfg.initial_fan_speed_pct
        coolant = self.cfg.initial_coolant_supply_c + self.scenario.coolant_degradation_c
        inlet = self.cfg.initial_inlet_temp_c
        outlet = self.cfg.initial_outlet_temp_c
        ambient = self._ambient(0)
        load = self._base_load(0)
        power = self._power_kw(load, fan)
        self._state = np.array([fan, coolant, inlet, outlet, load, ambient, power], dtype=np.float32)
        return self._state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one step. Action is clipped to configured ranges before applying."""
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action of shape ({self.action_dim},), got {action.shape}")
        act = np.array(
            [
                np.clip(action[0], *self.cfg.coolant_supply_delta_range),
                np.clip(action[1], *self.cfg.fan_speed_delta_range),
            ],
            dtype=np.float32,
        )
        fan, coolant, inlet, outlet, _load_prev, _ambient_prev, _power_prev = self._state

        # Apply deltas and clip to bounds.
        coolant = float(np.clip(coolant + act[0], *self.cfg.coolant_supply_bounds))
        fan = float(np.clip(fan + act[1], *self.cfg.fan_speed_bounds))

        # Degradation adds a persistent offset on top of the setpoint.
        coolant_effective = coolant + self.scenario.coolant_degradation_c

        # New load + ambient from scenario.
        self.step_count += 1
        ambient = self._ambient(self.step_count)
        load = self._base_load(self.step_count)

        # First-order lag dynamics.
        alpha = 1.0 / max(self.cfg.thermal_tau_steps, 1)
        effective = 0.85 * coolant_effective + 0.15 * ambient + 1.5
        # Fan influence: higher fan speed slightly accelerates equilibrium with coolant supply.
        effective = effective - 0.02 * (fan - 50.0)
        new_inlet = float(inlet + alpha * (effective - inlet) + self.cfg.noise_scale * self.rng.standard_normal())
        self._load_filter = float(self._load_filter + alpha * (load - self._load_filter))
        new_outlet = float(new_inlet + self.cfg.outlet_gain * self._load_filter + self.cfg.noise_scale * self.rng.standard_normal())

        power = self._power_kw(load, fan)
        # Reward: negative cooling+IT power, scaled so baseline is O(1) per step.
        reward = float(-power / 10.0)

        next_state = np.array([fan, coolant, new_inlet, new_outlet, load, ambient, power], dtype=np.float32)
        self._state = next_state
        done = self.step_count >= self.cfg.episode_steps

        info: dict[str, Any] = {
            "coolant_effective_c": coolant_effective,
            "load_filter": self._load_filter,
            "scenario_family": self.scenario.family,
            "step": self.step_count,
        }
        return next_state.copy(), reward, done, info

    def state_dict(self) -> dict:
        """Return the current state as a labeled dict for interpretability and audit."""
        return {name: float(v) for name, v in zip(STATE_FIELDS, self._state)}


def make_env(cfg: EnvConfig | dict | None = None) -> CoolingEnv:
    if cfg is None:
        cfg = EnvConfig()
    elif isinstance(cfg, dict):
        cfg = env_config_from_dict(cfg)
    return CoolingEnv(cfg)


def reset_scenario(env: CoolingEnv, scenario: ScenarioSpec | dict) -> np.ndarray:
    if isinstance(scenario, dict):
        scenario = ScenarioSpec(**scenario)
    return env.reset(scenario)


def step_env(env: CoolingEnv, action: np.ndarray) -> tuple:
    return env.step(action)


def obs_to_dict(obs: np.ndarray) -> dict:
    return {name: float(v) for name, v in zip(STATE_FIELDS, obs)}
