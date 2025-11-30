from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import random
import numpy as np


TemperatureMode = Literal["single", "distribution"]
DistributionType = Literal["uniform", "normal"]
TxEventsDistType = Literal["fixed", "normal", "poisson"]


@dataclass
class TemperatureConfig:
    """Configuration for sampling operating temperature."""

    mode: TemperatureMode = "single"
    value_C: float | None = None

    # Used when mode == "distribution"
    dist_type: DistributionType = "normal"
    mean_C: float = 25.0
    sigma_C: float = 5.0
    min_C: float = -40.0
    max_C: float = 85.0

    def sample(self, rng: random.Random) -> float:
        if self.mode == "single":
            if self.value_C is None:
                return self.mean_C
            return float(self.value_C)

        # distribution mode
        if self.dist_type == "normal":
            T = rng.gauss(self.mean_C, self.sigma_C)
        elif self.dist_type == "uniform":
            T = rng.uniform(self.min_C, self.max_C)
        else:
            raise ValueError(f"Unsupported dist_type={self.dist_type!r}")

        # Clip to physical bounds
        if T < self.min_C:
            T = self.min_C
        if T > self.max_C:
            T = self.max_C
        return float(T)


@dataclass
class AgingConfig:
    """Configuration for calendar aging in years."""

    years_nominal: float = 3.0
    sigma_years: float = 0.0  # 0 => deterministic

    def sample_years(self, rng: random.Random) -> float:
        if self.sigma_years <= 0.0:
            return float(max(0.0, self.years_nominal))
        years = rng.gauss(self.years_nominal, self.sigma_years)
        return float(max(0.0, years))


@dataclass
class VariationConfig:
    """Cell‑to‑cell variation configuration.

    sigma_cap_pct / sigma_R_pct represent 1σ spread as a percentage.
    The sampling returns *fractional* deltas (e.g. +0.05 → +5 %).
    """

    sigma_cap_pct: float = 3.0
    sigma_R_pct: float = 5.0

    def sample_cap_delta(self, rng: random.Random) -> float:
        if self.sigma_cap_pct <= 0.0:
            return 0.0
        sigma = self.sigma_cap_pct / 100.0
        return float(rng.gauss(0.0, sigma))

    def sample_R_delta(self, rng: random.Random) -> float:
        if self.sigma_R_pct <= 0.0:
            return 0.0
        sigma = self.sigma_R_pct / 100.0
        return float(rng.gauss(0.0, sigma))


@dataclass
class TxEventsConfig:
    """Configuration for TX event-based traffic model.

    Instead of a fixed TX duty cycle, TX activity is defined by:
    - N_events_per_day: number of TX events in a 24-hour period.
    - event_duration_s: duration of each TX event (seconds).

    The TX duty fraction is computed as:
    r_tx = (N_events_per_day * event_duration_s) / 86400,
    clamped to [0, max_duty].

    This allows MC sampling of traffic intensity via N_events distribution.
    """

    dist_type: TxEventsDistType = "fixed"

    # Event duration (constant across samples for now)
    event_duration_s: float = 0.1

    # Mean and sigma for normal distribution or deterministic value for fixed.
    mean_events_per_day: float = 100.0
    sigma_events_per_day: float = 0.0  # 0 => deterministic

    # Bounds for clipping sampled values
    min_events_per_day: float = 0.0
    max_events_per_day: float = 1e6

    # Upper bound on TX duty fraction (safety/realism constraint)
    max_duty: float = 0.9

    def sample_tx_events(self, rng: random.Random) -> tuple[float, float]:
        """Sample TX events/day and resulting duty fraction for this MC run.

        Returns (events_per_day, r_tx)
        """
        # Sample N_events_per_day based on distribution type
        if self.dist_type == "fixed":
            n_events = self.mean_events_per_day
        elif self.dist_type == "normal":
            n_events = rng.gauss(self.mean_events_per_day, self.sigma_events_per_day)
        elif self.dist_type == "poisson":
            # Use numpy poisson if available; otherwise fall back to normal approximation
            try:
                n_events = float(np.random.poisson(self.mean_events_per_day, random_state=rng.random()))
            except Exception:
                # Fallback: approximate Poisson with normal for large lambda
                n_events = rng.gauss(self.mean_events_per_day, np.sqrt(self.mean_events_per_day))
        else:
            raise ValueError(f"Unsupported dist_type={self.dist_type!r}")

        # Clip to valid range
        n_events = float(np.clip(n_events, self.min_events_per_day, self.max_events_per_day))

        # Compute TX duty fraction: (events/day * seconds/event) / seconds/day
        T_day_seconds = 86400.0
        r_tx_raw = (n_events * self.event_duration_s) / T_day_seconds

        # Clamp to [0, max_duty]
        r_tx = float(np.clip(r_tx_raw, 0.0, self.max_duty))

        return float(n_events), r_tx

    def sample_tx_residency(self, rng: random.Random) -> float:
        """Backward-compatible helper returning only duty fraction."""
        _, r_tx = self.sample_tx_events(rng)
        return r_tx
