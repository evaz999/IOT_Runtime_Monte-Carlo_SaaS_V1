from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SimulationOutputs:
    """Container for Monte‑Carlo simulation results and diagnostics."""

    runtimes_hours: np.ndarray
    brownout_flags: np.ndarray
    success_flags: np.ndarray
    target_runtime_hours: float

    # Diagnostics for uncertainty sampling
    temps_C: Optional[np.ndarray] = None
    years_in_field: Optional[np.ndarray] = None
    cap_unit_deltas: Optional[np.ndarray] = None
    R_unit_deltas: Optional[np.ndarray] = None
    tx_residency: Optional[np.ndarray] = None  # TX duty fraction per sample (if using tx_events)
    tx_events_per_day: Optional[np.ndarray] = None  # TX events/day per sample (if using tx_events)

    def feasibility_probability(self) -> float:
        """Fraction of samples that meet or exceed target runtime."""
        if self.success_flags.size == 0:
            return 0.0
        return float(np.mean(self.success_flags))

    def runtime_stats(self) -> Dict[str, float]:
        """Basic runtime statistics (mean / P10 / P50 / P90)."""
        rt = self.runtimes_hours
        if rt.size == 0:
            return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "mean": float(np.mean(rt)),
            "p10": float(np.percentile(rt, 10)),
            "p50": float(np.percentile(rt, 50)),
            "p90": float(np.percentile(rt, 90)),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Small JSON‑friendly summary of key metrics."""
        stats = self.runtime_stats()
        return {
            "target_runtime_hours": float(self.target_runtime_hours),
            "feasibility_probability": self.feasibility_probability(),
            "runtime_mean_hours": stats["mean"],
            "runtime_p10_hours": stats["p10"],
            "runtime_p50_hours": stats["p50"],
            "runtime_p90_hours": stats["p90"],
        }
