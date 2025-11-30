from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .outputs import SimulationOutputs


@dataclass
class SensitivityEntry:
    name: str
    corr_runtime: float
    corr_success: float
    delta_runtime_p90_p10: float


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def _compute_sensitivity_1d(
    name: str, x: np.ndarray, rt: np.ndarray, success: np.ndarray
) -> SensitivityEntry:
    x = np.asarray(x, dtype=float)
    rt = np.asarray(rt, dtype=float)
    success = np.asarray(success, dtype=float)

    corr_rt = _corr(rt, x)
    corr_succ = _corr(success, x)

    if x.size == 0:
        delta = 0.0
    else:
        p10 = float(np.percentile(x, 10))
        p90 = float(np.percentile(x, 90))
        low_mask = x <= p10
        high_mask = x >= p90
        if np.any(low_mask) and np.any(high_mask):
            rt_low = float(np.mean(rt[low_mask]))
            rt_high = float(np.mean(rt[high_mask]))
            delta = rt_high - rt_low
        else:
            delta = 0.0

    return SensitivityEntry(
        name=name,
        corr_runtime=corr_rt,
        corr_success=corr_succ,
        delta_runtime_p90_p10=delta,
    )


def compute_sensitivity_summary(outputs: SimulationOutputs) -> List[SensitivityEntry]:
    """Compute simple sensitivity metrics for each uncertainty variable."""
    rt = outputs.runtimes_hours
    success = outputs.success_flags

    entries: List[SensitivityEntry] = []

    if outputs.temps_C is not None:
        entries.append(
            _compute_sensitivity_1d("Temperature [°C]", outputs.temps_C, rt, success)
        )
    if outputs.years_in_field is not None:
        entries.append(
            _compute_sensitivity_1d(
                "Years in Field [yr]", outputs.years_in_field, rt, success
            )
        )
    if outputs.cap_unit_deltas is not None:
        entries.append(
            _compute_sensitivity_1d(
                "ΔCapacity (fraction)", outputs.cap_unit_deltas, rt, success
            )
        )
    if outputs.R_unit_deltas is not None:
        entries.append(
            _compute_sensitivity_1d(
                "ΔResistance (fraction)", outputs.R_unit_deltas, rt, success
            )
        )
    if outputs.tx_events_per_day is not None:
        entries.append(
            _compute_sensitivity_1d(
                "TX events per day", outputs.tx_events_per_day, rt, success
            )
        )
    elif outputs.tx_residency is not None:
        entries.append(
            _compute_sensitivity_1d(
                "TX Duty Fraction", outputs.tx_residency, rt, success
            )
        )

    entries.sort(key=lambda e: abs(e.corr_runtime), reverse=True)
    return entries
