from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class LoadModeConfig:
    """One operating mode of the system (TX, idle, sleep, etc.).

    The current model is voltage‑dependent:

        I_dc(V)   = I_ref_dc_mA   * 1e-3 * (V / V_ref) ** k_dc
        I_peak(V) = I_ref_peak_mA * 1e-3 * (V / V_ref) ** k_peak

    residency is the fraction of time spent in this mode (0..1).
    """

    name: str
    I_ref_dc_mA: float
    k_dc: float
    I_ref_peak_mA: float
    k_peak: float
    residency: float
    V_ref: float = 3.0

    def dc_current_A(self, V: float) -> float:
        if V <= 0.0 or self.I_ref_dc_mA <= 0.0:
            return 0.0
        return float(self.I_ref_dc_mA * 1e-3 * (V / self.V_ref) ** self.k_dc)

    def peak_current_A(self, V: float) -> float:
        if V <= 0.0 or self.I_ref_peak_mA <= 0.0:
            return 0.0
        return float(self.I_ref_peak_mA * 1e-3 * (V / self.V_ref) ** self.k_peak)


@dataclass
class CoinCellLoadProfile:
    """Full load profile for a coin‑cell system, consisting of several modes.

    The runtime calculation uses:
    - average DC current built from mode residencies and dc_current_A at a
      representative voltage, and
    - brownout feasibility using the worst‑case DC and worst‑case peak mode
      at each OCV in the SOC sweep.
    """

    modes: List[LoadModeConfig] = field(default_factory=list)

    def _normalized_residencies(self) -> np.ndarray:
        if not self.modes:
            return np.zeros(0, dtype=float)
        res = np.array([max(0.0, m.residency) for m in self.modes], dtype=float)
        s = float(res.sum())
        if s <= 0.0:
            # If all zeros, treat as equal share to avoid division by zero.
            res[:] = 1.0 / len(self.modes)
        else:
            res /= s
        return res

    # --- brownout helper ---

    def currents_for_voltage(self, V: float):
        """Return arrays (I_dc, I_peak) for all modes at voltage V (in volts).

        Both arrays are in amperes. If there are no modes, returns empty arrays.
        """
        if not self.modes:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        I_dc = np.array([m.dc_current_A(V) for m in self.modes], dtype=float)
        I_pk = np.array([m.peak_current_A(V) for m in self.modes], dtype=float)
        return I_dc, I_pk

    # --- average current helper ---

    def average_current_A(self, V_avg: float) -> float:
        """Compute average DC current in amperes using mode residencies.

        Peaks are intentionally ignored here; they are handled only in the
        brownout headroom calculation.
        """
        if not self.modes:
            return 0.0

        res = self._normalized_residencies()
        I_dc_modes = np.array([m.dc_current_A(V_avg) for m in self.modes], dtype=float)
        return float(np.sum(res * I_dc_modes))
