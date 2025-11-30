from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

import numpy as np
import random

from ..config import DEFAULTS
from ..models.battery import CoinCellParams, build_effective_battery
from ..models.load import CoinCellLoadProfile, LoadModeConfig
from ..models.uncertainty import TemperatureConfig, AgingConfig, VariationConfig, TxEventsConfig
from ..analytics.outputs import SimulationOutputs


@dataclass
class CoinCellSimConfig:
    """Top‑level configuration for a coin‑cell feasibility run."""

    battery: CoinCellParams
    load: CoinCellLoadProfile
    temperature: TemperatureConfig
    aging: AgingConfig
    variation: VariationConfig

    target_runtime_hours: float
    target_lifetime_years: float  # kept for API compatibility; aging uses its own config

    monte_carlo_samples: int = DEFAULTS.monte_carlo_samples

    # System‑level resistances outside the cell, added on top of battery DCR/ACR.
    system_dcr_ohm: float = 0.0
    system_acr_ohm: float = 0.0

    # System‑level minimum operating voltage (device cutoff threshold).
    # This is a system property, not a battery property.
    system_cutoff_voltage_V: float = 2.0

    # TX event-based traffic model (alternative to fixed duty cycle).
    # If provided, TX residency is sampled per Monte Carlo run instead of being fixed.
    tx_events: Optional[TxEventsConfig] = None

    # SOC sweep resolution (fraction of full range). 0.01 → 1 %.
    soc_step: float = 0.01

    # Representative voltage for average current calculation. If None, uses
    # battery.nominal_voltage_V.
    v_ref_for_Iavg: Optional[float] = None


def _soc_grid(step: float) -> np.ndarray:
    step = float(step)
    if step <= 0.0 or step > 1.0:
        step = 0.01
    n = int(round(1.0 / step))
    # from 1.0 down to 0.0 inclusive
    return np.linspace(1.0, 0.0, n + 1, dtype=float)


def _solve_single_sample(
    config: CoinCellSimConfig, rng: random.Random
) -> tuple[float, bool, bool, float, float, float, float, float, float]:
    """Solve brownout boundary and runtime for one Monte‑Carlo sample.

    Returns
    -------
    runtime_h : float
    brownout : bool
    success : bool
    T_C, years, cap_delta, R_delta, tx_residency, tx_events_per_day : diagnostics for this sample
    """
    # Sample uncertainties
    T_C = config.temperature.sample(rng)
    years = config.aging.sample_years(rng)
    cap_delta = config.variation.sample_cap_delta(rng)
    R_delta = config.variation.sample_R_delta(rng)

    # Sample TX residency from events model if provided, else use fixed load profile
    if config.tx_events is not None:
        tx_events_per_day, r_tx = config.tx_events.sample_tx_events(rng)
        # Assume load has at least 2 modes: [0]=TX, [1]=Idle
        # Create a modified load profile with sampled residencies
        modes = list(config.load.modes)
        if len(modes) >= 1:
            modes[0] = LoadModeConfig(
                name=modes[0].name,
                I_ref_dc_mA=modes[0].I_ref_dc_mA,
                k_dc=modes[0].k_dc,
                I_ref_peak_mA=modes[0].I_ref_peak_mA,
                k_peak=modes[0].k_peak,
                residency=float(r_tx),
                V_ref=modes[0].V_ref,
            )
        if len(modes) >= 2:
            modes[1] = LoadModeConfig(
                name=modes[1].name,
                I_ref_dc_mA=modes[1].I_ref_dc_mA,
                k_dc=modes[1].k_dc,
                I_ref_peak_mA=modes[1].I_ref_peak_mA,
                k_peak=modes[1].k_peak,
                residency=float(max(0.0, 1.0 - r_tx)),
                V_ref=modes[1].V_ref,
            )
        load_profile = CoinCellLoadProfile(modes=modes)
    else:
        r_tx = 0.0  # No TX events model; TX residency is implicit in fixed load
        tx_events_per_day = 0.0
        load_profile = config.load

    eff_batt = build_effective_battery(config.battery, T_C, years, cap_delta, R_delta)
    cutoff = config.system_cutoff_voltage_V

    socs = _soc_grid(config.soc_step)

    browned_out = False
    soc_min_usable = 1.0

    for soc in socs:
        voc = eff_batt.ocv(soc)
        R_dc_total = eff_batt.r_dc(soc) + config.system_dcr_ohm
        R_ac_total = eff_batt.r_ac(soc) + config.system_acr_ohm

        I_dc_vals, I_peak_vals = load_profile.currents_for_voltage(voc)
        if I_dc_vals.size == 0:
            # No load at all -> treat as success with very large runtime.
            runtime_h = config.target_runtime_hours * 10.0
            return runtime_h, False, True, T_C, years, cap_delta, R_delta, r_tx, tx_events_per_day

        # Find modes with highest DC and highest peak currents at this voltage.
        m_dc = int(np.argmax(I_dc_vals))
        m_pk = int(np.argmax(I_peak_vals))

        def v_term_for_idx(idx: int) -> float:
            I_dc = I_dc_vals[idx]
            I_peak = I_peak_vals[idx]
            return float(voc - I_dc * R_dc_total - (I_peak - I_dc) * R_ac_total)

        if m_dc == m_pk:
            V_term_worst = v_term_for_idx(m_dc)
        else:
            V_term_dc = v_term_for_idx(m_dc)
            V_term_pk = v_term_for_idx(m_pk)
            V_term_worst = min(V_term_dc, V_term_pk)

        if V_term_worst < cutoff:
            browned_out = True
            break

        soc_min_usable = soc

    usable_fraction = max(0.0, 1.0 - soc_min_usable)
    usable_capacity_mAh = eff_batt.capacity_mAh * usable_fraction

    # Average current (A) based on residencies and DC currents only.
    V_avg = config.v_ref_for_Iavg if config.v_ref_for_Iavg is not None else config.battery.nominal_voltage_V
    I_avg_A = load_profile.average_current_A(V_avg)

    if I_avg_A <= 0.0:
        # No net energy draw → arbitrarily large runtime.
        runtime_h = config.target_runtime_hours * 10.0
        success = True
        brownout = browned_out
    else:
        runtime_h = (usable_capacity_mAh / 1000.0) / I_avg_A  # mAh → Ah
        success = runtime_h >= config.target_runtime_hours
        brownout = browned_out

    return float(runtime_h), bool(brownout), bool(success), float(T_C), float(years), float(cap_delta), float(R_delta), float(r_tx), float(tx_events_per_day)


def run_coin_cell_feasibility(
    config: CoinCellSimConfig,
    seed: Optional[int] = None,
) -> SimulationOutputs:
    """Run the Monte‑Carlo feasibility analysis.

    Returns a :class:`SimulationOutputs` with runtime distribution,
    success / brownout flags, and diagnostics for uncertainties.
    """
    rng = random.Random(seed)
    N = int(config.monte_carlo_samples)

    runtimes = np.zeros(N, dtype=float)
    brownouts = np.zeros(N, dtype=bool)
    successes = np.zeros(N, dtype=bool)

    temps_C = np.zeros(N, dtype=float)
    years_in_field = np.zeros(N, dtype=float)
    cap_unit_deltas = np.zeros(N, dtype=float)
    R_unit_deltas = np.zeros(N, dtype=float)
    tx_residencies = np.zeros(N, dtype=float) if config.tx_events is not None else None
    tx_events_per_day = np.zeros(N, dtype=float) if config.tx_events is not None else None

    for i in range(N):
        rt_h, bo, succ, T_C, years, cap_d, R_d, tx_res, tx_evt = _solve_single_sample(config, rng)
        runtimes[i] = rt_h
        brownouts[i] = bo
        successes[i] = succ

        temps_C[i] = T_C
        years_in_field[i] = years
        cap_unit_deltas[i] = cap_d
        R_unit_deltas[i] = R_d
        if tx_residencies is not None:
            tx_residencies[i] = tx_res
        if tx_events_per_day is not None:
            tx_events_per_day[i] = tx_evt

    return SimulationOutputs(
        runtimes_hours=runtimes,
        brownout_flags=brownouts,
        success_flags=successes,
        target_runtime_hours=config.target_runtime_hours,
        temps_C=temps_C,
        years_in_field=years_in_field,
        cap_unit_deltas=cap_unit_deltas,
        R_unit_deltas=R_unit_deltas,
        tx_residency=tx_residencies,
        tx_events_per_day=tx_events_per_day,
    )
