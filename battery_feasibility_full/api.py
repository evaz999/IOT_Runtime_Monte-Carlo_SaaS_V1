from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from battery_feasibility.models.battery import (
    BatteryArchitecture,
    CoinCellParams,
    derive_pack_from_cell,
    load_battery_from_json,
)
from battery_feasibility.models.load import CoinCellLoadProfile, LoadModeConfig
from battery_feasibility.models.uncertainty import (
    AgingConfig,
    TemperatureConfig,
    TxEventsConfig,
    VariationConfig,
)
from battery_feasibility.simulation.monte_carlo import (
    CoinCellSimConfig,
    run_coin_cell_feasibility,
)


# --------- User input schema ---------


@dataclass
class LoadConfig:
    """Minimal load configuration for two-mode systems (TX + Idle)."""

    tx_dc_mA: float
    tx_peak_mA: float
    idle_dc_mA: float
    idle_peak_mA: float = 0.0
    v_ref: Optional[float] = None  # defaults to pack nominal V if None

    # TX traffic model (optional). If not provided, residencies must be baked into the currents elsewhere.
    tx_events_per_day: Optional[float] = None
    tx_events_sigma: float = 0.0
    tx_event_duration_s: float = 0.1
    tx_events_min: float = 0.0
    tx_events_max: float = 1e6
    tx_max_duty: float = 0.9


@dataclass
class UncertaintyConfig:
    """Uncertainty knobs for temperature, aging, and unit-to-unit variation."""

    temp_mode: str = "distribution"  # "single" or "distribution"
    temp_value_C: float = 25.0
    temp_mean_C: float = 25.0
    temp_sigma_C: float = 5.0
    temp_min_C: float = -40.0
    temp_max_C: float = 85.0

    aging_years: float = 1.0
    aging_sigma_years: float = 0.0

    sigma_capacity_pct: float = 3.0
    sigma_R_pct: float = 5.0


@dataclass
class UserConfig:
    """Top-level request for running a battery feasibility simulation."""

    battery_model_name: str
    series_count: int = 1
    parallel_count: int = 1

    load: LoadConfig = field(
        default_factory=lambda: LoadConfig(
            tx_dc_mA=0.0,
            tx_peak_mA=0.0,
            idle_dc_mA=0.0,
        )
    )
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    target_runtime_hours: float = 24.0
    target_lifetime_years: float = 1.0
    monte_carlo_samples: int = 1000

    system_dcr_ohm: float = 0.0
    system_acr_ohm: float = 0.0
    system_cutoff_voltage_V: float = 2.0
    soc_step: float = 0.01


@dataclass
class SimulationResult:
    feasibility_probability: float
    runtime_hours: np.ndarray
    runtime_mean_hours: float
    runtime_p10_hours: float
    runtime_p50_hours: float
    runtime_p90_hours: float


# --------- Helper functions ---------


def _find_battery_json(model_name: str) -> Path:
    """Locate a battery JSON by name in the project directory."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / model_name,
        base_dir / f"{model_name}.json",
        base_dir.parent / model_name,
        base_dir.parent / f"{model_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find battery JSON for model_name={model_name!r}")


def _build_load_profile(load_cfg: LoadConfig, v_ref_default: float) -> tuple[CoinCellLoadProfile, Optional[TxEventsConfig]]:
    """Construct load profile and optional TxEventsConfig."""
    v_ref = v_ref_default if load_cfg.v_ref is None else float(load_cfg.v_ref)
    tx_mode = LoadModeConfig(
        name="TX",
        I_ref_dc_mA=float(load_cfg.tx_dc_mA),
        k_dc=1.0,
        I_ref_peak_mA=float(load_cfg.tx_peak_mA),
        k_peak=1.0,
        residency=0.5,  # placeholder; residency adjusted if tx_events provided
        V_ref=v_ref,
    )
    idle_mode = LoadModeConfig(
        name="Idle",
        I_ref_dc_mA=float(load_cfg.idle_dc_mA),
        k_dc=1.0,
        I_ref_peak_mA=float(load_cfg.idle_peak_mA),
        k_peak=1.0,
        residency=0.5,
        V_ref=v_ref,
    )
    load_profile = CoinCellLoadProfile(modes=[tx_mode, idle_mode])

    tx_events_cfg: Optional[TxEventsConfig] = None
    if load_cfg.tx_events_per_day is not None:
        tx_events_cfg = TxEventsConfig(
            dist_type="normal" if load_cfg.tx_events_sigma > 0 else "fixed",
            event_duration_s=float(load_cfg.tx_event_duration_s),
            mean_events_per_day=float(load_cfg.tx_events_per_day),
            sigma_events_per_day=float(load_cfg.tx_events_sigma),
            min_events_per_day=float(load_cfg.tx_events_min),
            max_events_per_day=float(load_cfg.tx_events_max),
            max_duty=float(load_cfg.tx_max_duty),
        )

    return load_profile, tx_events_cfg


def _build_uncertainty(cfg: UncertaintyConfig) -> tuple[TemperatureConfig, AgingConfig, VariationConfig]:
    if cfg.temp_mode.lower() == "single":
        temp_cfg = TemperatureConfig(mode="single", value_C=float(cfg.temp_value_C))
    else:
        temp_cfg = TemperatureConfig(
            mode="distribution",
            dist_type="normal",
            mean_C=float(cfg.temp_mean_C),
            sigma_C=float(cfg.temp_sigma_C),
            min_C=float(cfg.temp_min_C),
            max_C=float(cfg.temp_max_C),
        )

    aging_cfg = AgingConfig(
        years_nominal=float(cfg.aging_years),
        sigma_years=float(cfg.aging_sigma_years),
    )

    variation_cfg = VariationConfig(
        sigma_cap_pct=float(cfg.sigma_capacity_pct),
        sigma_R_pct=float(cfg.sigma_R_pct),
    )

    return temp_cfg, aging_cfg, variation_cfg


# --------- Public API ---------


def run_battery_feasibility(user_config: UserConfig) -> SimulationResult:
    """High-level entry point to run Monte Carlo feasibility and return structured results."""
    # Load cell model
    json_path = _find_battery_json(user_config.battery_model_name)
    cell_params: CoinCellParams = load_battery_from_json(str(json_path))

    # Apply pack architecture
    arch = BatteryArchitecture(
        series_count=int(user_config.series_count),
        parallel_count=int(user_config.parallel_count),
    )
    pack_params = derive_pack_from_cell(cell_params, arch)

    # Build load and uncertainties
    load_profile, tx_events_cfg = _build_load_profile(
        user_config.load, v_ref_default=pack_params.nominal_voltage_V
    )
    temp_cfg, aging_cfg, variation_cfg = _build_uncertainty(user_config.uncertainty)

    sim_cfg = CoinCellSimConfig(
        battery=pack_params,
        load=load_profile,
        temperature=temp_cfg,
        aging=aging_cfg,
        variation=variation_cfg,
        tx_events=tx_events_cfg,
        target_runtime_hours=float(user_config.target_runtime_hours),
        target_lifetime_years=float(user_config.target_lifetime_years),
        monte_carlo_samples=int(user_config.monte_carlo_samples),
        system_dcr_ohm=float(user_config.system_dcr_ohm),
        system_acr_ohm=float(user_config.system_acr_ohm),
        system_cutoff_voltage_V=float(user_config.system_cutoff_voltage_V),
        soc_step=float(user_config.soc_step),
        v_ref_for_Iavg=None,
    )

    outputs = run_coin_cell_feasibility(sim_cfg)
    stats = outputs.runtime_stats()

    return SimulationResult(
        feasibility_probability=outputs.feasibility_probability(),
        runtime_hours=outputs.runtimes_hours,
        runtime_mean_hours=stats["mean"],
        runtime_p10_hours=stats["p10"],
        runtime_p50_hours=stats["p50"],
        runtime_p90_hours=stats["p90"],
    )
