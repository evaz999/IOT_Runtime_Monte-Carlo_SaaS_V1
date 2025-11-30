from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import json
import numpy as np


@dataclass
class OcvLut:
    """Lookup table for OCV vs SOC at reference temperature / beginning of life."""

    soc: np.ndarray  # shape (N,), values in [0, 1]
    voc: np.ndarray  # shape (N,), volts
    voc_stddev: Optional[np.ndarray] = None  # optional stddev vs SOC

    def voc_at(self, soc: float) -> float:
        """Return OCV at given SOC using linear interpolation."""
        s = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(s, self.soc, self.voc))

    def voc_stddev_at(self, soc: float) -> float:
        """Return OCV stddev at given SOC using linear interpolation if available."""
        if self.voc_stddev is None:
            return 0.0
        s = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(s, self.soc, self.voc_stddev))


@dataclass
class ImpedanceLut:
    """Lookup table for DC and AC resistance vs SOC at reference temperature / BOL."""

    soc: np.ndarray  # shape (N,), values in [0, 1]
    dcr_ohm: np.ndarray  # DC / long-term resistance (ohms)
    acr_ohm: np.ndarray  # AC / ~1 ms pulse resistance (ohms)
    dcr_min_ohm: Optional[np.ndarray] = None
    dcr_max_ohm: Optional[np.ndarray] = None
    acr_min_ohm: Optional[np.ndarray] = None
    acr_max_ohm: Optional[np.ndarray] = None

    def dcr_at(self, soc: float) -> float:
        s = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(s, self.soc, self.dcr_ohm))

    def acr_at(self, soc: float) -> float:
        s = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(s, self.soc, self.acr_ohm))


@dataclass
class CoinCellParams:
    """Static parameters and sensitivities for a given cell model.

    This is the "design-time" description of the cell at reference temperature
    and beginning of life. Aging, temperature, and unit-to-unit variation are
    applied on top of this via :func:`build_effective_battery`.
    """

    model_name: str
    nominal_capacity_mAh: float
    nominal_voltage_V: float

    ocv_lut: OcvLut
    impedance_lut: ImpedanceLut

    # Reference temperature for the LUTs
    temp_ref_C: float = 25.0

    # Capacity vs temperature: fractional change per °C relative to temp_ref_C.
    # Sign convention: positive => capacity increases with temperature.
    # Example: 0.003 -> +0.3 % / °C
    temp_capacity_coeff_per_C: float = 0.003

    # Resistance vs temperature: fractional change per °C relative to temp_ref_C.
    # Sign convention: negative => resistance decreases with temperature.
    # Example: -0.01 -> -1 % / °C
    temp_dcr_coeff_per_C: float = -0.01
    temp_acr_coeff_per_C: float = -0.01

    # Aging vs years: fractional change per year at reference conditions.
    # Example: 0.03 -> -3 % capacity per year, 0.08 -> +8 % resistance per year.
    capacity_aging_coeff_per_year: float = 0.03
    dcr_aging_coeff_per_year: float = 0.08
    acr_aging_coeff_per_year: float = 0.05

    # Optional small OCV tilt with temperature (V / °C).
    ocv_temp_coeff_per_C: float = 0.0

    # Unit-to-unit variation (1-sigma, as percentage points).
    # These describe min/max resistance and OCV variation at reference conditions.
    # Converted from min/max in JSON to sigma for sampling.
    sigma_capacity_pct: float = 3.0  # 1-sigma unit capacity variation (%)
    sigma_R_pct: float = 5.0  # 1-sigma unit resistance variation (%)


@dataclass
class EffectiveBatteryView:
    """Per-sample effective view of the battery for simulation.

    All aging / temperature / unit-to-unit variation has been folded into
    this object. The simulation only needs to call ocv(soc), r_dc(soc),
    and r_ac(soc), plus use capacity_mAh.
    """

    capacity_mAh: float
    ocv: Callable[[float], float]
    r_dc: Callable[[float], float]
    r_ac: Callable[[float], float]


@dataclass
class BatteryArchitecture:
    """Pack architecture: series (y) x parallel (x)."""

    series_count: int = 1
    parallel_count: int = 1


def build_effective_battery(
    params: CoinCellParams,
    T_C: float,
    years: float,
    cap_unit_delta: float,
    R_unit_delta: float,
) -> EffectiveBatteryView:
    """Combine base model with temperature, aging and unit-variation."""
    # ---- capacity scaling ----
    f_age_cap = 1.0 - params.capacity_aging_coeff_per_year * max(years, 0.0)
    f_age_cap = max(0.2, f_age_cap)

    dT = T_C - params.temp_ref_C
    f_temp_cap = 1.0 + params.temp_capacity_coeff_per_C * dT
    f_temp_cap = max(0.2, f_temp_cap)

    f_unit_cap = 1.0 + cap_unit_delta
    f_unit_cap = max(0.5, f_unit_cap)

    capacity_mAh = params.nominal_capacity_mAh * f_age_cap * f_temp_cap * f_unit_cap

    # ---- resistance scaling ----
    f_age_Rdc = 1.0 + params.dcr_aging_coeff_per_year * max(years, 0.0)
    f_age_Rdc = max(0.1, f_age_Rdc)
    f_age_Rac = 1.0 + params.acr_aging_coeff_per_year * max(years, 0.0)
    f_age_Rac = max(0.1, f_age_Rac)

    f_temp_Rdc = 1.0 + params.temp_dcr_coeff_per_C * dT
    f_temp_Rdc = max(0.1, f_temp_Rdc)
    f_temp_Rac = 1.0 + params.temp_acr_coeff_per_C * dT
    f_temp_Rac = max(0.1, f_temp_Rac)

    f_unit_R = 1.0 + R_unit_delta
    f_unit_R = max(0.1, f_unit_R)

    def ocv_eff(soc: float) -> float:
        base = params.ocv_lut.voc_at(soc)
        return float(base + params.ocv_temp_coeff_per_C * dT)

    def r_dc_eff(soc: float) -> float:
        base = params.impedance_lut.dcr_at(soc)
        return float(base * f_age_Rdc * f_temp_Rdc * f_unit_R)

    def r_ac_eff(soc: float) -> float:
        base = params.impedance_lut.acr_at(soc)
        return float(base * f_age_Rac * f_temp_Rac * f_unit_R)

    return EffectiveBatteryView(
        capacity_mAh=float(capacity_mAh),
        ocv=ocv_eff,
        r_dc=r_dc_eff,
        r_ac=r_ac_eff,
    )


# ---- JSON loader helpers ----

JsonPath = Union[str, Path]


def _multiplier_dict_to_linear_coeff(
    multiplier_dict: dict[str, float],
    ref_temp_C: float = 25.0,
) -> float:
    """Convert temperature multiplier table to linear coefficient per °C."""
    if not multiplier_dict:
        return 0.0

    temps = np.array(sorted([float(t) for t in multiplier_dict.keys()]))
    mults = np.array([multiplier_dict[str(int(t))] for t in temps])

    closest_idx = int(np.argmin(np.abs(temps - ref_temp_C)))

    if len(temps) < 2:
        return 0.0

    if closest_idx == 0:
        dT = temps[1] - temps[0]
        dm = mults[1] - mults[0]
    elif closest_idx == len(temps) - 1:
        dT = temps[-1] - temps[-2]
        dm = mults[-1] - mults[-2]
    else:
        dT_left = temps[closest_idx] - temps[closest_idx - 1]
        dm_left = mults[closest_idx] - mults[closest_idx - 1]
        dT_right = temps[closest_idx + 1] - temps[closest_idx]
        dm_right = mults[closest_idx + 1] - mults[closest_idx]
        dT = (dT_left + dT_right) / 2.0
        dm = (dm_left / dT_left + dm_right / dT_right) / 2.0 * dT

    M_ref = mults[closest_idx]
    if M_ref <= 0.0:
        return 0.0
    return float((dm / M_ref) / dT) if dT != 0 else 0.0


def load_battery_from_json(path: JsonPath) -> CoinCellParams:
    """Load a CoinCellParams instance from a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    soc = np.asarray(cfg["soc_points"], dtype=float)
    ocv = np.asarray(cfg["ocv_V"], dtype=float)
    ocv_stddev = (
        np.asarray(cfg["ocv_stddev_V"], dtype=float) if "ocv_stddev_V" in cfg else None
    )

    # ---- Resistance LUT: try new format first, fall back to old ----
    if "dcr_ohm_typ" in cfg:
        dcr = np.asarray(cfg["dcr_ohm_typ"], dtype=float)
        acr = np.asarray(cfg["acr_ohm_typ"], dtype=float)
        dcr_min = (
            np.asarray(cfg["dcr_ohm_min"], dtype=float) if "dcr_ohm_min" in cfg else None
        )
        dcr_max = (
            np.asarray(cfg["dcr_ohm_max"], dtype=float) if "dcr_ohm_max" in cfg else None
        )
        acr_min = (
            np.asarray(cfg["acr_ohm_min"], dtype=float) if "acr_ohm_min" in cfg else None
        )
        acr_max = (
            np.asarray(cfg["acr_ohm_max"], dtype=float) if "acr_ohm_max" in cfg else None
        )
    else:
        dcr = np.asarray(cfg.get("dcr_ohm", [0.05] * len(soc)), dtype=float)
        acr = np.asarray(cfg.get("acr_ohm", [0.05] * len(soc)), dtype=float)
        dcr_min = None
        dcr_max = None
        acr_min = None
        acr_max = None

    ocv_lut = OcvLut(soc=soc, voc=ocv, voc_stddev=ocv_stddev)
    imp_lut = ImpedanceLut(
        soc=soc,
        dcr_ohm=dcr,
        acr_ohm=acr,
        dcr_min_ohm=dcr_min,
        dcr_max_ohm=dcr_max,
        acr_min_ohm=acr_min,
        acr_max_ohm=acr_max,
    )

    temp_ref_C = float(cfg.get("temp_ref_C", 25.0))

    # ---- Temperature coefficients: try new multiplier format first ----
    if "temp_dcr_multiplier" in cfg:
        temp_dcr_coeff = _multiplier_dict_to_linear_coeff(
            cfg["temp_dcr_multiplier"], temp_ref_C
        )
    else:
        temp_dcr_coeff = float(cfg.get("temp_dcr_coeff_per_C", -0.01))

    if "temp_capacity_multiplier" in cfg:
        temp_cap_coeff = _multiplier_dict_to_linear_coeff(
            cfg["temp_capacity_multiplier"], temp_ref_C
        )
    else:
        temp_cap_coeff = float(cfg.get("temp_capacity_coeff_per_C", 0.003))

    temp_acr_coeff = float(cfg.get("temp_acr_coeff_per_C", temp_dcr_coeff))

    return CoinCellParams(
        model_name=cfg["model_name"],
        nominal_capacity_mAh=float(cfg["nominal_capacity_mAh"]),
        nominal_voltage_V=float(cfg["nominal_voltage_V"]),
        ocv_lut=ocv_lut,
        impedance_lut=imp_lut,
        temp_ref_C=temp_ref_C,
        temp_capacity_coeff_per_C=temp_cap_coeff,
        temp_dcr_coeff_per_C=temp_dcr_coeff,
        temp_acr_coeff_per_C=temp_acr_coeff,
        capacity_aging_coeff_per_year=float(cfg.get("capacity_aging_coeff_per_year", 0.01)),
        dcr_aging_coeff_per_year=float(cfg.get("dcr_aging_coeff_per_year", 0.05)),
        acr_aging_coeff_per_year=float(cfg.get("acr_aging_coeff_per_year", 0.04)),
        ocv_temp_coeff_per_C=float(cfg.get("ocv_temp_coeff_per_C", 0.0)),
        sigma_capacity_pct=float(cfg.get("sigma_capacity_pct", 3.0)),
        sigma_R_pct=float(cfg.get("sigma_R_pct", 5.0)),
    )


def derive_pack_from_cell(params: CoinCellParams, arch: BatteryArchitecture) -> CoinCellParams:
    """Derive a pack-level CoinCellParams from a single-cell model and architecture."""
    y = max(1, int(arch.series_count))
    x = max(1, int(arch.parallel_count))

    # Scale LUTs
    ocv_scaled = params.ocv_lut.voc * y
    ocv_std_scaled = params.ocv_lut.voc_stddev * y if params.ocv_lut.voc_stddev is not None else None

    dcr_scale = y / x
    acr_scale = y / x
    imp = params.impedance_lut
    dcr_scaled = imp.dcr_ohm * dcr_scale
    acr_scaled = imp.acr_ohm * acr_scale
    dcr_min_scaled = imp.dcr_min_ohm * dcr_scale if imp.dcr_min_ohm is not None else None
    dcr_max_scaled = imp.dcr_max_ohm * dcr_scale if imp.dcr_max_ohm is not None else None
    acr_min_scaled = imp.acr_min_ohm * acr_scale if imp.acr_min_ohm is not None else None
    acr_max_scaled = imp.acr_max_ohm * acr_scale if imp.acr_max_ohm is not None else None

    ocv_lut_pack = OcvLut(
        soc=params.ocv_lut.soc.copy(),
        voc=ocv_scaled,
        voc_stddev=ocv_std_scaled,
    )
    imp_lut_pack = ImpedanceLut(
        soc=imp.soc.copy(),
        dcr_ohm=dcr_scaled,
        acr_ohm=acr_scaled,
        dcr_min_ohm=dcr_min_scaled,
        dcr_max_ohm=dcr_max_scaled,
        acr_min_ohm=acr_min_scaled,
        acr_max_ohm=acr_max_scaled,
    )

    model_name = f"{params.model_name}_{y}S{x}P"
    return CoinCellParams(
        model_name=model_name,
        nominal_capacity_mAh=float(params.nominal_capacity_mAh * x),
        nominal_voltage_V=float(params.nominal_voltage_V * y),
        ocv_lut=ocv_lut_pack,
        impedance_lut=imp_lut_pack,
        temp_ref_C=params.temp_ref_C,
        temp_capacity_coeff_per_C=params.temp_capacity_coeff_per_C,
        temp_dcr_coeff_per_C=params.temp_dcr_coeff_per_C,
        temp_acr_coeff_per_C=params.temp_acr_coeff_per_C,
        capacity_aging_coeff_per_year=params.capacity_aging_coeff_per_year,
        dcr_aging_coeff_per_year=params.dcr_aging_coeff_per_year,
        acr_aging_coeff_per_year=params.acr_aging_coeff_per_year,
        ocv_temp_coeff_per_C=params.ocv_temp_coeff_per_C,
        sigma_capacity_pct=params.sigma_capacity_pct,
        sigma_R_pct=params.sigma_R_pct,
    )
