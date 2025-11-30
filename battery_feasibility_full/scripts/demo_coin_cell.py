from __future__ import annotations

import numpy as np

from battery_feasibility.models.battery import CoinCellParams, OcvLut, ImpedanceLut
from battery_feasibility.models.load import CoinCellLoadProfile, LoadModeConfig
from battery_feasibility.models.uncertainty import TemperatureConfig, AgingConfig, VariationConfig
from battery_feasibility.simulation.monte_carlo import CoinCellSimConfig, run_coin_cell_feasibility


def make_dummy_coin_cell() -> CoinCellParams:
    # Simple, not‑realistic toy curves for demonstration.
    soc = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    voc = np.array([3.1, 3.05, 3.0, 2.98, 2.95, 2.9, 2.8, 2.7, 2.6, 2.4, 2.0])
    dcr = np.array([60e-3, 60e-3, 65e-3, 70e-3, 75e-3, 80e-3, 90e-3, 110e-3, 130e-3, 160e-3, 200e-3])
    acr = dcr * 0.6  # somewhat lower than DCR for short pulses

    ocv_lut = OcvLut(soc=soc, voc=voc)
    imp_lut = ImpedanceLut(soc=soc, dcr_ohm=dcr, acr_ohm=acr)

    return CoinCellParams(
        model_name="CR2032_dummy",
        nominal_capacity_mAh=220.0,
        nominal_voltage_V=3.0,
        ocv_lut=ocv_lut,
        impedance_lut=imp_lut,
        temp_ref_C=25.0,
    )


def make_dummy_load_profile() -> CoinCellLoadProfile:
    # Example system: 1 % TX, 99 % idle. Currents at 3.0 V.
    modes = [
        LoadModeConfig(
            name="TX",
            I_ref_dc_mA=25.0,   # average during TX burst
            k_dc=0.5,
            I_ref_peak_mA=60.0,  # short peak
            k_peak=0.5,
            residency=0.01,
            V_ref=3.0,
        ),
        LoadModeConfig(
            name="Idle",
            I_ref_dc_mA=5.0,
            k_dc=0.3,
            I_ref_peak_mA=5.0,
            k_peak=0.3,
            residency=0.99,
            V_ref=3.0,
        ),
    ]
    return CoinCellLoadProfile(modes=modes)


def main() -> None:
    battery = make_dummy_coin_cell()
    load_profile = make_dummy_load_profile()

    temp_cfg = TemperatureConfig(
        mode="distribution",
        dist_type="normal",
        mean_C=25.0,
        sigma_C=5.0,
        min_C=-20.0,
        max_C=60.0,
    )

    aging_cfg = AgingConfig(years_nominal=2.0, sigma_years=0.0)

    variation_cfg = VariationConfig(
        sigma_cap_pct=3.0,
        sigma_R_pct=5.0,
    )

    sim_cfg = CoinCellSimConfig(
        battery=battery,
        load=load_profile,
        temperature=temp_cfg,
        aging=aging_cfg,
        variation=variation_cfg,
        target_runtime_hours=24.0,
        target_lifetime_years=2.0,
        monte_carlo_samples=1000,
        system_dcr_ohm=0.05,   # 50 mΩ of system resistance
        system_acr_ohm=0.05,
        system_cutoff_voltage_V=2.0,  # system minimum operating voltage
        soc_step=0.01,
    )

    outputs = run_coin_cell_feasibility(sim_cfg, seed=42)
    print(outputs.to_dict())


if __name__ == "__main__":
    main()
