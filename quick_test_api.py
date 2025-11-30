from __future__ import annotations

from battery_feasibility_full.api import (
    UserConfig,
    LoadConfig,
    UncertaintyConfig,
    run_battery_feasibility,
)


def main() -> None:
    # Build load and uncertainty configs using fields from api.py dataclasses.
    load_cfg = LoadConfig(
        tx_dc_mA=20.0,
        tx_peak_mA=35.0,
        idle_dc_mA=0.5,
        idle_peak_mA=1.0,
        v_ref=None,  # use pack nominal voltage
        tx_events_per_day=200.0,
        tx_events_sigma=20.0,
        tx_event_duration_s=0.05,
        tx_events_min=0.0,
        tx_events_max=1000.0,
        tx_max_duty=0.5,
    )

    uncertainty_cfg = UncertaintyConfig(
        temp_mode="distribution",
        temp_mean_C=25.0,
        temp_sigma_C=5.0,
        temp_min_C=-10.0,
        temp_max_C=50.0,
        aging_years=1.5,
        aging_sigma_years=0.5,
        sigma_capacity_pct=3.0,
        sigma_R_pct=5.0,
    )

    user_cfg = UserConfig(
        battery_model_name="CR2032_lab_corrected",
        series_count=2,
        parallel_count=1,
        load=load_cfg,
        uncertainty=uncertainty_cfg,
        target_runtime_hours=24.0,
        target_lifetime_years=1.0,
        monte_carlo_samples=400,
        system_dcr_ohm=0.05,
        system_acr_ohm=0.0,
        system_cutoff_voltage_V=2.0,
        soc_step=0.01,
    )

    result = run_battery_feasibility(user_cfg)

    print("Feasibility probability:", f"{result.feasibility_probability:.3f}")
    print("Mean runtime (h):", f"{result.runtime_mean_hours:.2f}")
    print(
        "Runtime P10 / P50 / P90 (h):",
        f"{result.runtime_p10_hours:.2f} / {result.runtime_p50_hours:.2f} / {result.runtime_p90_hours:.2f}",
    )
    print("Number of samples:", len(result.runtime_hours))


if __name__ == "__main__":
    main()
