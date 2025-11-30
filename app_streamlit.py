from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from battery_feasibility_full.api import (
    LoadConfig,
    UncertaintyConfig,
    UserConfig,
    run_battery_feasibility,
)


def main() -> None:
    st.title("Battery Runtime & Brownout Feasibility (V1)")

    st.sidebar.header("Battery Selection")
    battery_options = [
        "CR2032_lab_corrected",
        "CR2450_lab_corrected",
        "CR2477_lab_corrected",
        "AA_alkaline_lab_validated",
    ]
    battery_model = st.sidebar.selectbox("Battery model JSON", battery_options, index=0)

    st.sidebar.header("Architecture (yS xP)")
    series_count = int(st.sidebar.number_input("Series count (y)", min_value=1, value=1, step=1))
    parallel_count = int(st.sidebar.number_input("Parallel count (x)", min_value=1, value=1, step=1))

    st.sidebar.header("Load Configuration")
    v_ref = st.sidebar.number_input("V_ref (0 = use pack nominal)", min_value=0.0, value=0.0, step=0.1)
    tx_dc_mA = st.sidebar.number_input("TX DC current (mA)", min_value=0.0, value=20.0, step=1.0)
    tx_peak_mA = st.sidebar.number_input("TX peak current (mA)", min_value=0.0, value=35.0, step=1.0)
    idle_dc_mA = st.sidebar.number_input("Idle DC current (mA)", min_value=0.0, value=0.5, step=0.1)
    idle_peak_mA = st.sidebar.number_input("Idle peak current (mA)", min_value=0.0, value=1.0, step=0.1)

    st.sidebar.subheader("TX Traffic (events/day model)")
    tx_events_per_day = st.sidebar.number_input("Mean events per day", min_value=0.0, value=200.0, step=10.0)
    tx_events_sigma = st.sidebar.number_input("Sigma events per day", min_value=0.0, value=20.0, step=5.0)
    tx_event_duration_s = st.sidebar.number_input("Event duration (seconds)", min_value=0.001, value=0.05, step=0.01, format="%.3f")
    tx_events_min = st.sidebar.number_input("Min events per day", min_value=0.0, value=0.0, step=10.0)
    tx_events_max = st.sidebar.number_input("Max events per day", min_value=0.0, value=1000.0, step=50.0)
    tx_max_duty = st.sidebar.number_input("Max TX duty fraction", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.sidebar.header("Uncertainty Configuration")
    temp_mode_choice = st.sidebar.radio("Temperature mode", ["distribution", "single"], index=0)
    if temp_mode_choice == "single":
        temp_value_C = st.sidebar.number_input("Temperature (°C)", value=25.0, step=1.0)
        temp_mean_C = 25.0
        temp_sigma_C = 0.0
        temp_min_C = temp_value_C
        temp_max_C = temp_value_C
    else:
        temp_value_C = 25.0
        temp_mean_C = st.sidebar.number_input("Temp mean (°C)", value=25.0, step=1.0)
        temp_sigma_C = st.sidebar.number_input("Temp sigma (°C)", value=5.0, step=0.5)
        temp_min_C = st.sidebar.number_input("Temp min (°C)", value=-10.0, step=1.0)
        temp_max_C = st.sidebar.number_input("Temp max (°C)", value=50.0, step=1.0)
    aging_years = st.sidebar.number_input("Aging years nominal", min_value=0.0, value=1.5, step=0.5)
    aging_sigma_years = st.sidebar.number_input("Aging sigma (years)", min_value=0.0, value=0.5, step=0.5)
    sigma_capacity_pct = st.sidebar.number_input("Sigma capacity (%)", min_value=0.0, value=3.0, step=0.5)
    sigma_R_pct = st.sidebar.number_input("Sigma resistance (%)", min_value=0.0, value=5.0, step=0.5)

    st.sidebar.header("Simulation")
    mc_samples = int(st.sidebar.number_input("Monte Carlo samples", min_value=1, value=400, step=50))
    target_runtime_hours = st.sidebar.number_input("Target runtime (hours)", min_value=0.1, value=24.0, step=1.0)
    system_cutoff_voltage_V = st.sidebar.number_input("System cutoff voltage (V)", min_value=0.1, value=2.0, step=0.1)
    system_dcr_ohm = st.sidebar.number_input("System DCR (ohm)", min_value=0.0, value=0.0, step=0.01)
    system_acr_ohm = st.sidebar.number_input("System ACR (ohm)", min_value=0.0, value=0.0, step=0.01)

    run_btn = st.button("Run Simulation")

    if run_btn:
        load_cfg = LoadConfig(
            tx_dc_mA=tx_dc_mA,
            tx_peak_mA=tx_peak_mA,
            idle_dc_mA=idle_dc_mA,
            idle_peak_mA=idle_peak_mA,
            v_ref=None if v_ref <= 0 else v_ref,
            tx_events_per_day=tx_events_per_day,
            tx_events_sigma=tx_events_sigma,
            tx_event_duration_s=tx_event_duration_s,
            tx_events_min=tx_events_min,
            tx_events_max=tx_events_max,
            tx_max_duty=tx_max_duty,
        )

        uncertainty_cfg = UncertaintyConfig(
            temp_mode=temp_mode_choice,
            temp_value_C=temp_value_C,
            temp_mean_C=temp_mean_C,
            temp_sigma_C=temp_sigma_C,
            temp_min_C=temp_min_C,
            temp_max_C=temp_max_C,
            aging_years=aging_years,
            aging_sigma_years=aging_sigma_years,
            sigma_capacity_pct=sigma_capacity_pct,
            sigma_R_pct=sigma_R_pct,
        )

        user_cfg = UserConfig(
            battery_model_name=battery_model,
            series_count=series_count,
            parallel_count=parallel_count,
            load=load_cfg,
            uncertainty=uncertainty_cfg,
            target_runtime_hours=target_runtime_hours,
            target_lifetime_years=1.0,
            monte_carlo_samples=mc_samples,
            system_dcr_ohm=system_dcr_ohm,
            system_acr_ohm=system_acr_ohm,
            system_cutoff_voltage_V=system_cutoff_voltage_V,
            soc_step=0.01,
        )

        try:
            outputs = run_battery_feasibility(user_cfg)
            st.subheader("Results")
            st.metric("Feasibility probability", f"{outputs.feasibility_probability:.3f}")
            st.metric("Mean runtime (h)", f"{outputs.runtime_mean_hours:.2f}")
            st.write(
                f"Runtime P10 / P50 / P90 (h): "
                f"{outputs.runtime_p10_hours:.2f} / {outputs.runtime_p50_hours:.2f} / {outputs.runtime_p90_hours:.2f}"
            )

            if hasattr(outputs, "runtime_hours") and outputs.runtime_hours is not None:
                rt = np.asarray(outputs.runtime_hours)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(rt, bins=40, color="#3478bf", alpha=0.8)
                ax.set_xlabel("Runtime (hours)")
                ax.set_ylabel("Count")
                ax.set_title("Runtime distribution")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Simulation failed: {e}")


if __name__ == "__main__":
    main()
