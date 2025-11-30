"""Streamlit GUI for battery_feasibility Monte Carlo simulations.

This app uses the package API described in the task prompt. It attempts to import the V2-style
APIs first and falls back to V1-style names found in the repository in case of differences.

Run with:

    C:\\path\\to\\python -m pip install streamlit matplotlib
    streamlit run app_streamlit.py

"""
from __future__ import annotations

import io
import json
import typing as t
from pathlib import Path
from dataclasses import dataclass

import numpy as np
try:
    import streamlit as st
except Exception:  # pragma: no cover - allow importing module without streamlit installed
    st = None
import matplotlib.pyplot as plt

# Attempt imports for V2 API; fall back to V1 names present in repo when possible.
try:
    # V2-style imports (as requested)
    from battery_feasibility.models.battery import (
        OcvLut,
        ImpedanceLut,
        CoinCellParams,
        EffectiveBatteryView,
        build_effective_battery,
        load_battery_from_json,
        BatteryArchitecture,
        derive_pack_from_cell,
    )
    from battery_feasibility.models.load import LoadModeConfig, CoinCellLoadProfile
    from battery_feasibility.models.uncertainty import (
        TemperatureConfig,
        AgingConfig,
        VariationConfig,
        TxEventsConfig,
    )
    from battery_feasibility.simulation.monte_carlo import (
        CoinCellSimConfig,
        run_coin_cell_feasibility,
    )
    from battery_feasibility.analytics.outputs import SimulationOutputs
    from battery_feasibility.analytics.plots import (
        plot_uncertainty_histograms,
        plot_runtime_trends,
    )
    from battery_feasibility.analytics.sensitivity import compute_sensitivity_summary
    FALLBACK_IMPORTED = False
except Exception:
    # Fallback to V1-style names present in the repository to maximize compatibility.
    # Note: some names differ; adapt as reasonably as possible.
    from battery_feasibility.models.battery import OcvLut, IrLut as ImpedanceLut, CoinCellParams
    try:
        # V1 doesn't have build_effective_battery; provide a simple shim.
        from battery_feasibility.models.battery import build_effective_battery  # type: ignore
        from battery_feasibility.models.battery import load_battery_from_json  # type: ignore
    except Exception:
        def build_effective_battery(params, T_C, years, cap_unit_delta, R_unit_delta):
            """Minimal shim: return params unchanged wrapped in simple namespace."""
            class EV:
                def __init__(self, p):
                    self.params = p
            return EV(params)

        def load_battery_from_json(fpath: t.Union[str, Path, io.BytesIO]):
            # Try reading JSON and mapping keys to CoinCellParams approx.
            if isinstance(fpath, (str, Path)):
                data = json.loads(Path(fpath).read_text())
            else:
                # file-like
                data = json.load(fpath)
            # Expect keys similar to demo: soc, voc, r arrays
            soc = np.array(data.get("soc", [0.0, 0.5, 1.0]))
            voc = np.array(data.get("voc", [2.0, 3.0, 3.1]))
            r = np.array(data.get("r", [0.02, 0.03, 0.05]))
            ocv = OcvLut(soc=soc, voc=voc)
            imp = ImpedanceLut(soc=soc, dcr_ohm=r, acr_ohm=r)
            return CoinCellParams(
                model_name=data.get("model_name", "loaded"),
                nominal_capacity_mAh=float(data.get("nominal_capacity_mAh", 200.0)),
                nominal_voltage_V=float(data.get("nominal_voltage_V", 3.0)),
                cutoff_voltage_V=float(data.get("cutoff_voltage_V", 2.0)),
                ocv_lut=ocv,
                impedance_lut=imp,
            )

    from battery_feasibility.models.load import LoadPhaseConfig as LoadModeConfig, CoinCellLoadProfile
    @dataclass
    class BatteryArchitecture:  # type: ignore
        series_count: int = 1
        parallel_count: int = 1
    def derive_pack_from_cell(params, arch):  # type: ignore
        return params
    from battery_feasibility.models.uncertainty import (
        TemperatureConfig,
        AgingConfig,
        VariationConfig,
        TxEventsConfig,
    )
    from battery_feasibility.simulation.monte_carlo import (
        CoinCellSimConfig,
        run_coin_cell_feasibility,
    )
    from battery_feasibility.analytics.outputs import SimulationOutputs
    # Fallback plotting / sensitivity: try to import, else provide no-op shims
    try:
        from battery_feasibility.analytics.plots import (
            plot_uncertainty_histograms,
            plot_runtime_trends,
        )
    except Exception:
        def plot_uncertainty_histograms(outputs: SimulationOutputs, bins: int = 40):
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.hist(outputs.runtimes_hours, bins=bins)
            ax.set_title("Runtimes histogram (fallback)", fontsize=8)
            ax.tick_params(labelsize=7)
            return fig

        def plot_runtime_trends(outputs: SimulationOutputs, bins: int = 20):
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(np.sort(outputs.runtimes_hours))
            ax.set_title("Runtime trends (fallback)", fontsize=8)
            ax.tick_params(labelsize=7)
            return fig

    try:
        from battery_feasibility.analytics.sensitivity import compute_sensitivity_summary
    except Exception:
        def compute_sensitivity_summary(outputs: SimulationOutputs):
            # Minimal fallback: return empty list
            return []

    FALLBACK_IMPORTED = True

# Helpers & UI logic

def make_dummy_coin_cell() -> CoinCellParams:
    """Create the demo dummy coin cell used by demo_coin_cell.py."""
    soc = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    voc = np.array([2.0, 2.6, 2.9, 3.0, 3.1])
    r = np.array([20e-3, 25e-3, 30e-3, 40e-3, 60e-3])
    ocv = OcvLut(soc=soc, voc=voc)
    imp = ImpedanceLut(soc=soc, dcr_ohm=r, acr_ohm=r)
    return CoinCellParams(
        model_name="CR2032_dummy",
        nominal_capacity_mAh=220.0,
        nominal_voltage_V=3.0,
        ocv_lut=ocv,
        impedance_lut=imp,
    )


def load_lab_corrected_battery() -> t.Optional[CoinCellParams]:
    """Load CR203x lab-corrected JSON from project directory."""
    return load_lab_battery_by_name("CR2032_lab_corrected.json")


def load_lab_battery_by_name(fname: str) -> t.Optional[CoinCellParams]:
    """Load a specific lab-corrected battery JSON by filename."""
    try:
        for base in [Path(__file__).parent, Path(__file__).parent.parent]:
            candidate = base / fname
            if candidate.exists():
                return load_battery_from_json(str(candidate))
    except Exception:
        return None
    return None


def list_lab_battery_files() -> list[Path]:
    """Return available lab-corrected battery JSONs in the project directory."""
    candidates = [
        "CR2032_lab_corrected.json",
        "CR2450_lab_corrected.json",
        "CR2477_lab_corrected.json",
        "AA_alkaline_lab_validated.json",
    ]
    found: list[Path] = []
    for base in [Path(__file__).parent, Path(__file__).parent.parent]:
        for name in candidates:
            p = base / name
            if p.exists() and p not in found:
                found.append(p)
    return found


def build_load_profile_from_sidebar(tx_residency_pct: float, tx_dc_mA: float, tx_peak_mA: float,
                                    idle_residency_pct: float, idle_dc_mA: float, idle_peak_mA: float,
                                    v_ref: float) -> CoinCellLoadProfile:
    """Assemble a CoinCellLoadProfile with two modes (TX and Idle) and normalize residencies."""
    # Normalize
    tx_r = max(tx_residency_pct, 0.0)
    idle_r = max(idle_residency_pct, 0.0)
    s = tx_r + idle_r
    if s <= 0:
        tx_frac = 0.5
        idle_frac = 0.5
    else:
        tx_frac = tx_r / s
        idle_frac = idle_r / s

    # Map to LoadModeConfig or LoadPhaseConfig depending on available class
    tx_mode = LoadModeConfig(
        name="tx",
        I_ref_dc_mA=float(tx_dc_mA),
        k_dc=2.7,
        I_ref_peak_mA=float(tx_peak_mA),
        k_peak=2.7,
        residency=float(tx_frac),
        V_ref=float(v_ref),
    )
    idle_mode = LoadModeConfig(
        name="idle",
        I_ref_dc_mA=float(idle_dc_mA),
        k_dc=1.0,
        I_ref_peak_mA=float(idle_peak_mA),
        k_peak=1.0,
        residency=float(idle_frac),
        V_ref=float(v_ref),
    )

    profile = CoinCellLoadProfile(modes=[tx_mode, idle_mode])
    return profile


def assemble_sim_config(battery: CoinCellParams,
                        load: CoinCellLoadProfile,
                        temp_cfg: TemperatureConfig,
                        aging_cfg: AgingConfig,
                        var_cfg: VariationConfig,
                        tx_events_cfg: t.Optional[TxEventsConfig],
                        target_runtime_hours: float,
                        target_lifetime_years: float,
                        monte_carlo_samples: int,
                        system_dcr_ohm: float,
                        system_acr_ohm: float,
                        system_cutoff_voltage_V: float,
                        soc_step: float,
                        v_ref_for_Iavg: t.Optional[float]) -> CoinCellSimConfig:
    """Create a CoinCellSimConfig suitable for `run_coin_cell_feasibility`.

    This function maps the streamlit inputs into the simulation dataclass.
    """
    cfg = CoinCellSimConfig(
        battery=battery,
        load=load,
        temperature=temp_cfg,
        aging=aging_cfg,
        variation=var_cfg,
        tx_events=tx_events_cfg,
        target_runtime_hours=float(target_runtime_hours),
        target_lifetime_years=float(target_lifetime_years),
        monte_carlo_samples=int(monte_carlo_samples),
        system_dcr_ohm=float(system_dcr_ohm),
        system_acr_ohm=float(system_acr_ohm),
        system_cutoff_voltage_V=float(system_cutoff_voltage_V),
        soc_step=float(soc_step),
        v_ref_for_Iavg=(None if v_ref_for_Iavg is None else float(v_ref_for_Iavg)),
    )
    return cfg


def render_main_metrics(outputs: SimulationOutputs) -> None:
    """Show top-line numeric results.

    Uses Streamlit metrics and a small table for runtime stats.
    """
    stats = outputs.runtime_stats()
    prob = outputs.feasibility_probability()

    col1, col2, col3 = st.columns(3)
    col1.metric("Feasibility", f"{prob*100:.1f}%")
    col2.metric("Mean runtime (h)", f"{stats['mean']:.2f}")
    col3.metric("P10 / P50 / P90 (h)", f"{stats['p10']:.2f} / {stats['p50']:.2f} / {stats['p90']:.2f}")

    with st.expander("Full output JSON"):
        st.json(outputs.to_dict())


def plot_runtime_histogram(outputs: SimulationOutputs, bins: int = 40):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(outputs.runtimes_hours, bins=bins, color="#3478bf", alpha=0.8)
    ax.set_xlabel("Runtime (hours)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Monte Carlo Runtime Distribution", fontsize=8)
    ax.tick_params(labelsize=7)
    st.pyplot(fig)


def show_sensitivity_table(outputs: SimulationOutputs):
    rows = compute_sensitivity_summary(outputs)
    # rows may be list of dataclasses or dicts — try to normalize
    if rows is None:
        st.info("No sensitivity data available.")
        return

    if isinstance(rows, list) and len(rows) > 0:
        # Convert to list of dicts
        def to_dict(r):
            if hasattr(r, "__dict__"):
                return r.__dict__
            if isinstance(r, dict):
                return r
            # Fallback: create basic mapping
            return {
                "variable": getattr(r, "variable", ""),
                "corr_runtime": getattr(r, "corr_runtime", None),
                "corr_success": getattr(r, "corr_success", None),
                "delta_runtime_p90_p10": getattr(r, "delta_runtime_p90_p10", None),
            }

        table = [to_dict(r) for r in rows]
        st.dataframe(table)
    else:
        st.info("Sensitivity summary empty or not provided by analytics module.")


def load_json_from_uploader(uploader: t.Any) -> t.Optional[CoinCellParams]:
    """Read uploaded JSON file and convert to CoinCellParams using loader shim.

    Returns None on failure.
    """
    if uploader is None:
        return None
    try:
        raw = uploader.read()
        if isinstance(raw, bytes):
            s = raw.decode("utf-8")
            data = json.loads(s)
        else:
            data = json.load(io.StringIO(raw))
        # Use loader if available
        try:
            # Wrap data in BytesIO for our loader shim if it expects file-like
            bio = io.StringIO(json.dumps(data))
            params = load_battery_from_json(bio)
            return params
        except Exception:
            # Last-resort: build manually
            soc = np.array(data.get("soc", [0.0, 0.5, 1.0]))
            voc = np.array(data.get("voc", [2.0, 3.0, 3.1]))
            r = np.array(data.get("r", [0.02, 0.03, 0.05]))
            ocv = OcvLut(soc=soc, voc=voc)
            imp = ImpedanceLut(soc=soc, dcr_ohm=r, acr_ohm=r)
            params = CoinCellParams(
                model_name=data.get("model_name", "uploaded"),
                nominal_capacity_mAh=float(data.get("nominal_capacity_mAh", 200.0)),
                nominal_voltage_V=float(data.get("nominal_voltage_V", 3.0)),
                cutoff_voltage_V=float(data.get("cutoff_voltage_V", 2.0)),
                ocv_lut=ocv,
                impedance_lut=imp,
            )
            return params
    except Exception as e:
        if st:
            st.error(f"Failed to read battery JSON: {e}")
        else:
            print(f"Failed to read battery JSON: {e}")
        return None


def main() -> None:
    st.set_page_config(page_title="Battery Feasibility Simulator", layout="wide")
    st.title("Battery Feasibility Simulator (Streamlit)")

    # Sidebar: battery selection
    st.sidebar.header("Battery Selection")
    battery_choice = st.sidebar.radio("Battery source:", ("Lab-corrected", "Dummy from demo", "Upload JSON file"))

    uploaded_battery = None
    battery_params: t.Optional[CoinCellParams] = None

    if battery_choice == "Lab-corrected":
        lab_files = list_lab_battery_files()
        if not lab_files:
            st.sidebar.warning("No lab-corrected battery JSON files found. Using dummy instead.")
            battery_params = make_dummy_coin_cell()
        else:
            display_names = [p.stem for p in lab_files]
            selected_name = st.sidebar.selectbox("Choose lab battery", display_names, index=0)
            selected_path = lab_files[display_names.index(selected_name)]
            battery_params = load_battery_from_json(str(selected_path))
            if battery_params is None:
                st.sidebar.warning("Failed to load selected lab battery. Using dummy instead.")
                battery_params = make_dummy_coin_cell()
    elif battery_choice == "Upload JSON file":
        uploader = st.sidebar.file_uploader("Upload battery JSON", type=["json"])  # type: ignore
        if uploader is not None:
            battery_params = load_json_from_uploader(uploader)
            if battery_params is None:
                st.sidebar.error("Unable to parse uploaded battery JSON. Falling back to dummy.")
                battery_params = make_dummy_coin_cell()
        else:
            st.sidebar.info("No file uploaded — using dummy battery until a file is provided.")
            battery_params = make_dummy_coin_cell()
    else:
        battery_params = make_dummy_coin_cell()

    # Pack architecture selection (applies to all batteries)
    st.sidebar.subheader("Pack Architecture (yS xP)")
    series_count = st.sidebar.number_input("Series cells (y)", min_value=1, value=1, step=1)
    parallel_count = st.sidebar.number_input("Parallel strings (x)", min_value=1, value=1, step=1)
    architecture = BatteryArchitecture(series_count=int(series_count), parallel_count=int(parallel_count))

    # Derive pack-level battery params from cell-level selection
    cell_params = battery_params
    battery_params = derive_pack_from_cell(cell_params, architecture)

    # Show brief battery info in main pane
    with st.expander("Battery Info", expanded=True):
        st.write("**Cell model:**", getattr(cell_params, "model_name", "unknown"))
        st.write("**Architecture:**", f"{architecture.series_count}S{architecture.parallel_count}P")
        st.write("**Pack nominal capacity (mAh):**", getattr(battery_params, "nominal_capacity_mAh", "?"))
        st.write("**Pack nominal voltage (V):**", getattr(battery_params, "nominal_voltage_V", "?"))

    # Sidebar: load configuration
    st.sidebar.header("Load Configuration")
    st.sidebar.info("TX events are sampled per Monte Carlo run; residency is computed from event frequency and duration.")
    
    st.sidebar.subheader("TX Events Configuration")
    tx_dist_type = st.sidebar.selectbox("Distribution type", ["fixed", "normal", "poisson"], index=0)
    
    event_duration_ms = st.sidebar.number_input(
        "Event duration (milliseconds)", 
        min_value=1, 
        value=100, 
        help="Duration of each TX event"
    )
    event_duration_s = event_duration_ms / 1000.0
    
    mean_events = st.sidebar.slider(
        "Mean events per day",
        min_value=0,
        max_value=10000,
        value=10,
        help="Average number of TX events per day"
    )
    
    sigma_events = None
    if tx_dist_type == "normal":
        sigma_events = st.sidebar.slider(
            "Sigma (events per day)",
            min_value=0,
            max_value=1000,
            value=1,
            help="Standard deviation for normal distribution"
        )
    
    min_events = st.sidebar.slider(
        "Min events per day",
        min_value=0,
        max_value=1000,
        value=0,
        help="Lower bound on events per day"
    )
    
    max_events = st.sidebar.slider(
        "Max events per day",
        min_value=0,
        max_value=10000,
        value=1000,
        help="Upper bound on events per day"
    )
    
    max_duty = st.sidebar.slider(
        "Max TX residency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Maximum allowed TX residency (fraction of day)"
    )
    
    # Build TxEventsConfig
    tx_events_cfg = TxEventsConfig(
        dist_type=tx_dist_type,
        event_duration_s=float(event_duration_s),
        mean_events_per_day=float(mean_events),
        sigma_events_per_day=float(sigma_events) if sigma_events is not None else 0.0,
        min_events_per_day=float(min_events),
        max_events_per_day=float(max_events),
        max_duty=float(max_duty),
    )
    
    st.sidebar.subheader("TX mode")
    tx_dc_mA = st.sidebar.number_input("TX DC current at V_ref (mA)", value=0.0)
    tx_peak_mA = st.sidebar.number_input("TX peak current (mA)", value=30.0)

    st.sidebar.subheader("Idle mode")
    idle_dc_mA = st.sidebar.number_input("Idle DC current at V_ref (mA)", value=3.0)
    idle_peak_mA = st.sidebar.number_input("Idle peak current (mA)", value=0.0)

    v_ref = st.sidebar.number_input("V_ref for currents (V)", value=float(battery_params.nominal_voltage_V))

    # Create default load profile (residencies will be overridden by tx_events sampling in MC)
    load_profile = build_load_profile_from_sidebar(0.5, tx_dc_mA, tx_peak_mA,
                                                   0.5, idle_dc_mA, idle_peak_mA,
                                                   v_ref)

    # Sidebar: uncertainties
    st.sidebar.header("Uncertainties")
    st.sidebar.subheader("Temperature")
    temp_mode = st.sidebar.radio("Temperature mode", ("Single", "Normal"))
    if temp_mode == "Single":
        temp_value = st.sidebar.slider("Temperature (°C)", min_value=-40.0, max_value=85.0, value=25.0)
        temp_cfg = TemperatureConfig(mode="single", value_C=float(temp_value))
    else:
        mean_C = st.sidebar.number_input("Mean (°C)", value=25.0)
        sigma_C = st.sidebar.number_input("Sigma (°C)", value=5.0)
        min_C = st.sidebar.number_input("Min (°C)", value=-10.0)
        max_C = st.sidebar.number_input("Max (°C)", value=50.0)
        temp_cfg = TemperatureConfig(mode="distribution", dist_type="normal", mean_C=float(mean_C), sigma_C=float(sigma_C), min_C=float(min_C), max_C=float(max_C))

    st.sidebar.subheader("Aging")
    years_nominal = st.sidebar.slider("Nominal years in field", min_value=0.0, max_value=10.0, value=1.0)
    sigma_years = st.sidebar.number_input("Aging sigma (years)", value=0.0)
    aging_cfg = AgingConfig(years_nominal=float(years_nominal), sigma_years=float(sigma_years))

    # Variation comes from battery JSON, not user input
    var_cfg = VariationConfig(
        sigma_cap_pct=float(battery_params.sigma_capacity_pct),
        sigma_R_pct=float(battery_params.sigma_R_pct)
    )

    # Sidebar: simulation settings
    st.sidebar.header("Simulation Settings")
    target_runtime_hours = st.sidebar.number_input("Target runtime (hours)", value=24.0)
    target_lifetime_years = st.sidebar.number_input("Target lifetime (years)", value=1.0)
    monte_carlo_samples = st.sidebar.number_input("Monte Carlo samples", min_value=1, max_value=2000000, value=1000)
    system_dcr_ohm = st.sidebar.number_input("System DCR (ohm)", value=0.0)
    system_cutoff_voltage_V = st.sidebar.number_input("System cutoff voltage (V)", value=2.0)
    soc_step = st.sidebar.number_input("SOC step", min_value=0.0001, max_value=0.1, value=0.01)
    v_ref_for_Iavg = st.sidebar.number_input("V_ref for Iavg (leave 0 for auto)", value=0.0)
    if v_ref_for_Iavg <= 0:
        v_ref_for_Iavg_val = None
    else:
        v_ref_for_Iavg_val = float(v_ref_for_Iavg)

    seed_val = st.sidebar.number_input("Random seed (0 = random)", value=42)
    seed = None if int(seed_val) == 0 else int(seed_val)

    run_btn = st.sidebar.button("Run simulation")

    # Run simulation when requested
    outputs: t.Optional[SimulationOutputs] = None
    if run_btn:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                sim_cfg = assemble_sim_config(
                    battery=battery_params,
                    load=load_profile,
                    temp_cfg=temp_cfg,
                    aging_cfg=aging_cfg,
                    var_cfg=var_cfg,
                    tx_events_cfg=tx_events_cfg,
                    target_runtime_hours=target_runtime_hours,
                    target_lifetime_years=target_lifetime_years,
                    monte_carlo_samples=monte_carlo_samples,
                    system_dcr_ohm=system_dcr_ohm,
                    system_acr_ohm=0.0,
                    system_cutoff_voltage_V=system_cutoff_voltage_V,
                    soc_step=soc_step,
                    v_ref_for_Iavg=v_ref_for_Iavg_val,
                )
                outputs = run_coin_cell_feasibility(sim_cfg, seed=seed)
                st.success("Simulation finished")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # Display results area
    st.header("Results")
    if outputs is None:
        st.info("No results yet. Configure options and click 'Run simulation'.")
    else:
        render_main_metrics(outputs)
        st.subheader("Runtime Histogram")
        plot_runtime_histogram(outputs)

        st.subheader("Uncertainty Histograms")
        try:
            fig1 = plot_uncertainty_histograms(outputs)
            if fig1 is not None:
                try:
                    fig1.set_size_inches(6, 4)
                    fig1.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig1)
        except Exception as e:
            st.warning(f"Could not plot uncertainty histograms: {e}")

        st.subheader("Runtime Trends")
        try:
            fig2 = plot_runtime_trends(outputs)
            if fig2 is not None:
                try:
                    fig2.set_size_inches(6, 6)
                    fig2.tight_layout()
                except Exception:
                    pass
                st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not plot runtime trends: {e}")

        st.subheader("Sensitivity Summary")
        try:
            show_sensitivity_table(outputs)
        except Exception as e:
            st.warning(f"Could not compute sensitivity summary: {e}")


if __name__ == "__main__":
    main()
