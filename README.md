# Battery Feasibility Simulator

Battery runtime and brownout feasibility Monte Carlo simulator for IoT devices. Given a battery model, load profile, and uncertainty ranges, it estimates expected lifetime and probability of brownouts across temperature, aging, capacity, and internal resistance variation.

## Features
- Battery JSON library for common cells (CR series, AA, AAA)
- Configurable yS x P architecture (series/parallel) applied to any cell type
- Monte Carlo simulation including temperature, aging, capacity, and R variation
- Brownout feasibility evaluation with probability outputs
- Streamlit web UI for quick scenario setup and visualization

## Project Layout
- `battery_feasibility_full/battery_feasibility/` — core engine (models, simulation, analytics)
- `battery_feasibility_full/api.py` — API wrapper exposing `run_battery_feasibility(user_config)` and related config/result models
- `app_streamlit.py` — Streamlit web UI to configure batteries, ySxP, load, uncertainty, and Monte Carlo samples
- `quick_test_api.py` — simple command-line wrapper to exercise the API
- Battery model JSONs (e.g., `CR2032_lab_corrected.json`, `CR2450_lab_corrected.json`, `AA_alkaline_lab_validated.json`, etc.)

## Requirements
- Python 3.11
- Windows PowerShell for the commands below

## Setup (create and activate virtual environment)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Streamlit app
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app_streamlit.py
```

## Run the API test script
From the project root:
```powershell
.\.venv\Scripts\Activate.ps1
python quick_test_api.py
```

## Notes
- This is an early engineering estimation tool, not a certified safety analysis tool.
- Battery JSONs define cell characteristics; ySxP scaling is applied uniformly to all supported cell types.
- Monte Carlo outputs include runtime distributions and brownout feasibility probabilities.
