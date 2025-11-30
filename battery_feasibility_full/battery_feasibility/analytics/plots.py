from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .outputs import SimulationOutputs


def _safe_hist(ax, data, title: str, xlabel: str, bins: int = 40) -> None:
    if data is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        return
    arr = np.asarray(data)
    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        return
    ax.hist(arr, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def plot_uncertainty_histograms(outputs: SimulationOutputs, bins: int = 40):
    """Plot histograms of the sampled uncertainties: T, years, cap delta, R delta, and optionally TX residency."""
    # Count how many variables we have
    variables = [
        ("Temperature Samples", outputs.temps_C, "Temperature [°C]"),
        ("Aging Samples", outputs.years_in_field, "Years in Field [yr]"),
        ("Capacity Variation Samples", outputs.cap_unit_deltas, "ΔCapacity (fraction)"),
        ("Resistance Variation Samples", outputs.R_unit_deltas, "ΔResistance (fraction)"),
    ]
    if outputs.tx_events_per_day is not None:
        variables.append(("TX Events/Day Samples", outputs.tx_events_per_day, "TX events per day"))
    elif outputs.tx_residency is not None:
        variables.append(("TX Residency Samples", outputs.tx_residency, "TX Duty Fraction"))
    
    n_vars = len(variables)
    n_cols = 2
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    plt.rcParams.update({'font.size': 7})

    for ax_idx, (title, data, xlabel) in enumerate(variables):
        _safe_hist(axes[ax_idx], data, title, xlabel, bins=bins)
    
    # Hide unused subplots
    for ax_idx in range(n_vars, len(axes)):
        axes[ax_idx].axis('off')

    fig.tight_layout()
    return fig


def _binned_trend(x, y, bins: int = 20):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_mean = np.full(bins, np.nan, dtype=float)
    y_count = np.zeros(bins, dtype=int)

    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(mask):
            y_mean[i] = float(np.mean(y[mask]))
            y_count[i] = int(np.sum(mask))

    return centers, y_mean, y_count


def plot_runtime_trends(outputs: SimulationOutputs, bins: int = 20):
    """Plot runtime vs each uncertainty variable (scatter + binned mean trend)."""
    rt = np.asarray(outputs.runtimes_hours)
    plt.rcParams.update({'font.size': 7})

    vars_data = [
        ("Temperature [°C]", outputs.temps_C),
        ("Years in Field [yr]", outputs.years_in_field),
        ("ΔCapacity (fraction)", outputs.cap_unit_deltas),
        ("ΔResistance (fraction)", outputs.R_unit_deltas),
    ]
    if outputs.tx_events_per_day is not None:
        vars_data.append(("TX events per day", outputs.tx_events_per_day))
    elif outputs.tx_residency is not None:
        vars_data.append(("TX Duty Fraction", outputs.tx_residency))
    
    vars_data = [(label, data) for (label, data) in vars_data if data is not None]

    if not vars_data:
        raise ValueError("No uncertainty diagnostics stored in SimulationOutputs.")

    n = len(vars_data)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    plt.rcParams.update({'font.size': 7})

    for ax, (label, data) in zip(axes, vars_data):
        x = np.asarray(data)
        ax.scatter(x, rt, s=5, alpha=0.3)
        centers, y_mean, _ = _binned_trend(x, rt, bins=bins)
        ax.plot(centers, y_mean, linewidth=2)
        ax.set_xlabel(label, fontsize=7)
        ax.set_ylabel("Runtime [h]", fontsize=7)
        ax.set_title(f"Runtime vs {label}", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots if any
    for ax in axes[len(vars_data) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig
