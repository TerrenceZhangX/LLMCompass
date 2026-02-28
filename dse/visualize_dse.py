#!/usr/bin/env python3
"""Visualization for DSE results.

Generates publication-quality PDF figures:
  1. Pareto frontier (2D projections with area color)
  2. Sensitivity bar charts
  3. Parameter sweep curves
  4. Design space heatmaps

Usage:
    python -m dse.visualize_dse [--input-dir results/dse] [--output-dir results/dse/figures]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Skip seaborn to avoid slow scipy import chain
_HAS_SEABORN = False
plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_PARAM_COLORS = {
    "core_count": "#1f77b4",
    "hbm_channel_count": "#ff7f0e",
    "global_buffer_MB": "#2ca02c",
    "systolic_array_dim": "#d62728",
    "sram_KB": "#9467bd",
}

_PARAM_LABELS = {
    "core_count": "SM Count",
    "hbm_channel_count": "HBM Channels",
    "global_buffer_MB": "L2 Cache (MB)",
    "systolic_array_dim": "Systolic Array Dim",
    "sram_KB": "SRAM per Core (KB)",
}

_OBJ_LABELS = {
    "prefill_latency_ms": "Prefill Latency (ms)",
    "decode_latency_ms": "Decode Latency (ms)",
    "die_area_mm2": "Die Area (mm²)",
}


def load_dse_results(input_dir: str):
    """Load all DSE result files."""
    with open(os.path.join(input_dir, "all_designs.json")) as f:
        all_designs = json.load(f)
    with open(os.path.join(input_dir, "pareto_designs.json")) as f:
        pareto_designs = json.load(f)
    with open(os.path.join(input_dir, "sensitivity.json")) as f:
        sensitivity = json.load(f)
    return all_designs, pareto_designs, sensitivity


# ---------------------------------------------------------------------------
# 1. Pareto Frontier Scatter Plot
# ---------------------------------------------------------------------------

def plot_pareto_frontier(all_designs, pareto_designs, output_dir):
    """2D Pareto frontier projections: prefill vs decode, colored by area."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    projections = [
        ("prefill_latency_ms", "decode_latency_ms", "die_area_mm2"),
        ("die_area_mm2", "prefill_latency_ms", "decode_latency_ms"),
        ("die_area_mm2", "decode_latency_ms", "prefill_latency_ms"),
    ]

    for ax, (x_key, y_key, c_key) in zip(axes, projections):
        # All designs (faded)
        xs = [d[x_key] for d in all_designs]
        ys = [d[y_key] for d in all_designs]
        cs = [d[c_key] for d in all_designs]
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", alpha=0.15, s=8,
                        edgecolors="none")

        # Pareto designs (prominent)
        px = [d[x_key] for d in pareto_designs]
        py = [d[y_key] for d in pareto_designs]
        pc = [d[c_key] for d in pareto_designs]
        ax.scatter(px, py, c=pc, cmap="viridis", alpha=0.9, s=50,
                   edgecolors="black", linewidths=0.5, zorder=5)

        # Pareto line
        pareto_sorted = sorted(zip(px, py), key=lambda t: t[0])
        ax.plot([t[0] for t in pareto_sorted], [t[1] for t in pareto_sorted],
                "r--", alpha=0.5, linewidth=1)

        ax.set_xlabel(_OBJ_LABELS.get(x_key, x_key))
        ax.set_ylabel(_OBJ_LABELS.get(y_key, y_key))
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(_OBJ_LABELS.get(c_key, c_key))

    fig.suptitle("DSE Pareto Frontier — 2D Projections", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_frontier.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 2. Sensitivity Bar Chart
# ---------------------------------------------------------------------------

def plot_sensitivity_bars(sensitivity, output_dir):
    """Grouped bar chart of parameter sensitivity per objective."""
    objectives = ["prefill_latency_ms", "decode_latency_ms", "die_area_mm2"]
    params = list(_PARAM_COLORS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, obj in zip(axes, objectives):
        values = []
        colors = []
        labels = []
        for p in params:
            val = sensitivity[p][obj]["range_percent"]
            values.append(val)
            colors.append(_PARAM_COLORS[p])
            labels.append(_PARAM_LABELS[p])

        # Sort by impact
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        bars = ax.barh(
            [labels[i] for i in order],
            [values[i] for i in order],
            color=[colors[i] for i in order],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for bar, idx in zip(bars, order):
            width = bar.get_width()
            ax.annotate(
                f"{values[idx]:.1f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left", va="center", fontsize=9,
            )

        ax.set_xlabel("Range of Variation (%)")
        ax.set_title(_OBJ_LABELS[obj])
        ax.set_xlim(0, max(values) * 1.3 if max(values) > 0 else 10)

    fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "sensitivity_bars.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 3. Parameter Sweep Curves
# ---------------------------------------------------------------------------

def plot_parameter_sweeps(sensitivity, output_dir):
    """Line plots: mean objective value vs each parameter value."""
    params = list(_PARAM_COLORS.keys())
    objectives = ["prefill_latency_ms", "decode_latency_ms", "die_area_mm2"]

    fig, axes = plt.subplots(len(params), len(objectives),
                             figsize=(15, 3.5 * len(params)))
    if len(params) == 1:
        axes = [axes]

    for row, param in enumerate(params):
        for col, obj in enumerate(objectives):
            ax = axes[row][col]
            means = sensitivity[param][obj]["means"]
            x_vals = sorted([float(k) for k in means.keys()])
            y_vals = [means[str(int(x)) if x == int(x) else str(x)] for x in x_vals]

            ax.plot(x_vals, y_vals, "o-", color=_PARAM_COLORS[param],
                    markersize=6, linewidth=2)
            ax.set_xlabel(_PARAM_LABELS[param])
            ax.set_ylabel(_OBJ_LABELS[obj])

            # Highlight diminishing returns
            if len(y_vals) > 2:
                deltas = [y_vals[i+1] - y_vals[i] for i in range(len(y_vals)-1)]
                if all(d <= 0 for d in deltas) or all(d >= 0 for d in deltas):
                    # Monotonic — check for diminishing returns
                    ratios = []
                    for i in range(1, len(deltas)):
                        if abs(deltas[i-1]) > 0:
                            ratios.append(abs(deltas[i]) / abs(deltas[i-1]))
                    if ratios and max(ratios) < 0.5:
                        ax.set_facecolor("#fff3e0")  # light orange = diminishing returns

            if row == 0:
                ax.set_title(_OBJ_LABELS[obj])

    fig.suptitle("Parameter Sweep — Mean Objective vs Parameter Value",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "parameter_sweeps.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 4. Design Space Heatmap (core_count × systolic_array_dim → latency)
# ---------------------------------------------------------------------------

def plot_design_heatmaps(all_designs, output_dir):
    """Heatmaps of latency across key parameter pairs."""
    # Group by (core_count, systolic_array_dim) → mean prefill latency
    param_pairs = [
        ("core_count", "systolic_array_dim", "prefill_latency_ms"),
        ("core_count", "hbm_channel_count", "decode_latency_ms"),
    ]

    fig, axes = plt.subplots(1, len(param_pairs), figsize=(14, 5.5))
    if len(param_pairs) == 1:
        axes = [axes]

    for ax, (p1, p2, obj) in zip(axes, param_pairs):
        groups = defaultdict(list)
        for d in all_designs:
            key = (d[p1], d[p2])
            val = d[obj]
            if val < float("inf"):
                groups[key].append(val)

        # Build 2D array
        p1_vals = sorted(set(k[0] for k in groups.keys()))
        p2_vals = sorted(set(k[1] for k in groups.keys()))
        matrix = np.full((len(p2_vals), len(p1_vals)), np.nan)

        for i, v2 in enumerate(p2_vals):
            for j, v1 in enumerate(p1_vals):
                if (v1, v2) in groups:
                    matrix[i, j] = np.mean(groups[(v1, v2)])

        if _HAS_SEABORN:
            hm = sns.heatmap(
                matrix, ax=ax, cmap="YlOrRd_r",
                xticklabels=[str(v) for v in p1_vals],
                yticklabels=[str(v) for v in p2_vals],
                annot=True, fmt=".0f", linewidths=0.5,
                cbar_kws={"label": _OBJ_LABELS[obj]},
            )
        else:
            im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
            ax.set_xticks(range(len(p1_vals)))
            ax.set_xticklabels([str(v) for v in p1_vals])
            ax.set_yticks(range(len(p2_vals)))
            ax.set_yticklabels([str(v) for v in p2_vals])
            plt.colorbar(im, ax=ax, label=_OBJ_LABELS[obj])

        ax.set_xlabel(_PARAM_LABELS[p1])
        ax.set_ylabel(_PARAM_LABELS[p2])
        ax.set_title(f"Mean {_OBJ_LABELS[obj]}")

    fig.suptitle("Design Space Heatmaps", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "design_heatmaps.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 5. Area Breakdown Pie (compute vs IO)
# ---------------------------------------------------------------------------

def plot_area_breakdown(pareto_designs, output_dir):
    """Stacked bar chart of compute vs IO area for Pareto designs."""
    sorted_pareto = sorted(pareto_designs, key=lambda d: d["die_area_mm2"])

    # Select a representative subset if too many
    if len(sorted_pareto) > 20:
        indices = np.linspace(0, len(sorted_pareto)-1, 20, dtype=int)
        subset = [sorted_pareto[i] for i in indices]
    else:
        subset = sorted_pareto

    labels = [
        f"C{d['core_count']}\nSA{d['systolic_array_dim']}\nHBM{d['hbm_channel_count']}\nDC{d['device_count']}"
        for d in subset
    ]
    compute = [d["compute_area_mm2"] for d in subset]
    io = [d["io_area_mm2"] for d in subset]

    fig, ax = plt.subplots(figsize=(max(12, len(subset) * 0.6), 5))
    x = np.arange(len(subset))
    width = 0.7

    ax.bar(x, compute, width, label="Compute Chiplet", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, io, width, bottom=compute, label="IO Die", color="#ff7f0e",
           edgecolor="black", linewidth=0.5)

    # Reticle limit line
    ax.axhline(y=900, color="red", linestyle="--", linewidth=1.5,
               label="Reticle Limit (900 mm²)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Die Area (mm²)")
    ax.set_title("Area Breakdown — Pareto-Optimal Designs")
    ax.legend(loc="upper left")
    plt.tight_layout()
    path = os.path.join(output_dir, "area_breakdown.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 6. Prefill vs Decode Tradeoff (colored by device count)
# ---------------------------------------------------------------------------

def plot_prefill_decode_tradeoff(all_designs, pareto_designs, output_dir):
    """Scatter plot showing prefill-decode tradeoff colored by device count."""
    fig, ax = plt.subplots(figsize=(10, 7))

    dc_colors = {1: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}

    for dc, color in dc_colors.items():
        subset = [d for d in all_designs if d["device_count"] == dc]
        xs = [d["prefill_latency_ms"] for d in subset]
        ys = [d["decode_latency_ms"] for d in subset]
        ax.scatter(xs, ys, alpha=0.15, s=10, color=color, edgecolors="none",
                   label=f"DC={dc} (all)")

    for dc, color in dc_colors.items():
        subset = [d for d in pareto_designs if d["device_count"] == dc]
        if subset:
            xs = [d["prefill_latency_ms"] for d in subset]
            ys = [d["decode_latency_ms"] for d in subset]
            ax.scatter(xs, ys, alpha=0.9, s=80, color=color,
                       edgecolors="black", linewidths=0.8, zorder=5,
                       marker="D", label=f"DC={dc} (Pareto)")

    ax.set_xlabel("Prefill Latency (ms)")
    ax.set_ylabel("Decode Latency (ms)")
    ax.set_title("Prefill vs Decode Latency Tradeoff by Device Count")
    ax.legend(loc="upper right")
    ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(output_dir, "prefill_decode_tradeoff.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize DSE results")
    parser.add_argument("--input-dir", default="results/dse",
                        help="Directory with DSE result JSONs")
    parser.add_argument("--output-dir", default="results/dse/figures",
                        help="Directory for PDF figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading DSE results from {args.input_dir}...")
    all_designs, pareto_designs, sensitivity = load_dse_results(args.input_dir)
    print(f"  {len(all_designs)} total designs, {len(pareto_designs)} Pareto")

    print("Generating figures...")
    plot_pareto_frontier(all_designs, pareto_designs, args.output_dir)
    plot_sensitivity_bars(sensitivity, args.output_dir)
    plot_parameter_sweeps(sensitivity, args.output_dir)
    plot_design_heatmaps(all_designs, args.output_dir)
    plot_area_breakdown(pareto_designs, args.output_dir)
    plot_prefill_decode_tradeoff(all_designs, pareto_designs, args.output_dir)
    print("Done! All DSE figures saved.")


if __name__ == "__main__":
    main()
