"""Visualization functions for LLMCompass profiling results.

Generates publication-quality PDF figures using matplotlib + seaborn.
"""

import os
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

_LLMCOMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, _LLMCOMPASS_ROOT)

from profiling.bottleneck_report import (
    BottleneckReport,
    WorkloadBottleneckReport,
)
from profiling.profiler import SYSTOLIC_OPS, VECTOR_OPS, COMM_OPS

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_OP_COLORS = {
    "qkv_proj": "#1f77b4",
    "q_mul_k": "#aec7e8",
    "a_mul_v": "#ff7f0e",
    "h_matmul0": "#ffbb78",
    "h_matmul1": "#2ca02c",
    "h_matmul2": "#98df8a",
    "softmax": "#d62728",
    "layernorm_0": "#ff9896",
    "layernorm_1": "#e377c2",
    "gelu": "#9467bd",
    "allreduce_mha": "#8c564b",
    "allreduce_ffn": "#c49c94",
}

_BOTTLENECK_COLORS = {
    "compute_bound": "#d62728",
    "memory_bound": "#1f77b4",
    "balanced": "#2ca02c",
}


# ---------------------------------------------------------------------------
# 1. Roofline plot
# ---------------------------------------------------------------------------

def plot_roofline(
    report: WorkloadBottleneckReport,
    output_path: str = "roofline.pdf",
    title: Optional[str] = None,
):
    """Plot a roofline diagram with per-operator points.

    The roofline shows two ceilings (systolic and vector) against the
    memory bandwidth wall.  Each operator is plotted at its arithmetic
    intensity vs achieved throughput.
    """

    if report.layer_report is None:
        raise ValueError("Report has no layer_report to plot")

    peak_sys = report.peak_systolic_flops
    peak_vec = report.peak_vector_flops
    hbm_bw = report.peak_hbm_bandwidth

    fig, ax = plt.subplots(figsize=(10, 6))

    # Arithmetic intensity range
    ai_min, ai_max = 0.1, 1e5
    ai_range = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    # Roofline for systolic array / matmul
    roofline_sys = np.minimum(peak_sys, ai_range * hbm_bw)
    ax.plot(ai_range, roofline_sys, "r-", linewidth=2, label=f"Systolic peak ({peak_sys/1e12:.1f} TFLOPS)")

    # Roofline for vector unit
    roofline_vec = np.minimum(peak_vec, ai_range * hbm_bw)
    ax.plot(ai_range, roofline_vec, "b--", linewidth=1.5, label=f"Vector peak ({peak_vec/1e12:.1f} TFLOPS)")

    # Memory bandwidth line
    bw_line = ai_range * hbm_bw
    ax.plot(ai_range, bw_line, "k:", linewidth=1, alpha=0.3)

    # Plot operators
    for op_report in report.layer_report.operator_reports:
        if op_report.latency_seconds <= 0:
            continue
        ai = op_report.arithmetic_intensity
        if ai <= 0 or ai == float("inf"):
            continue

        # Achieved throughput = flop_count / latency
        achieved = op_report.flop_count / op_report.latency_seconds if op_report.latency_seconds > 0 else 0
        if achieved <= 0:
            continue

        color = _OP_COLORS.get(op_report.operator_name, "#7f7f7f")
        marker = "o" if op_report.operator_name in SYSTOLIC_OPS else "s"
        ax.scatter(
            ai, achieved,
            color=color, marker=marker, s=100, zorder=5,
            edgecolors="black", linewidths=0.5,
            label=op_report.operator_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)")
    ax.set_ylabel("Throughput (FLOPS)")
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(1e9, max(peak_sys, peak_vec) * 2)

    if title is None:
        title = (
            f"Roofline — {report.model_name} | {report.phase} | "
            f"BS={report.batch_size} SL={report.seq_len} | "
            f"{report.device_count}x GPU"
        )
    ax.set_title(title, fontsize=12)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    ax.legend(
        [h for h, _ in unique],
        [l for _, l in unique],
        loc="upper left",
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Roofline plot saved to {output_path}")


# ---------------------------------------------------------------------------
# 2. Latency breakdown stacked bar chart
# ---------------------------------------------------------------------------

def plot_latency_breakdown(
    reports: List[WorkloadBottleneckReport],
    output_path: str = "latency_breakdown.pdf",
    title: Optional[str] = None,
):
    """Stacked bar chart showing per-operator latency for each workload."""

    if not reports:
        return

    # Gather data
    labels = []
    op_data = {}  # op_name -> list of latencies per report
    for r in reports:
        label = f"{r.task_id}\n{r.phase}"
        labels.append(label)
        if r.layer_report:
            for op in r.layer_report.operator_reports:
                op_data.setdefault(op.operator_name, []).append(
                    op.latency_seconds * 1e3  # convert to ms
                )
        else:
            for name in _OP_COLORS:
                op_data.setdefault(name, []).append(0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 1.2), 6))
    x = np.arange(len(labels))
    bar_width = 0.6
    bottom = np.zeros(len(labels))

    for op_name in _OP_COLORS:
        if op_name not in op_data:
            continue
        values = np.array(op_data[op_name])
        color = _OP_COLORS[op_name]
        ax.bar(x, values, bar_width, bottom=bottom, label=op_name, color=color)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title or "Per-Operator Latency Breakdown")
    ax.legend(
        loc="upper left",
        fontsize=7,
        ncol=3,
        bbox_to_anchor=(1.01, 1),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency breakdown plot saved to {output_path}")


# ---------------------------------------------------------------------------
# 3. Bottleneck heatmap
# ---------------------------------------------------------------------------

def plot_bottleneck_heatmap(
    reports: List[WorkloadBottleneckReport],
    output_path: str = "bottleneck_heatmap.pdf",
    title: Optional[str] = None,
):
    """Heatmap showing bottleneck type (compute/memory/balanced) for each
    operator across workloads."""

    if not reports:
        return

    # Map bottleneck labels to numeric values for coloring
    bn_to_num = {"compute_bound": 2, "balanced": 1, "memory_bound": 0}

    # Collect operator names from first report
    if not reports[0].layer_report:
        return
    op_names = [op.operator_name for op in reports[0].layer_report.operator_reports]

    labels = [f"{r.task_id}" for r in reports]
    data = np.zeros((len(op_names), len(reports)))

    for j, r in enumerate(reports):
        if r.layer_report:
            for i, op in enumerate(r.layer_report.operator_reports):
                data[i, j] = bn_to_num.get(op.bottleneck, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 0.8), max(4, len(op_names) * 0.5)))

    # Custom colormap: blue (memory) -> green (balanced) -> red (compute)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(op_names)))
    ax.set_yticklabels(op_names, fontsize=9)
    ax.set_title(title or "Bottleneck Classification Heatmap")

    # Color bar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Memory-bound", "Balanced", "Compute-bound"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Bottleneck heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# 4. Utilization bar charts
# ---------------------------------------------------------------------------

def plot_utilization(
    reports: List[WorkloadBottleneckReport],
    output_path: str = "utilization.pdf",
    title: Optional[str] = None,
):
    """Grouped bar chart showing compute and memory utilization per workload."""

    if not reports:
        return

    labels = [f"{r.task_id}\n{r.phase}" for r in reports]
    compute_utils = []
    memory_utils = []

    for r in reports:
        if r.layer_report:
            # Weighted average utilization across operators
            total_lat = sum(op.latency_seconds for op in r.layer_report.operator_reports)
            if total_lat > 0:
                cu = sum(
                    op.compute_utilization * op.latency_seconds
                    for op in r.layer_report.operator_reports
                ) / total_lat
                mu = sum(
                    op.bandwidth_utilization * op.latency_seconds
                    for op in r.layer_report.operator_reports
                ) / total_lat
            else:
                cu, mu = 0, 0
            compute_utils.append(cu)
            memory_utils.append(mu)
        else:
            compute_utils.append(0)
            memory_utils.append(0)

    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 1.2), 5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, compute_utils, width, label="Compute Utilization", color="#d62728", alpha=0.8)
    ax.bar(x + width / 2, memory_utils, width, label="Bandwidth Utilization", color="#1f77b4", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Utilization (0–1)")
    ax.set_ylim(0, 1.1)
    ax.set_title(title or "Compute & Memory Bandwidth Utilization")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Utilization plot saved to {output_path}")
