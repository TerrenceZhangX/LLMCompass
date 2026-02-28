#!/usr/bin/env python3
"""Generate publication-quality figures for the GPU DSE paper."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Use a clean style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(path):
    with open(os.path.join(RESULTS_DIR, path)) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Pareto Frontier — Prefill vs Decode Latency
# =============================================================================
def fig_pareto():
    pareto = load_json('dse/pareto_designs.json')
    all_designs = load_json('dse/all_designs.json')

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Plot all feasible designs as light gray background
    all_prefill = [d['prefill_latency_ms'] for d in all_designs]
    all_decode = [d['decode_latency_ms'] for d in all_designs]
    all_area = [d['die_area_mm2'] for d in all_designs]
    ax.scatter(all_prefill, all_decode, c='#e0e0e0', s=3, alpha=0.3,
               zorder=1, rasterized=True)

    # Pareto points colored by device count
    colors = {4: '#2196F3', 8: '#FF5722', 1: '#4CAF50'}
    labels = {4: '4 devices', 8: '8 devices', 1: '1 device'}

    for dc in [4, 8, 1]:
        pts = [d for d in pareto if d['device_count'] == dc]
        if not pts:
            continue
        x = [d['prefill_latency_ms'] for d in pts]
        y = [d['decode_latency_ms'] for d in pts]
        areas = [d['die_area_mm2'] for d in pts]
        sizes = [max(20, min(120, (a / 900) * 100)) for a in areas]
        ax.scatter(x, y, c=colors[dc], s=sizes, alpha=0.8, edgecolors='black',
                   linewidths=0.5, label=labels[dc], zorder=3)

    # Sort Pareto points by prefill for frontier line
    pareto_sorted = sorted(pareto, key=lambda d: d['prefill_latency_ms'])
    px = [d['prefill_latency_ms'] for d in pareto_sorted]
    py = [d['decode_latency_ms'] for d in pareto_sorted]
    ax.plot(px, py, 'k--', alpha=0.4, linewidth=0.8, zorder=2)

    ax.set_xlabel('Prefill Latency (ms)')
    ax.set_ylabel('Decode Latency (ms)')
    ax.set_title('Pareto Frontier: Prefill vs Decode Latency')
    ax.legend(title='Device Count', loc='upper right')

    # Add annotation for trade-off
    ax.annotate('Decode-optimized\n(4-device)',
                xy=(max(px) * 0.85, min(py) * 1.05),
                fontsize=8, ha='center', style='italic', color='#2196F3')
    ax.annotate('Prefill-optimized\n(8-device)',
                xy=(min(px) * 1.15, max(py) * 0.95),
                fontsize=8, ha='center', style='italic', color='#FF5722')

    fig.savefig(os.path.join(FIGURES_DIR, 'pareto_frontier.pdf'))
    plt.close(fig)
    print('  ✓ pareto_frontier.pdf')


# =============================================================================
# Figure 2: Sensitivity Heatmap
# =============================================================================
def fig_sensitivity_heatmap():
    sens = load_json('dse/sensitivity.json')

    params = ['systolic_array_dim', 'core_count', 'hbm_channel_count',
              'global_buffer_MB', 'sram_KB']
    param_labels = ['SA Dimension', 'Core Count', 'HBM Channels',
                    'L2 Cache (MB)', 'SRAM (KB)']
    objectives = ['prefill_latency_ms', 'decode_latency_ms', 'die_area_mm2']
    obj_labels = ['Prefill Latency', 'Decode Latency', 'Die Area']

    matrix = np.zeros((len(params), len(objectives)))
    for i, p in enumerate(params):
        for j, obj in enumerate(objectives):
            matrix[i, j] = sens[p][obj]['range_percent']

    # Log scale for better visualization given the huge range
    log_matrix = np.log10(matrix + 1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(obj_labels)))
    ax.set_xticklabels(obj_labels, rotation=20, ha='right')
    ax.set_yticks(range(len(param_labels)))
    ax.set_yticklabels(param_labels)

    # Annotate with actual values
    for i in range(len(params)):
        for j in range(len(objectives)):
            val = matrix[i, j]
            text = f'{val:.1f}%' if val >= 1 else f'{val:.2f}%'
            color = 'white' if log_matrix[i, j] > 1.5 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    ax.set_title('Parameter Sensitivity (Range %)')
    fig.colorbar(im, ax=ax, label='log₁₀(range% + 1)', shrink=0.8)

    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_heatmap.pdf'))
    plt.close(fig)
    print('  ✓ sensitivity_heatmap.pdf')


# =============================================================================
# Figure 3: OAT Sensitivity Curves (key parameters)
# =============================================================================
def fig_oat_curves():
    oat = load_json('insights/oat_sensitivity.json')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # --- Panel A: Systolic Array → Prefill ---
    param = 'systolic_array_dim'
    data = oat[param]
    values = [d['value'] for d in data]
    prefill_lat = [d['workloads']['llama_70b_prefill']['latency_ms'] for d in data]
    ax = axes[0]
    ax.plot(values, prefill_lat, 'o-', color='#D32F2F', linewidth=2, markersize=6)
    ax.set_xlabel('Systolic Array Dimension')
    ax.set_ylabel('Prefill Latency (ms)')
    ax.set_title('(a) SA Dim → Prefill\n(Quadratic scaling)')
    ax.set_xticks(values)

    # --- Panel B: HBM Channels → Decode ---
    param = 'hbm_channel_count'
    data = oat[param]
    values = [d['value'] for d in data]
    decode_lat = [d['workloads']['llama_70b_decode']['latency_ms'] for d in data]
    ax = axes[1]
    ax.plot(values, decode_lat, 's-', color='#1976D2', linewidth=2, markersize=6)
    ax.set_xlabel('HBM Channel Count')
    ax.set_ylabel('Decode Latency (ms)')
    ax.set_title('(b) HBM Channels → Decode\n(Linear scaling)')
    ax.set_xticks(values)

    # --- Panel C: Core Count → Prefill (diminishing returns) ---
    param = 'core_count'
    data = oat[param]
    values = [d['value'] for d in data]
    prefill_lat = [d['workloads']['llama_70b_prefill']['latency_ms'] for d in data]
    ax = axes[2]
    ax.plot(values, prefill_lat, 'D-', color='#388E3C', linewidth=2, markersize=6)
    ax.set_xlabel('Core Count')
    ax.set_ylabel('Prefill Latency (ms)')
    ax.set_title('(c) Core Count → Prefill\n(Diminishing returns)')
    ax.set_xticks(values)
    # Mark the 128-core cliff
    if 128 in values and 192 in values:
        idx_128 = values.index(128)
        idx_192 = values.index(192)
        ax.axvspan(values[idx_128], values[idx_192], alpha=0.15, color='red')
        ax.annotate('↑ Negative\nreturns',
                    xy=(160, prefill_lat[idx_192]),
                    fontsize=7, ha='center', color='red')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'oat_sensitivity.pdf'))
    plt.close(fig)
    print('  ✓ oat_sensitivity.pdf')


# =============================================================================
# Figure 4: Bottleneck Classification by Phase & Model
# =============================================================================
def fig_bottleneck():
    bp = load_json('insights/bottleneck_patterns.json')

    models_data = bp['model_scaling']
    # Split by phase
    prefill = [d for d in models_data if d['phase'] == 'prefill']
    decode = [d for d in models_data if d['phase'] == 'decode']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    # --- Prefill ---
    ax = axes[0]
    model_names = [d['model'] for d in prefill]
    compute_frac = [d['compute_fraction'] for d in prefill]
    memory_frac = [d['memory_fraction'] for d in prefill]
    x = range(len(model_names))
    ax.bar(x, compute_frac, color='#EF5350', label='Compute-bound', width=0.6)
    ax.bar(x, memory_frac, bottom=compute_frac, color='#42A5F5',
           label='Memory-bound', width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of Total Latency')
    ax.set_title('(a) Prefill Phase')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0, 1.05)

    # --- Decode ---
    ax = axes[1]
    model_names = [d['model'] for d in decode]
    compute_frac = [d['compute_fraction'] for d in decode]
    memory_frac = [d['memory_fraction'] for d in decode]
    x = range(len(model_names))
    ax.bar(x, compute_frac, color='#EF5350', label='Compute-bound', width=0.6)
    ax.bar(x, memory_frac, bottom=compute_frac, color='#42A5F5',
           label='Memory-bound', width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
    ax.set_title('(b) Decode Phase')
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'bottleneck_classification.pdf'))
    plt.close(fig)
    print('  ✓ bottleneck_classification.pdf')


# =============================================================================
# Figure 5: Diminishing Returns — ROI per mm²
# =============================================================================
def fig_diminishing_returns():
    dr = load_json('insights/diminishing_returns.json')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- Panel A: Prefill ROI ---
    ax = axes[0]
    for param, color, marker, label in [
        ('systolic_array_dim', '#D32F2F', 'o', 'SA Dimension'),
        ('core_count', '#388E3C', 'D', 'Core Count'),
        ('hbm_channel_count', '#1976D2', 's', 'HBM Channels'),
    ]:
        entries = dr[param]['prefill_roi']
        if not entries:
            continue
        x_labels = [f"{e['from']}→{e['to']}" for e in entries]
        roi = [e['roi_ms_per_mm2'] for e in entries]
        ax.plot(range(len(roi)), roi, f'{marker}-', color=color, label=label,
                linewidth=1.5, markersize=5)

    ax.set_ylabel('ROI (ms/mm²)')
    ax.set_title('(a) Prefill Latency ROI')
    ax.legend(fontsize=8)
    ax.set_xlabel('Parameter Step')
    ax.set_yscale('symlog', linthresh=10)

    # --- Panel B: Decode ROI ---
    ax = axes[1]
    for param, color, marker, label in [
        ('hbm_channel_count', '#1976D2', 's', 'HBM Channels'),
        ('core_count', '#388E3C', 'D', 'Core Count'),
    ]:
        entries = dr[param]['decode_roi']
        if not entries:
            continue
        roi = [e['roi_ms_per_mm2'] for e in entries]
        ax.plot(range(len(roi)), roi, f'{marker}-', color=color, label=label,
                linewidth=1.5, markersize=5)

    ax.set_ylabel('ROI (ms/mm²)')
    ax.set_title('(b) Decode Latency ROI')
    ax.legend(fontsize=8)
    ax.set_xlabel('Parameter Step')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'diminishing_returns.pdf'))
    plt.close(fig)
    print('  ✓ diminishing_returns.pdf')


# =============================================================================
# Figure 6: Operator Latency Breakdown (stacked bar for prefill & decode)
# =============================================================================
def fig_operator_breakdown():
    bp = load_json('insights/bottleneck_patterns.json')

    # Pick llama_70b
    prefill_entry = None
    decode_entry = None
    for d in bp['model_scaling']:
        if 'llama_70b' in d['model'].lower().replace('-', '_').replace(' ', '_'):
            if d['phase'] == 'prefill':
                prefill_entry = d
            elif d['phase'] == 'decode':
                decode_entry = d

    if not prefill_entry or not decode_entry:
        for d in bp['model_scaling']:
            if d['phase'] == 'prefill':
                prefill_entry = d
            elif d['phase'] == 'decode':
                decode_entry = d

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for idx, (entry, title) in enumerate([(prefill_entry, 'Prefill'),
                                           (decode_entry, 'Decode')]):
        ax = axes[idx]
        ops = entry['operators']
        # Sort by latency descending
        ops_sorted = sorted(ops, key=lambda o: o['latency'], reverse=True)
        names = [o['name'] for o in ops_sorted]
        latencies = [o['latency'] for o in ops_sorted]
        total = sum(latencies)
        fracs = [l / total * 100 for l in latencies]
        bottlenecks = [o['bottleneck'] for o in ops_sorted]

        colors = []
        for b in bottlenecks:
            if 'compute' in b:
                colors.append('#EF5350')
            elif 'communication' in b:
                colors.append('#FFA726')
            else:
                colors.append('#42A5F5')

        y_pos = np.arange(len(names))
        ax.barh(y_pos, fracs, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('% of Total Latency')
        ax.set_title(f'({chr(97+idx)}) {entry.get("model", "LLaMA-70B")} — {title}')
        ax.invert_yaxis()

        # Percentage labels
        for i, frac in enumerate(fracs):
            if frac > 3:
                ax.text(frac + 0.5, i, f'{frac:.1f}%', va='center', fontsize=7)

    # Simple legend inside the figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF5350', label='Compute-bound'),
        Patch(facecolor='#42A5F5', label='Memory-bound'),
        Patch(facecolor='#FFA726', label='Communication'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'operator_breakdown.pdf'))
    plt.close(fig)
    print('  ✓ operator_breakdown.pdf')


# =============================================================================
# Figure 7: Area-efficiency comparison (SA dim vs Core count)
# =============================================================================
def fig_area_efficiency():
    oat = load_json('insights/oat_sensitivity.json')
    dr = load_json('insights/diminishing_returns.json')

    fig, ax = plt.subplots(figsize=(5, 4))

    # SA dim steps
    sa_entries = dr['systolic_array_dim']['prefill_roi']
    core_entries = dr['core_count']['prefill_roi']

    categories = []
    rois = []
    colors_list = []

    for e in sa_entries:
        categories.append(f"SA {e['from']}→{e['to']}")
        rois.append(abs(e['roi_ms_per_mm2']))
        colors_list.append('#D32F2F')

    for e in core_entries:
        categories.append(f"Core {e['from']}→{e['to']}")
        rois.append(abs(e['roi_ms_per_mm2']))
        colors_list.append('#388E3C')

    y = range(len(categories))
    ax.barh(y, rois, color=colors_list, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlabel('|ROI| (ms/mm²)')
    ax.set_title('Area Efficiency: SA Dim vs Core Count (Prefill)')
    ax.set_xscale('log')
    ax.invert_yaxis()

    # Annotate the 31.7x gap
    if sa_entries and core_entries:
        sa_best = abs(sa_entries[0]['roi_ms_per_mm2'])
        core_best = abs(core_entries[0]['roi_ms_per_mm2'])
        ratio = sa_best / core_best if core_best > 0 else 0
        ax.annotate(f'{ratio:.0f}× more\narea-efficient',
                    xy=(sa_best * 0.5, 0),
                    fontsize=9, fontweight='bold', color='#D32F2F',
                    ha='center', va='bottom')

    fig.savefig(os.path.join(FIGURES_DIR, 'area_efficiency.pdf'))
    plt.close(fig)
    print('  ✓ area_efficiency.pdf')


# =============================================================================
# Figure 8: Batch Size & Sequence Length Scaling
# =============================================================================
def fig_scaling():
    bp = load_json('insights/bottleneck_patterns.json')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- Batch size scaling ---
    ax = axes[0]
    bs_data = bp['batch_size_scaling']
    for phase, color, marker in [('prefill', '#D32F2F', 'o'),
                                  ('decode', '#1976D2', 's')]:
        pts = [d for d in bs_data if d['phase'] == phase]
        pts.sort(key=lambda d: d['batch_size'])
        bs = [d['batch_size'] for d in pts]
        lat = [d['latency_ms'] for d in pts]
        ax.plot(bs, lat, f'{marker}-', color=color, label=phase.capitalize(),
                linewidth=1.5, markersize=5)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('(a) Batch Size Scaling')
    ax.legend()
    ax.set_xscale('log', base=2)

    # --- Sequence length scaling ---
    ax = axes[1]
    sl_data = bp['seq_len_scaling']
    for phase, color, marker in [('prefill', '#D32F2F', 'o'),
                                  ('decode', '#1976D2', 's')]:
        pts = [d for d in sl_data if d['phase'] == phase]
        pts.sort(key=lambda d: d['seq_len'])
        sl = [d['seq_len'] for d in pts]
        lat = [d['latency_ms'] for d in pts]
        ax.plot(sl, lat, f'{marker}-', color=color, label=phase.capitalize(),
                linewidth=1.5, markersize=5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('(b) Sequence Length Scaling')
    ax.legend()
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'workload_scaling.pdf'))
    plt.close(fig)
    print('  ✓ workload_scaling.pdf')


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('Generating paper figures...')
    print(f'  Results dir: {os.path.abspath(RESULTS_DIR)}')
    print(f'  Figures dir: {os.path.abspath(FIGURES_DIR)}')
    print()

    funcs = [
        ('pareto', fig_pareto),
        ('sensitivity', fig_sensitivity_heatmap),
        ('oat', fig_oat_curves),
        ('bottleneck', fig_bottleneck),
        ('diminishing', fig_diminishing_returns),
        ('operator', fig_operator_breakdown),
        ('area_eff', fig_area_efficiency),
        ('scaling', fig_scaling),
    ]
    # Allow running specific figures via CLI args
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for name, func in funcs:
        if only and name not in only:
            continue
        func()

    print('\nAll figures generated successfully!')
