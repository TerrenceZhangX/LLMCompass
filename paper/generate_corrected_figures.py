#!/usr/bin/env python3
"""Generate corrected publication-quality figures (post critical thinking)."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def find_design(designs, cores=64, sdim=16, hbm=4, sram=128, gbuf=60):
    """Find a specific design point by parameter values."""
    for d in designs:
        if (d['core_count'] == cores and d['systolic_array_dim'] == sdim and 
            d['hbm_channel_count'] == hbm and d['sram_KB'] == sram and
            d['global_buffer_MB'] == gbuf):
            return d
    return None


# =============================================================================
# Figure: Corrected Sensitivity Heatmap (Controlled Single-Param Sweeps)
# =============================================================================
def fig_corrected_sensitivity():
    designs = load_json('dse/all_designs.json')
    
    # Controlled single-parameter sweep with baseline = (64, 16, 4, 128, 60)
    params = {
        'systolic_array_dim': [8, 16, 32],
        'core_count': [32, 48, 64, 96, 128],
        'hbm_channel_count': [3, 4, 5, 6, 8],
        'global_buffer_MB': [20, 40, 60, 80, 120],
        'sram_KB': [64, 128, 192, 256, 512],
    }
    param_labels = ['SA Dimension', 'Core Count', 'HBM Channels',
                    'Global Buffer', 'SRAM']
    obj_labels = ['Prefill Latency', 'Decode Latency', 'Die Area']
    
    matrix = np.zeros((5, 3))
    
    for i, (param, vals) in enumerate(params.items()):
        baseline = {'cores': 64, 'sdim': 16, 'hbm': 4, 'sram': 128, 'gbuf': 60}
        key_map = {'systolic_array_dim': 'sdim', 'core_count': 'cores',
                   'hbm_channel_count': 'hbm', 'sram_KB': 'sram', 'global_buffer_MB': 'gbuf'}
        
        prefills, decodes, areas = [], [], []
        for v in vals:
            kw = dict(baseline)
            kw[key_map[param]] = v
            d = find_design(designs, **kw)
            if d:
                prefills.append(d['prefill_latency_ms'])
                decodes.append(d['decode_latency_ms'])
                areas.append(d['die_area_mm2'])
        
        if prefills:
            pf_range = (max(prefills) - min(prefills)) / min(prefills) * 100
            dc_range = (max(decodes) - min(decodes)) / min(decodes) * 100
            ar_range = (max(areas) - min(areas)) / min(areas) * 100
            matrix[i] = [pf_range, dc_range, ar_range]

    # Log scale for visualization
    log_matrix = np.log10(matrix + 1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(obj_labels, rotation=20, ha='right')
    ax.set_yticks(range(5))
    ax.set_yticklabels(param_labels)
    
    for i in range(5):
        for j in range(3):
            val = matrix[i, j]
            if val < 0.01:
                text = '0.0%'
            elif val >= 1:
                text = f'{val:.1f}%'
            else:
                text = f'{val:.2f}%'
            color = 'white' if log_matrix[i, j] > 1.5 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)
    
    ax.set_title('Controlled Single-Parameter Sensitivity')
    fig.colorbar(im, ax=ax, label='log₁₀(range% + 1)', shrink=0.8)
    
    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_heatmap_corrected.pdf'))
    plt.close(fig)
    print('  ✓ sensitivity_heatmap_corrected.pdf')


# =============================================================================
# Figure: Decode Latency Formula (NEW)
# =============================================================================
def fig_decode_formula():
    designs = load_json('dse/all_designs.json')
    
    hbm_vals = [3, 4, 5, 6, 8]
    bws, lats = [], []
    
    for hbm in hbm_vals:
        d = find_design(designs, hbm=hbm)
        if d:
            bws.append(d['hbm_bandwidth_TBs'])
            lats.append(d['decode_latency_ms'])
    
    bws = np.array(bws)
    lats = np.array(lats)
    
    # Fit: latency = A/BW + B
    A_mat = np.vstack([1.0/bws, np.ones(len(bws))]).T
    result = np.linalg.lstsq(A_mat, lats, rcond=None)
    A, B = result[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Data + Fit
    bw_smooth = np.linspace(0.8, 5, 100)
    lat_fit = A / bw_smooth + B
    
    ax1.plot(bw_smooth, lat_fit, '-', color='#D32F2F', linewidth=2, 
             label=f'$T = {A:.1f}/BW + {B:.1f}$ ms', zorder=3)
    ax1.scatter(bws, lats, s=80, c='#1976D2', edgecolors='black', 
                linewidths=1, zorder=4, label='Measured')
    ax1.axhline(y=B, color='gray', linestyle='--', alpha=0.7, 
                label=f'Compute floor = {B:.1f} ms')
    ax1.fill_between(bw_smooth, B, lat_fit, alpha=0.15, color='#1976D2',
                     label='BW-dependent component')
    ax1.fill_between(bw_smooth, 0, B, alpha=0.15, color='#FF9800',
                     label='Compute floor (BW-independent)')
    
    ax1.set_xlabel('HBM Bandwidth (TB/s)')
    ax1.set_ylabel('Decode Latency (ms)')
    ax1.set_title('(a) Decode Latency vs HBM Bandwidth')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.set_xlim(0.8, 4.0)
    ax1.set_ylim(0, 90)
    
    # Panel B: Floor fraction
    floor_pct = B / lats * 100
    ax2.bar(range(len(hbm_vals)), floor_pct, color='#FF9800', edgecolor='black',
            alpha=0.8, label='Compute floor')
    ax2.bar(range(len(hbm_vals)), 100 - floor_pct, bottom=floor_pct,
            color='#1976D2', edgecolor='black', alpha=0.8, label='BW-dependent')
    
    ax2.set_xticks(range(len(hbm_vals)))
    ax2.set_xticklabels([f'HBM={h}\n({bws[i]:.1f} TB/s)' for i, h in enumerate(hbm_vals)],
                        fontsize=8)
    ax2.set_ylabel('Fraction of Decode Latency (%)')
    ax2.set_title('(b) Compute Floor Dominance')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)
    
    for i, pct in enumerate(floor_pct):
        ax2.text(i, pct / 2, f'{pct:.0f}%', ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'decode_formula.pdf'))
    plt.close(fig)
    print('  ✓ decode_formula.pdf')


# =============================================================================
# Figure: ROI Comparison (Systolic vs Core Count)
# =============================================================================
def fig_roi_comparison():
    designs = load_json('dse/all_designs.json')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: Prefill latency scaling
    # Systolic
    systolic_vals = [8, 16, 32]
    sys_pf = []
    sys_area = []
    for s in systolic_vals:
        d = find_design(designs, sdim=s)
        if d:
            sys_pf.append(d['prefill_latency_ms'])
            sys_area.append(d['die_area_mm2'])
    
    # Core count
    core_vals = [32, 48, 64, 96, 128]
    core_pf = []
    core_area = []
    for c in core_vals:
        d = find_design(designs, cores=c)
        if d:
            core_pf.append(d['prefill_latency_ms'])
            core_area.append(d['die_area_mm2'])
    
    ax1.plot(sys_area, sys_pf, 'o-', color='#D32F2F', linewidth=2, markersize=8,
             label='Systolic Array Dim', zorder=3)
    for i, s in enumerate(systolic_vals):
        ax1.annotate(f'{s}×{s}', (sys_area[i], sys_pf[i]), 
                     textcoords='offset points', xytext=(10, 5), fontsize=8)
    
    ax1.plot(core_area, core_pf, 's-', color='#388E3C', linewidth=2, markersize=8,
             label='Core Count', zorder=3)
    for i, c in enumerate(core_vals):
        ax1.annotate(f'{c}', (core_area[i], core_pf[i]),
                     textcoords='offset points', xytext=(10, 5), fontsize=8)
    
    ax1.set_xlabel('Die Area (mm²)')
    ax1.set_ylabel('Prefill Latency (ms)')
    ax1.set_title('(a) Prefill Latency vs Die Area')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Panel B: ROI bar chart
    sys_roi = (sys_pf[0] / sys_pf[-1]) / ((sys_area[-1] - sys_area[0]) / sys_area[0])
    core_roi = (core_pf[0] / core_pf[-1]) / ((core_area[-1] - core_area[0]) / core_area[0])
    
    bars = ax2.bar(['Systolic\n(8→32)', 'Core Count\n(32→128)'],
                   [sys_roi, core_roi],
                   color=['#D32F2F', '#388E3C'], edgecolor='black', alpha=0.8)
    
    ax2.set_ylabel('ROI (Speedup / Area Fraction)')
    ax2.set_title(f'(b) Area-Efficiency: Systolic is {sys_roi/core_roi:.0f}× better')
    
    for bar, val in zip(bars, [sys_roi, core_roi]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'roi_comparison.pdf'))
    plt.close(fig)
    print('  ✓ roi_comparison.pdf')


# =============================================================================
# Figure: On-Chip Memory Zero Effect (model limitation)
# =============================================================================
def fig_memory_zero_effect():
    designs = load_json('dse/all_designs.json')
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Panel A: SRAM sweep
    sram_vals = [64, 128, 192, 256, 512]
    pf_sram = []
    area_sram = []
    for s in sram_vals:
        d = find_design(designs, sram=s)
        if d:
            pf_sram.append(d['prefill_latency_ms'])
            area_sram.append(d['die_area_mm2'])
    
    ax = axes[0]
    ax.plot(sram_vals, pf_sram, 'o-', color='#D32F2F', linewidth=2, markersize=8)
    ax.set_xlabel('SRAM per Core (KB)')
    ax.set_ylabel('Prefill Latency (ms)')
    ax.set_title('(a) SRAM → Prefill\n(Zero effect)')
    ymin, ymax = min(pf_sram) * 0.99, max(pf_sram) * 1.01
    ax.set_ylim(ymin, ymax)
    ax.axhline(y=pf_sram[0], color='gray', linestyle='--', alpha=0.5)
    ax.text(0.5, 0.95, 'FLAT: Zero latency change', transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Panel B: Global Buffer sweep
    gbuf_vals = [20, 40, 60, 80, 120]
    pf_gbuf = []
    for g in gbuf_vals:
        d = find_design(designs, gbuf=g)
        if d:
            pf_gbuf.append(d['prefill_latency_ms'])
    
    ax = axes[1]
    ax.plot(gbuf_vals, pf_gbuf, 's-', color='#1976D2', linewidth=2, markersize=8)
    ax.set_xlabel('Global Buffer (MB)')
    ax.set_ylabel('Prefill Latency (ms)')
    ax.set_title('(b) Global Buffer → Prefill\n(Zero effect)')
    ymin, ymax = min(pf_gbuf) * 0.99, max(pf_gbuf) * 1.01
    ax.set_ylim(ymin, ymax)
    ax.axhline(y=pf_gbuf[0], color='gray', linestyle='--', alpha=0.5)
    ax.text(0.5, 0.95, 'FLAT: Zero latency change', transform=ax.transAxes,
            ha='center', va='top', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Panel C: Area-only impact
    ax = axes[2]
    ax.plot(sram_vals, area_sram, 'o-', color='#D32F2F', linewidth=2, markersize=8,
            label='SRAM')
    gbuf_area = []
    for g in gbuf_vals:
        d = find_design(designs, gbuf=g)
        if d:
            gbuf_area.append(d['die_area_mm2'])
    ax.plot(gbuf_vals, gbuf_area, 's-', color='#1976D2', linewidth=2, markersize=8,
            label='Global Buffer')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Die Area (mm²)')
    ax.set_title('(c) Area Increase Only\n(Cost with no benefit)')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'memory_zero_effect.pdf'))
    plt.close(fig)
    print('  ✓ memory_zero_effect.pdf')


if __name__ == '__main__':
    print('Generating corrected figures...')
    print(f'  Results dir: {os.path.abspath(RESULTS_DIR)}')
    print(f'  Figures dir: {os.path.abspath(FIGURES_DIR)}')
    print()

    funcs = [
        ('sensitivity', fig_corrected_sensitivity),
        ('decode', fig_decode_formula),
        ('roi', fig_roi_comparison),
        ('memory', fig_memory_zero_effect),
    ]
    
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for name, func in funcs:
        if only and name not in only:
            continue
        func()

    print('\nAll corrected figures generated!')
