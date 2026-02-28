#!/usr/bin/env python3
"""Design Space Exploration for GPU architectures targeting LLM workloads.

Uses LLMCompass roofline model for fast evaluation + cost model for die area.
Produces Pareto-optimal design points across latency and area objectives.

Usage:
    python -m dse.dse_runner [--config configs/dse_config.json]
                              [--output-dir results/dse]
"""

import argparse
import copy
import io
import itertools
import json
import os
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

_LLMCOMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, _LLMCOMPASS_ROOT)

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DesignPoint:
    """A single point in the design space with its evaluation results."""
    # Architecture parameters
    core_count: int = 0
    hbm_channel_count: int = 0
    global_buffer_MB: int = 0
    systolic_array_dim: int = 0
    sram_KB: int = 0
    device_count: int = 1

    # Computed metrics
    die_area_mm2: float = 0.0
    compute_area_mm2: float = 0.0
    io_area_mm2: float = 0.0
    peak_systolic_tflops: float = 0.0
    hbm_bandwidth_TBs: float = 0.0

    # Per-workload results: {workload_name: latency_seconds}
    workload_latencies: Dict[str, float] = field(default_factory=dict)

    # Aggregate objectives
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0

    # Pareto status
    is_pareto: bool = False
    pareto_rank: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# DSE Engine
# ---------------------------------------------------------------------------

class DSEEngine:
    """Grid-search DSE engine with Pareto analysis."""

    def __init__(self, dse_config_path: str, baseline_config_path: str = None):
        with open(dse_config_path, "r") as f:
            self.dse_config = json.load(f)

        baseline_path = baseline_config_path or self.dse_config.get("baseline", "configs/GA100.json")
        self.baseline_arch = read_architecture_template(baseline_path)
        self.workloads = self.dse_config["workloads"]
        self.constraints = self.dse_config.get("constraints", {})
        self.max_area = self.constraints.get("max_die_area_mm2", 900)

        # Build parameter grid
        params = self.dse_config["parameters"]
        self.param_names = list(params.keys())
        self.param_values = [params[p]["values"] for p in self.param_names]

        # Fixed params
        fixed = self.dse_config.get("fixed_parameters", {})
        self.device_counts = fixed.get("device_counts", [1, 4, 8])

    def _make_arch_config(self, core_count, hbm_channels, global_buffer_MB,
                          sa_dim, sram_KB, device_count) -> dict:
        """Create a modified architecture config from baseline + DSE parameters."""
        cfg = copy.deepcopy(self.baseline_arch)

        # Core count
        cfg["device"]["compute_chiplet"]["core_count"] = core_count
        cfg["device"]["compute_chiplet"]["physical_core_count"] = core_count

        # Systolic array
        cfg["device"]["compute_chiplet"]["core"]["systolic_array"]["array_height"] = sa_dim
        cfg["device"]["compute_chiplet"]["core"]["systolic_array"]["array_width"] = sa_dim

        # SRAM
        cfg["device"]["compute_chiplet"]["core"]["SRAM_KB"] = sram_KB

        # Global buffer (L2)
        cfg["device"]["io"]["global_buffer_MB"] = global_buffer_MB
        cfg["device"]["io"]["physical_global_buffer_MB"] = global_buffer_MB
        # Scale L2 bandwidth proportionally to size (A100: 48MB → 5120 B/cycle)
        cfg["device"]["io"]["global_buffer_bandwidth_per_cycle_byte"] = int(
            5120 * global_buffer_MB / 48
        )

        # HBM channels
        cfg["device"]["io"]["memory_channel_active_count"] = hbm_channels
        cfg["device"]["io"]["memory_channel_physical_count"] = hbm_channels
        cfg["device"]["memory"]["total_capacity_GB"] = hbm_channels * 16  # 16GB per HBM stack

        # Device count & topology
        cfg["device_count"] = device_count
        if device_count <= 4:
            cfg["interconnect"]["topology"] = "FC"
        else:
            cfg["interconnect"]["topology"] = "RING"

        return cfg

    def _compute_area(self, cfg: dict) -> Tuple[float, float, float]:
        """Compute die area. Returns (total, compute_chiplet, io_die)."""
        try:
            compute_area = calc_compute_chiplet_area_mm2(cfg)
            io_area = calc_io_die_area_mm2(cfg)
            return compute_area + io_area, compute_area, io_area
        except Exception:
            return float("inf"), 0, 0

    def _evaluate_workload(self, system, workload: dict) -> float:
        """Run roofline model for a single workload, return latency in seconds."""
        fp16 = data_type_dict["fp16"]
        d_model = workload["d_model"]
        n_heads = workload["n_heads"]
        n_layers = workload["n_layers"]
        batch_size = workload["batch_size"]
        seq_len = workload["seq_len"]
        phase = workload["phase"]
        device_count = workload["device_count"]

        if n_heads % device_count != 0:
            return float("inf")

        try:
            if phase == "prefill":
                model = TransformerBlockInitComputationTP(
                    d_model=d_model, n_heads=n_heads,
                    device_count=device_count, data_type=fp16
                )
                X = Tensor([batch_size, seq_len, d_model], fp16)
                model(X)
            else:
                model = TransformerBlockAutoRegressionTP(
                    d_model=d_model, n_heads=n_heads,
                    device_count=device_count, data_type=fp16
                )
                X = Tensor([batch_size, 1, d_model], fp16)
                model(X, seq_len=seq_len)

            buf = io.StringIO()
            with redirect_stdout(buf):
                latency = model.roofline_model(system)

            return latency * n_layers
        except Exception:
            return float("inf")

    def run(self) -> List[DesignPoint]:
        """Execute the full DSE grid search."""
        all_points = []
        total_combos = 1
        for vals in self.param_values:
            total_combos *= len(vals)
        total_with_dc = total_combos * len(self.device_counts)
        print(f"DSE: {total_combos} arch combos × {len(self.device_counts)} device counts = {total_with_dc} total")
        print(f"Workloads: {len(self.workloads)}")
        print(f"Max area constraint: {self.max_area} mm²")
        print()

        t0 = time.time()
        evaluated = 0
        pruned_area = 0

        for combo in itertools.product(*self.param_values):
            param_dict = dict(zip(self.param_names, combo))
            core_count = param_dict["core_count"]
            hbm_channels = param_dict["hbm_channel_count"]
            global_buffer = param_dict["global_buffer_MB"]
            sa_dim = param_dict["systolic_array_dim"]
            sram = param_dict["sram_KB"]

            for dc in self.device_counts:
                cfg = self._make_arch_config(
                    core_count, hbm_channels, global_buffer, sa_dim, sram, dc
                )

                # Area check first (cheapest filter)
                total_area, compute_area, io_area = self._compute_area(cfg)
                if total_area > self.max_area or total_area == float("inf"):
                    pruned_area += 1
                    continue

                # Build system and evaluate workloads
                try:
                    system = template_to_system(cfg)
                except Exception:
                    continue

                dp = DesignPoint(
                    core_count=core_count,
                    hbm_channel_count=hbm_channels,
                    global_buffer_MB=global_buffer,
                    systolic_array_dim=sa_dim,
                    sram_KB=sram,
                    device_count=dc,
                    die_area_mm2=total_area,
                    compute_area_mm2=compute_area,
                    io_area_mm2=io_area,
                )

                # Compute peaks for reference
                dp.peak_systolic_tflops = system.device.compute_module.total_systolic_array_flops / 1e12
                dp.hbm_bandwidth_TBs = system.device.io_module.bandwidth / 1e12

                # Evaluate each workload
                for wl in self.workloads:
                    wl_with_dc = {**wl, "device_count": dc}
                    lat = self._evaluate_workload(system, wl_with_dc)
                    dp.workload_latencies[wl["name"]] = lat

                # Compute aggregate objectives
                prefill_lats = [
                    v for k, v in dp.workload_latencies.items()
                    if "prefill" in k and v < float("inf")
                ]
                decode_lats = [
                    v for k, v in dp.workload_latencies.items()
                    if "decode" in k and v < float("inf")
                ]
                dp.prefill_latency_ms = (
                    max(prefill_lats) * 1000 if prefill_lats else float("inf")
                )
                dp.decode_latency_ms = (
                    max(decode_lats) * 1000 if decode_lats else float("inf")
                )

                all_points.append(dp)
                evaluated += 1

                if evaluated % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  Evaluated {evaluated} designs ({pruned_area} pruned by area) in {elapsed:.1f}s")

        elapsed = time.time() - t0
        print(f"\nDSE complete: {evaluated} feasible designs evaluated in {elapsed:.1f}s")
        print(f"  ({pruned_area} pruned by area constraint)")

        # Pareto analysis
        self._pareto_rank(all_points)

        pareto_count = sum(1 for p in all_points if p.is_pareto)
        print(f"  Pareto-optimal designs: {pareto_count}")

        return all_points

    def _pareto_rank(self, points: List[DesignPoint]):
        """Assign Pareto ranks. Lower is better for all objectives."""
        # Objectives: minimize prefill_latency, decode_latency, die_area
        for i, p in enumerate(points):
            p.is_pareto = True
            for j, q in enumerate(points):
                if i == j:
                    continue
                # q dominates p if q is <= p on all objectives and < on at least one
                if (q.prefill_latency_ms <= p.prefill_latency_ms and
                    q.decode_latency_ms <= p.decode_latency_ms and
                    q.die_area_mm2 <= p.die_area_mm2 and
                    (q.prefill_latency_ms < p.prefill_latency_ms or
                     q.decode_latency_ms < p.decode_latency_ms or
                     q.die_area_mm2 < p.die_area_mm2)):
                    p.is_pareto = False
                    break

    def sensitivity_analysis(self, points: List[DesignPoint]) -> Dict:
        """Analyze which parameters have the most impact on objectives."""
        params = ["core_count", "hbm_channel_count", "global_buffer_MB",
                  "systolic_array_dim", "sram_KB"]
        objectives = ["prefill_latency_ms", "decode_latency_ms", "die_area_mm2"]

        sensitivity = {}
        for param in params:
            sensitivity[param] = {}
            for obj in objectives:
                # Group points by parameter value, compute mean objective
                groups = {}
                for p in points:
                    val = getattr(p, param)
                    if val not in groups:
                        groups[val] = []
                    groups[val].append(getattr(p, obj))

                means = {v: sum(vals)/len(vals) for v, vals in groups.items()}
                if means:
                    vals = list(means.values())
                    range_pct = (max(vals) - min(vals)) / min(vals) * 100 if min(vals) > 0 else 0
                    sensitivity[param][obj] = {
                        "means": {str(k): v for k, v in sorted(means.items())},
                        "range_percent": range_pct,
                    }

        # Rank parameters by impact
        for obj in objectives:
            rankings = sorted(
                params,
                key=lambda p: sensitivity[p][obj]["range_percent"],
                reverse=True,
            )
            for rank, param in enumerate(rankings):
                sensitivity[param][obj]["rank"] = rank + 1

        return sensitivity


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(points: List[DesignPoint], sensitivity: Dict, output_dir: str):
    """Save DSE results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    # All design points
    all_path = os.path.join(output_dir, "all_designs.json")
    with open(all_path, "w") as f:
        json.dump([p.to_dict() for p in points], f, indent=2)
    print(f"All designs saved to {all_path}")

    # Pareto-optimal points
    pareto = [p.to_dict() for p in points if p.is_pareto]
    pareto_path = os.path.join(output_dir, "pareto_designs.json")
    with open(pareto_path, "w") as f:
        json.dump(pareto, f, indent=2)
    print(f"Pareto designs saved to {pareto_path} ({len(pareto)} points)")

    # Sensitivity analysis
    sens_path = os.path.join(output_dir, "sensitivity.json")
    with open(sens_path, "w") as f:
        json.dump(sensitivity, f, indent=2)
    print(f"Sensitivity analysis saved to {sens_path}")

    # Summary table
    print("\n" + "=" * 100)
    print("PARETO-OPTIMAL DESIGNS")
    print("=" * 100)
    header = (
        f"{'Cores':>6} {'HBM_ch':>6} {'L2_MB':>6} {'SA_dim':>6} {'SRAM_KB':>7} "
        f"{'DC':>3} {'Area_mm2':>9} {'Prefill_ms':>11} {'Decode_ms':>10} "
        f"{'TFLOPS':>7} {'BW_TB/s':>8}"
    )
    print(header)
    print("-" * 100)
    for p in sorted(pareto, key=lambda x: x["die_area_mm2"]):
        print(
            f"{p['core_count']:>6} {p['hbm_channel_count']:>6} "
            f"{p['global_buffer_MB']:>6} {p['systolic_array_dim']:>6} "
            f"{p['sram_KB']:>7} {p['device_count']:>3} "
            f"{p['die_area_mm2']:>9.1f} {p['prefill_latency_ms']:>11.2f} "
            f"{p['decode_latency_ms']:>10.4f} "
            f"{p['peak_systolic_tflops']:>7.1f} {p['hbm_bandwidth_TBs']:>8.3f}"
        )
    print("=" * 100)

    # Sensitivity summary
    print("\nPARAMETER SENSITIVITY RANKING")
    print("-" * 60)
    objectives = ["prefill_latency_ms", "decode_latency_ms", "die_area_mm2"]
    for obj in objectives:
        ranked = sorted(
            sensitivity.keys(),
            key=lambda p: sensitivity[p][obj]["rank"]
        )
        print(f"\n  {obj}:")
        for p in ranked:
            s = sensitivity[p][obj]
            print(f"    #{s['rank']} {p}: {s['range_percent']:.1f}% range")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run GPU Architecture DSE")
    parser.add_argument(
        "--config", default="configs/dse_config.json",
        help="Path to DSE config JSON"
    )
    parser.add_argument(
        "--output-dir", default="results/dse",
        help="Directory for results"
    )
    args = parser.parse_args()

    engine = DSEEngine(args.config)
    points = engine.run()
    sensitivity = engine.sensitivity_analysis(points)
    save_results(points, sensitivity, args.output_dir)


if __name__ == "__main__":
    main()
