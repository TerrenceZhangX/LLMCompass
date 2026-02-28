#!/usr/bin/env python3
"""Milestone 3: Architectural Insights Extraction from profiling & DSE data.

This module performs rigorous analysis with controlled experiments:
1. OAT (One-At-a-Time) sensitivity with proper baselines
2. Cross-workload bottleneck pattern analysis
3. Hardware evolution validation
4. Diminishing returns analysis

All analysis uses REAL LLMCompass evaluations, not mock data.
"""

import copy
import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

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
# Workload definitions
# ---------------------------------------------------------------------------

MODELS = {
    "gpt2_small":  {"d_model": 768,   "n_heads": 12, "n_layers": 12},
    "llama_7b":    {"d_model": 4096,  "n_heads": 32, "n_layers": 32},
    "llama_13b":   {"d_model": 5120,  "n_heads": 40, "n_layers": 40},
    "llama_70b":   {"d_model": 8192,  "n_heads": 64, "n_layers": 80},
    "gpt3_175b":   {"d_model": 12288, "n_heads": 96, "n_layers": 96},
}

# A100-like baseline
BASELINE_PARAMS = {
    "core_count": 108,
    "hbm_channel_count": 5,
    "global_buffer_MB": 48,
    "systolic_array_dim": 16,
    "sram_KB": 192,
    "device_count": 1,
}

# OAT sweep ranges — each parameter varied independently from baseline
OAT_SWEEPS = {
    "core_count":         [32, 48, 64, 96, 108, 128, 160, 192],
    "hbm_channel_count":  [3, 4, 5, 6, 8, 10, 12],
    "global_buffer_MB":   [20, 40, 48, 60, 80, 120],
    "systolic_array_dim": [8, 16, 32, 64, 128],
    "sram_KB":            [64, 128, 192, 256, 512],
}


def make_config(baseline_path: str, params: dict) -> dict:
    """Build LLMCompass config from parameter dict."""
    cfg = read_architecture_template(baseline_path)
    cfg["device"]["compute_chiplet"]["core_count"] = params["core_count"]
    cfg["device"]["compute_chiplet"]["physical_core_count"] = params["core_count"]
    cfg["device"]["compute_chiplet"]["core"]["systolic_array"]["array_height"] = params["systolic_array_dim"]
    cfg["device"]["compute_chiplet"]["core"]["systolic_array"]["array_width"] = params["systolic_array_dim"]
    cfg["device"]["compute_chiplet"]["core"]["SRAM_KB"] = params["sram_KB"]
    cfg["device"]["io"]["global_buffer_MB"] = params["global_buffer_MB"]
    cfg["device"]["io"]["physical_global_buffer_MB"] = params["global_buffer_MB"]
    cfg["device"]["io"]["global_buffer_bandwidth_per_cycle_byte"] = int(
        5120 * params["global_buffer_MB"] / 48
    )
    cfg["device"]["io"]["memory_channel_active_count"] = params["hbm_channel_count"]
    cfg["device"]["io"]["memory_channel_physical_count"] = params["hbm_channel_count"]
    cfg["device"]["memory"]["total_capacity_GB"] = params["hbm_channel_count"] * 16
    cfg["device_count"] = params.get("device_count", 1)
    if params.get("device_count", 1) <= 4:
        cfg["interconnect"]["topology"] = "FC"
    else:
        cfg["interconnect"]["topology"] = "RING"
    return cfg


def evaluate_workload(system, d_model, n_heads, n_layers,
                      batch_size, seq_len, phase, device_count) -> dict:
    """Run a single workload and return detailed results."""
    fp16 = data_type_dict["fp16"]
    if n_heads % device_count != 0:
        return {"latency": float("inf"), "error": "n_heads not divisible by device_count"}

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

        # Extract operator-level data
        operators = []
        op_names = [
            "qkv_proj", "q_mul_k", "a_mul_v", "h_matmul0",
            "h_matmul1", "h_matmul2", "softmax",
            "layernorm0", "layernorm1", "gelu",
            "allreduce_mha", "allreduce_ffn"
        ]
        op_objects = [
            model.Q_proj, model.Q_mul_K, model.A_mul_V, model.H_matmul0,
            model.H_matmul1, model.H_matmul2, model.A_softmax,
            model.layer_norm0, model.layer_norm1, model.H_gelu,
            model.allreduce_mha, model.allreduce_ffn
        ]
        roofline_latencies = model.roofline_log.split(", ") if model.roofline_log else []

        device = system.device
        peak_systolic = device.compute_module.total_systolic_array_flops
        peak_vector = device.compute_module.total_vector_flops
        peak_bw = device.io_module.bandwidth

        for i, (name, op) in enumerate(zip(op_names, op_objects)):
            op_lat = float(roofline_latencies[i]) if i < len(roofline_latencies) else 0
            flop_count = getattr(op, "flop_count", 0) or 0
            io_count = getattr(op, "io_count", 0) or 0

            # Determine io_bytes
            word_size = fp16.word_size
            if name in ["softmax", "layernorm0", "layernorm1", "gelu"]:
                io_bytes = io_count  # already in bytes for vector ops
            else:
                io_bytes = io_count * word_size

            # Compute times
            if name in ["allreduce_mha", "allreduce_ffn"]:
                compute_time = 0
                memory_time = op_lat
            elif name in ["softmax", "layernorm0", "layernorm1", "gelu"]:
                compute_time = flop_count / peak_vector if peak_vector > 0 else 0
                memory_time = io_bytes / peak_bw if peak_bw > 0 else 0
            else:
                compute_time = flop_count / peak_systolic if peak_systolic > 0 else 0
                memory_time = io_bytes / peak_bw if peak_bw > 0 else 0

            ai = flop_count / io_bytes if io_bytes > 0 else float("inf")
            bottleneck = "compute_bound" if compute_time >= memory_time else "memory_bound"
            if name in ["allreduce_mha", "allreduce_ffn"]:
                bottleneck = "communication_bound"

            operators.append({
                "name": name,
                "latency": op_lat,
                "flop_count": flop_count,
                "io_bytes": io_bytes,
                "arithmetic_intensity": ai,
                "compute_time": compute_time,
                "memory_time": memory_time,
                "bottleneck": bottleneck,
            })

        compute_total = sum(o["compute_time"] for o in operators if o["bottleneck"] != "communication_bound")
        memory_total = sum(o["memory_time"] for o in operators if o["bottleneck"] != "communication_bound")
        non_comm_lat = sum(o["latency"] for o in operators if o["bottleneck"] != "communication_bound")

        return {
            "latency": latency * n_layers,
            "per_layer_latency": latency,
            "operators": operators,
            "compute_fraction": compute_total / (compute_total + memory_total) if (compute_total + memory_total) > 0 else 0,
            "memory_fraction": memory_total / (compute_total + memory_total) if (compute_total + memory_total) > 0 else 0,
        }
    except Exception as e:
        return {"latency": float("inf"), "error": str(e)}


def compute_area(cfg: dict) -> Tuple[float, float, float]:
    """Returns (total, compute, io) area in mm²."""
    try:
        compute = calc_compute_chiplet_area_mm2(cfg)
        io_area = calc_io_die_area_mm2(cfg)
        return compute + io_area, compute, io_area
    except Exception:
        return float("inf"), 0, 0


# ===================================================================
# Analysis 1: Controlled OAT Sensitivity
# ===================================================================

def run_oat_sensitivity(baseline_path: str) -> dict:
    """
    One-At-a-Time sensitivity: vary each parameter from baseline while
    holding all others constant. This avoids the confounding/sampling bias
    of marginal averages from the grid search.
    """
    print("=" * 70)
    print("ANALYSIS 1: One-At-a-Time Sensitivity (controlled)")
    print("=" * 70)

    # Standard workloads for comparison
    workloads = [
        {"name": "llama_7b_prefill",  "model": "llama_7b",  "bs": 1, "sl": 2048, "phase": "prefill"},
        {"name": "llama_7b_decode",   "model": "llama_7b",  "bs": 1, "sl": 2048, "phase": "decode"},
        {"name": "llama_70b_prefill", "model": "llama_70b", "bs": 1, "sl": 2048, "phase": "prefill"},
        {"name": "llama_70b_decode",  "model": "llama_70b", "bs": 1, "sl": 2048, "phase": "decode"},
    ]

    results = {}

    for param_name, sweep_values in OAT_SWEEPS.items():
        print(f"\n--- Sweeping {param_name}: {sweep_values} ---")
        param_results = []

        for val in sweep_values:
            params = dict(BASELINE_PARAMS)
            params[param_name] = val

            cfg = make_config(baseline_path, params)
            total_area, comp_area, io_area = compute_area(cfg)

            # Skip if area exceeds 1200mm² (generous limit for OAT)
            if total_area > 1200:
                print(f"  {param_name}={val}: area={total_area:.0f}mm² SKIPPED (>1200)")
                continue

            try:
                system = template_to_system(cfg)
            except Exception as e:
                print(f"  {param_name}={val}: system build failed: {e}")
                continue

            point = {
                "value": val,
                "die_area_mm2": total_area,
                "compute_area_mm2": comp_area,
                "io_area_mm2": io_area,
                "workloads": {},
            }

            for wl in workloads:
                m = MODELS[wl["model"]]
                res = evaluate_workload(
                    system, m["d_model"], m["n_heads"], m["n_layers"],
                    wl["bs"], wl["sl"], wl["phase"], params.get("device_count", 1)
                )
                point["workloads"][wl["name"]] = {
                    "latency_ms": res["latency"] * 1000 if res["latency"] < float("inf") else None,
                    "compute_fraction": res.get("compute_fraction"),
                    "memory_fraction": res.get("memory_fraction"),
                }

            param_results.append(point)
            print(f"  {param_name}={val}: area={total_area:.0f}mm², "
                  f"7B_pf={point['workloads']['llama_7b_prefill']['latency_ms']:.1f}ms, "
                  f"7B_dc={point['workloads']['llama_7b_decode']['latency_ms']:.3f}ms")

        results[param_name] = param_results

    return results


# ===================================================================
# Analysis 2: Cross-Workload Bottleneck Patterns
# ===================================================================

def run_bottleneck_analysis(baseline_path: str) -> dict:
    """
    Analyze how bottleneck patterns shift across:
    - Model sizes (7B → 70B)
    - Phases (prefill vs decode)
    - Batch sizes (1 → 64)
    - Sequence lengths (128 → 8192)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Cross-Workload Bottleneck Patterns")
    print("=" * 70)

    params = dict(BASELINE_PARAMS)
    cfg = make_config(baseline_path, params)
    system = template_to_system(cfg)

    results = {}

    # 2a: Model size scaling
    print("\n--- Model size scaling (bs=1, sl=2048) ---")
    model_results = []
    for model_name in ["gpt2_small", "llama_7b", "llama_13b", "llama_70b", "gpt3_175b"]:
        m = MODELS[model_name]
        for phase in ["prefill", "decode"]:
            res = evaluate_workload(system, m["d_model"], m["n_heads"], m["n_layers"],
                                    1, 2048, phase, 1)
            entry = {
                "model": model_name,
                "phase": phase,
                "latency_ms": res["latency"] * 1000 if res["latency"] < float("inf") else None,
                "compute_fraction": res.get("compute_fraction"),
                "memory_fraction": res.get("memory_fraction"),
                "operators": res.get("operators", []),
            }
            model_results.append(entry)
            print(f"  {model_name} {phase}: lat={entry['latency_ms']:.3f}ms, "
                  f"compute={entry['compute_fraction']:.1%}")
    results["model_scaling"] = model_results

    # 2b: Batch size effect  
    print("\n--- Batch size effect (llama_7b, sl=2048) ---")
    m = MODELS["llama_7b"]
    bs_results = []
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        for phase in ["prefill", "decode"]:
            res = evaluate_workload(system, m["d_model"], m["n_heads"], m["n_layers"],
                                    bs, 2048, phase, 1)
            entry = {
                "batch_size": bs,
                "phase": phase,
                "latency_ms": res["latency"] * 1000 if res["latency"] < float("inf") else None,
                "compute_fraction": res.get("compute_fraction"),
            }
            bs_results.append(entry)
            if entry["latency_ms"]:
                print(f"  bs={bs} {phase}: lat={entry['latency_ms']:.3f}ms, "
                      f"compute={entry['compute_fraction']:.1%}")
    results["batch_size_scaling"] = bs_results

    # 2c: Sequence length effect
    print("\n--- Sequence length effect (llama_7b, bs=1) ---")
    sl_results = []
    for sl in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for phase in ["prefill", "decode"]:
            res = evaluate_workload(system, m["d_model"], m["n_heads"], m["n_layers"],
                                    1, sl, phase, 1)
            entry = {
                "seq_len": sl,
                "phase": phase,
                "latency_ms": res["latency"] * 1000 if res["latency"] < float("inf") else None,
                "compute_fraction": res.get("compute_fraction"),
            }
            sl_results.append(entry)
            if entry["latency_ms"]:
                print(f"  sl={sl} {phase}: lat={entry['latency_ms']:.3f}ms, "
                      f"compute={entry['compute_fraction']:.1%}")
    results["seq_len_scaling"] = sl_results

    return results


# ===================================================================
# Analysis 3: Hardware Evolution Validation
# ===================================================================

# Approximate real GPU configs
REAL_GPUS = {
    "A100": {
        "core_count": 108, "hbm_channel_count": 5,
        "global_buffer_MB": 48, "systolic_array_dim": 16,
        "sram_KB": 192, "device_count": 1,
        # Real specs for comparison
        "real_tflops_fp16": 312, "real_hbm_bw_TBs": 2.0,
        "real_die_area_mm2": 826,
    },
    "H100": {
        "core_count": 132, "hbm_channel_count": 8,
        "global_buffer_MB": 50, "systolic_array_dim": 32,
        "sram_KB": 256, "device_count": 1,
        "real_tflops_fp16": 989, "real_hbm_bw_TBs": 3.35,
        "real_die_area_mm2": 814,
    },
}


def run_hw_evolution_validation(baseline_path: str) -> dict:
    """Compare LLMCompass predictions across GPU generations."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Hardware Evolution Validation")
    print("=" * 70)

    results = {}
    workloads = [
        {"name": "llama_7b_prefill",  "model": "llama_7b",  "bs": 1, "sl": 2048, "phase": "prefill"},
        {"name": "llama_7b_decode",   "model": "llama_7b",  "bs": 1, "sl": 2048, "phase": "decode"},
        {"name": "llama_70b_prefill", "model": "llama_70b", "bs": 1, "sl": 2048, "phase": "prefill"},
        {"name": "llama_70b_decode",  "model": "llama_70b", "bs": 1, "sl": 2048, "phase": "decode"},
    ]

    for gpu_name, gpu_params in REAL_GPUS.items():
        print(f"\n--- {gpu_name} ---")
        params = {k: v for k, v in gpu_params.items() if not k.startswith("real_")}
        cfg = make_config(baseline_path, params)
        total_area, comp_area, io_area = compute_area(cfg)

        try:
            system = template_to_system(cfg)
        except Exception as e:
            print(f"  Failed to build system: {e}")
            continue

        peak_tflops = system.device.compute_module.total_systolic_array_flops / 1e12
        peak_bw = system.device.io_module.bandwidth / 1e12

        gpu_result = {
            "modeled_area_mm2": total_area,
            "real_area_mm2": gpu_params.get("real_die_area_mm2"),
            "modeled_tflops": peak_tflops,
            "real_tflops": gpu_params.get("real_tflops_fp16"),
            "modeled_bw_TBs": peak_bw,
            "real_bw_TBs": gpu_params.get("real_hbm_bw_TBs"),
            "workloads": {},
        }

        print(f"  Area: modeled={total_area:.0f}mm² real={gpu_params.get('real_die_area_mm2')}mm²")
        print(f"  TFLOPS: modeled={peak_tflops:.1f} real={gpu_params.get('real_tflops_fp16')}")
        print(f"  BW: modeled={peak_bw:.2f}TB/s real={gpu_params.get('real_hbm_bw_TBs')}TB/s")

        for wl in workloads:
            m = MODELS[wl["model"]]
            res = evaluate_workload(system, m["d_model"], m["n_heads"], m["n_layers"],
                                    wl.get("bs", 1), wl.get("sl", 2048), wl["phase"], 1)
            gpu_result["workloads"][wl["name"]] = {
                "latency_ms": res["latency"] * 1000 if res["latency"] < float("inf") else None,
                "compute_fraction": res.get("compute_fraction"),
            }
            if res["latency"] < float("inf"):
                print(f"  {wl['name']}: {res['latency']*1000:.3f}ms (compute={res.get('compute_fraction', 0):.1%})")

        results[gpu_name] = gpu_result

    # Compute speedup ratios
    if "A100" in results and "H100" in results:
        print("\n--- A100 → H100 Speedup ---")
        for wl_name in results["A100"]["workloads"]:
            a100_lat = results["A100"]["workloads"][wl_name]["latency_ms"]
            h100_lat = results["H100"]["workloads"][wl_name]["latency_ms"]
            if a100_lat and h100_lat:
                speedup = a100_lat / h100_lat
                print(f"  {wl_name}: {speedup:.2f}×")
        results["speedups"] = {
            "A100_to_H100": {
                wl: results["A100"]["workloads"][wl]["latency_ms"] / results["H100"]["workloads"][wl]["latency_ms"]
                for wl in results["A100"]["workloads"]
                if results["A100"]["workloads"][wl]["latency_ms"] and results["H100"]["workloads"][wl]["latency_ms"]
            }
        }

    return results


# ===================================================================
# Analysis 4: Diminishing Returns Analysis
# ===================================================================

def run_diminishing_returns(oat_results: dict) -> dict:
    """Analyze marginal ROI for each parameter increment."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Diminishing Returns")
    print("=" * 70)

    results = {}

    for param_name, sweeps in oat_results.items():
        if len(sweeps) < 2:
            continue

        param_analysis = {"prefill_roi": [], "decode_roi": []}

        for i in range(1, len(sweeps)):
            prev = sweeps[i-1]
            curr = sweeps[i]

            area_delta = curr["die_area_mm2"] - prev["die_area_mm2"]

            for wl_key, roi_key in [("llama_70b_prefill", "prefill_roi"),
                                     ("llama_70b_decode", "decode_roi")]:
                prev_lat = prev["workloads"].get(wl_key, {}).get("latency_ms")
                curr_lat = curr["workloads"].get(wl_key, {}).get("latency_ms")
                if prev_lat and curr_lat and area_delta != 0:
                    lat_improvement_ms = prev_lat - curr_lat
                    roi = lat_improvement_ms / area_delta if area_delta > 0 else 0
                    pct_improvement = lat_improvement_ms / prev_lat * 100

                    param_analysis[roi_key].append({
                        "from": prev["value"],
                        "to": curr["value"],
                        "latency_from_ms": prev_lat,
                        "latency_to_ms": curr_lat,
                        "improvement_ms": lat_improvement_ms,
                        "improvement_pct": pct_improvement,
                        "area_delta_mm2": area_delta,
                        "roi_ms_per_mm2": roi,
                    })

        results[param_name] = param_analysis

        # Print summary
        print(f"\n--- {param_name} (LLaMA-70B prefill ROI) ---")
        for step in param_analysis["prefill_roi"]:
            print(f"  {step['from']}→{step['to']}: "
                  f"{step['improvement_pct']:+.1f}% latency, "
                  f"{step['area_delta_mm2']:+.0f}mm² area, "
                  f"ROI={step['roi_ms_per_mm2']:.2f} ms/mm²")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Architectural Insights Extraction")
    parser.add_argument("--baseline", default="configs/GA100.json")
    parser.add_argument("--output-dir", default="results/insights")
    parser.add_argument("--skip-oat", action="store_true", help="Skip OAT (use cached)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Analysis 1: OAT Sensitivity
    oat_path = os.path.join(args.output_dir, "oat_sensitivity.json")
    if args.skip_oat and os.path.exists(oat_path):
        print("Loading cached OAT results...")
        with open(oat_path) as f:
            oat_results = json.load(f)
    else:
        oat_results = run_oat_sensitivity(args.baseline)
        with open(oat_path, "w") as f:
            json.dump(oat_results, f, indent=2)
        print(f"\nOAT results saved to {oat_path}")

    # Analysis 2: Bottleneck Patterns
    bottleneck_results = run_bottleneck_analysis(args.baseline)
    bn_path = os.path.join(args.output_dir, "bottleneck_patterns.json")
    with open(bn_path, "w") as f:
        json.dump(bottleneck_results, f, indent=2)
    print(f"\nBottleneck patterns saved to {bn_path}")

    # Analysis 3: Hardware Evolution
    hw_results = run_hw_evolution_validation(args.baseline)
    hw_path = os.path.join(args.output_dir, "hw_evolution.json")
    with open(hw_path, "w") as f:
        json.dump(hw_results, f, indent=2)
    print(f"\nHW evolution results saved to {hw_path}")

    # Analysis 4: Diminishing Returns
    dr_results = run_diminishing_returns(oat_results)
    dr_path = os.path.join(args.output_dir, "diminishing_returns.json")
    with open(dr_path, "w") as f:
        json.dump(dr_results, f, indent=2)
    print(f"\nDiminishing returns saved to {dr_path}")

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print(f"Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
