#!/usr/bin/env python3
"""Batch profiling script — runs all tasks from workloads.json and saves results.

Usage (from the LLMCompass directory):
    python -m profiling.run_profiling [--config configs/GA100.json]
                                      [--workloads configs/workloads.json]
                                      [--output-dir results/profiling]
                                      [--tasks CB-01,MB-01,...]
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

_LLMCOMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, _LLMCOMPASS_ROOT)

from profiling.profiler import profile_workload
from profiling.bottleneck_report import WorkloadBottleneckReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_workloads(workloads_path: str) -> dict:
    with open(workloads_path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _print_summary_table(reports: List[WorkloadBottleneckReport]):
    """Print a concise summary of all profiling results."""
    header = (
        f"{'Task':<10} {'Model':<20} {'Phase':<8} {'BS':>4} {'SeqLen':>6} "
        f"{'GPUs':>4} {'Latency(ms)':>12} {'Bottleneck':<16}"
    )
    print("\n" + "=" * len(header))
    print("PROFILING SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in reports:
        latency_ms = r.total_latency_seconds * 1e3
        print(
            f"{r.task_id:<10} {r.model_name[:20]:<20} {r.phase:<8} "
            f"{r.batch_size:>4} {r.seq_len:>6} {r.device_count:>4} "
            f"{latency_ms:>12.4f} {r.dominant_bottleneck:<16}"
        )
    print("=" * len(header))
    print(f"Total tasks: {len(reports)}\n")


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def _run_single_task(
    task: dict,
    workloads: dict,
    config_path: str,
    output_dir: str,
) -> List[WorkloadBottleneckReport]:
    """Run a single task entry from the task_matrix.  Returns a list because
    sweep tasks (SC-01, SC-02, SZ-01, SZ-02) expand into multiple runs."""

    models = workloads["models"]
    scenarios = workloads["workload_scenarios"]

    task_id = task["id"]
    phase = task["phase"]

    # Handle sweep over device_count
    device_counts = task["device_count"]
    if not isinstance(device_counts, list):
        device_counts = [device_counts]

    # Handle sweep over model
    model_keys = task["model"]
    if not isinstance(model_keys, list):
        model_keys = [model_keys]

    reports = []
    for model_key in model_keys:
        model_cfg = models[model_key]
        scenario_key = task["scenario"]
        scenario_cfg = scenarios[scenario_key]

        for dc in device_counts:
            # Validate: n_heads must be divisible by device_count
            if model_cfg["n_heads"] % dc != 0:
                print(
                    f"  SKIP {task_id} model={model_key} dc={dc}: "
                    f"n_heads={model_cfg['n_heads']} not divisible by {dc}"
                )
                continue

            sub_id = task_id
            if len(model_keys) > 1:
                sub_id += f"_{model_key}"
            if len(device_counts) > 1:
                sub_id += f"_dc{dc}"

            print(f"  Running {sub_id} ({model_key}, {scenario_key}, {phase}, dc={dc})...")
            t0 = time.time()
            try:
                report = profile_workload(
                    config_path=config_path,
                    model_config=model_cfg,
                    workload_config=scenario_cfg,
                    phase=phase,
                    device_count=dc,
                    task_id=sub_id,
                )
                elapsed = time.time() - t0
                print(f"    Done in {elapsed:.2f}s — {report.dominant_bottleneck}")
                # Save individual result
                out_path = os.path.join(output_dir, f"{sub_id}.json")
                report.to_json(out_path)
                reports.append(report)
            except Exception as e:
                print(f"    ERROR: {e}")
    return reports


def _run_sensitivity_task(
    task: dict,
    workloads: dict,
    config_path: str,
    output_dir: str,
) -> List[WorkloadBottleneckReport]:
    """Run a sensitivity analysis task (SENS-BS or SENS-SL)."""
    models = workloads["models"]
    task_id = task["id"]
    model_key = task["model"]
    model_cfg = models[model_key]
    phases = task.get("phases", ["prefill", "decode"])
    dc = task.get("device_count", 1)

    reports = []

    if "batch_sizes" in task:
        # Batch-size sweep
        seq_len = task["seq_len"]
        for bs in task["batch_sizes"]:
            for phase in phases:
                sub_id = f"{task_id}_bs{bs}_{phase}"
                wc = {"batch_size": bs, "seq_len": seq_len}
                print(f"  Running {sub_id} (bs={bs}, seq_len={seq_len}, {phase})...")
                t0 = time.time()
                try:
                    report = profile_workload(
                        config_path=config_path,
                        model_config=model_cfg,
                        workload_config=wc,
                        phase=phase,
                        device_count=dc,
                        task_id=sub_id,
                    )
                    elapsed = time.time() - t0
                    print(f"    Done in {elapsed:.2f}s — {report.dominant_bottleneck}")
                    out_path = os.path.join(output_dir, f"{sub_id}.json")
                    report.to_json(out_path)
                    reports.append(report)
                except Exception as e:
                    print(f"    ERROR: {e}")

    elif "seq_lens" in task:
        # Sequence-length sweep
        bs = task["batch_size"]
        for sl in task["seq_lens"]:
            for phase in phases:
                sub_id = f"{task_id}_sl{sl}_{phase}"
                wc = {"batch_size": bs, "seq_len": sl}
                print(f"  Running {sub_id} (bs={bs}, seq_len={sl}, {phase})...")
                t0 = time.time()
                try:
                    report = profile_workload(
                        config_path=config_path,
                        model_config=model_cfg,
                        workload_config=wc,
                        phase=phase,
                        device_count=dc,
                        task_id=sub_id,
                    )
                    elapsed = time.time() - t0
                    print(f"    Done in {elapsed:.2f}s — {report.dominant_bottleneck}")
                    out_path = os.path.join(output_dir, f"{sub_id}.json")
                    report.to_json(out_path)
                    reports.append(report)
                except Exception as e:
                    print(f"    ERROR: {e}")

    return reports


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_path: str = "configs/GA100.json",
    workloads_path: str = "configs/workloads.json",
    output_dir: str = "results/profiling",
    task_filter: Optional[List[str]] = None,
    skip_sensitivity: bool = False,
):
    """Run all profiling tasks and save results."""

    workloads = _load_workloads(workloads_path)
    _ensure_dir(output_dir)

    all_reports: List[WorkloadBottleneckReport] = []

    # --- Core task matrix ---
    print("=" * 60)
    print("Running core task matrix...")
    print("=" * 60)
    for task in workloads["task_matrix"]:
        tid = task["id"]
        if task_filter and tid not in task_filter:
            continue
        print(f"\n[{tid}] {task.get('description', '')}")
        reports = _run_single_task(task, workloads, config_path, output_dir)
        all_reports.extend(reports)

    # --- Sensitivity analysis ---
    if not skip_sensitivity:
        print("\n" + "=" * 60)
        print("Running sensitivity analysis tasks...")
        print("=" * 60)
        for task in workloads.get("sensitivity_analysis_tasks", []):
            tid = task["id"]
            if task_filter and tid not in task_filter:
                continue
            print(f"\n[{tid}] {task.get('description', '')}")
            reports = _run_sensitivity_task(task, workloads, config_path, output_dir)
            all_reports.extend(reports)

    # --- Save combined results ---
    combined_path = os.path.join(output_dir, "all_results.json")
    combined = [r.to_dict() for r in all_reports]
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nAll results saved to {combined_path}")

    # --- Summary ---
    _print_summary_table(all_reports)

    return all_reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLMCompass bottleneck profiling")
    parser.add_argument(
        "--config",
        default="configs/GA100.json",
        help="Path to GPU architecture JSON",
    )
    parser.add_argument(
        "--workloads",
        default="configs/workloads.json",
        help="Path to workloads.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/profiling",
        help="Directory to save results",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated list of task IDs to run (default: all)",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis tasks",
    )
    args = parser.parse_args()

    task_filter = None
    if args.tasks:
        task_filter = [t.strip() for t in args.tasks.split(",")]

    main(
        config_path=args.config,
        workloads_path=args.workloads,
        output_dir=args.output_dir,
        task_filter=task_filter,
        skip_sensitivity=args.skip_sensitivity,
    )
