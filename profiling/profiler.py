"""Main profiling engine for LLMCompass bottleneck analysis.

Runs LLMCompass roofline model on a given GPU config and workload, then
classifies each operator as compute-bound, memory-bound, or balanced.
"""

import sys
import os
import io
from contextlib import redirect_stdout
from typing import Optional

# Ensure the LLMCompass root is on sys.path so internal imports resolve.
_LLMCOMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LLMCOMPASS_ROOT not in sys.path:
    sys.path.insert(0, _LLMCOMPASS_ROOT)

from design_space_exploration.dse import read_architecture_template, template_to_system
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from profiling.bottleneck_report import (
    BottleneckReport,
    LayerBottleneckReport,
    WorkloadBottleneckReport,
)


# ---------------------------------------------------------------------------
# Operator metadata helpers
# ---------------------------------------------------------------------------

# The 12 logical operator entries in roofline_log order.
OPERATOR_NAMES = [
    "qkv_proj",
    "q_mul_k",
    "a_mul_v",
    "h_matmul0",
    "h_matmul1",
    "h_matmul2",
    "softmax",
    "layernorm_0",
    "layernorm_1",
    "gelu",
    "allreduce_mha",
    "allreduce_ffn",
]

# Which operators use the systolic array vs. vector unit.
SYSTOLIC_OPS = {"qkv_proj", "q_mul_k", "a_mul_v", "h_matmul0", "h_matmul1", "h_matmul2"}
VECTOR_OPS = {"softmax", "layernorm_0", "layernorm_1", "gelu"}
COMM_OPS = {"allreduce_mha", "allreduce_ffn"}


def _get_matmul_flop_io(op, word_size: int):
    """Return (flop_count, io_bytes) for a Matmul operator after __call__."""
    # Matmul stores io_count in *elements*; convert to bytes.
    return int(op.flop_count), int(op.io_count * word_size)


def _get_batched_matmul_flop_io(op, word_size: int):
    """Return (flop_count, io_bytes) for a BatchedMatmul operator after __call__."""
    flops = 2 * op.M * op.K * op.N * op.bs
    io_elems = (op.M * op.K + op.K * op.N + op.M * op.N) * op.bs
    return int(flops), int(io_elems * word_size)


def _get_softmax_flop_io(op, device):
    """Return (flop_count, io_bytes) for Softmax after roofline_model."""
    return int(op.flop_count), int(op.io_count)


def _get_layernorm_flop_io(op, device):
    """Return (flop_count, io_bytes) for LayerNorm after roofline_model.

    Note: The transformer only calls roofline_model on layer_norm0 and
    reuses its latency for layer_norm1.  For layer_norm1, flop_count and
    io_count may not be set; we recompute from M and N.
    """
    if hasattr(op, "flop_count") and op.flop_count > 0:
        return int(op.flop_count), int(op.io_count)
    # Fallback: compute from shape attributes set during __call__
    if hasattr(op, "M") and hasattr(op, "N"):
        io_bytes = int(op.M * op.N * op.data_type.word_size * 2)
        flop_count = int(op.M * op.N * 7)
        return flop_count, io_bytes
    return 0, 0


def _get_gelu_flop_io(op, device):
    """Return (flop_count, io_bytes) for GeLU after roofline_model.

    GeLU's roofline_model computes totals inline without storing io_count
    on the object, so we recompute from attributes.
    """
    word_size = device.compute_module.core.vector_unit.data_type.word_size
    io_bytes = int(op.M * 2 * word_size)
    flops_per_exp = device.compute_module.core.vector_unit.flops_per_exp
    flop_count = int(op.M * (10 + flops_per_exp))
    return flop_count, io_bytes


def _get_allreduce_io(op, word_size: int):
    """Return io_bytes for AllReduce (no compute flops)."""
    from utils import size as _size
    if hasattr(op, "input_shape") and op.input_shape is not None:
        return int(_size(op.input_shape) * word_size)
    return 0


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_BALANCED_THRESHOLD = 1.15  # within 15% → balanced


def _classify(compute_time: float, memory_time: float) -> str:
    if compute_time <= 0 and memory_time <= 0:
        return "balanced"
    if memory_time <= 0:
        return "compute_bound"
    if compute_time <= 0:
        return "memory_bound"
    ratio = compute_time / memory_time
    if ratio > _BALANCED_THRESHOLD:
        return "compute_bound"
    elif ratio < 1.0 / _BALANCED_THRESHOLD:
        return "memory_bound"
    return "balanced"


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_workload(
    config_path: str,
    model_config: dict,
    workload_config: dict,
    phase: str,
    device_count: int,
    task_id: str = "",
) -> WorkloadBottleneckReport:
    """Profile a single workload and return a structured bottleneck report.

    Parameters
    ----------
    config_path : str
        Path to GPU architecture JSON (e.g. ``configs/GA100.json``).
    model_config : dict
        Must contain ``d_model``, ``n_heads``, ``n_layers``, ``name``.
    workload_config : dict
        Must contain ``batch_size``, ``seq_len``.
    phase : str
        ``"prefill"`` or ``"decode"``.
    device_count : int
        Number of GPUs for tensor parallelism.
    task_id : str, optional
        Identifier for this profiling run.
    """

    # ---- Load hardware ----
    arch_specs = read_architecture_template(config_path)
    arch_specs["device_count"] = device_count
    if device_count <= 4:
        arch_specs["interconnect"]["topology"] = "FC"
    else:
        arch_specs["interconnect"]["topology"] = "RING"
    system = template_to_system(arch_specs)

    device = system.device
    fp16 = data_type_dict["fp16"]
    word_size = fp16.word_size

    d_model = model_config["d_model"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]
    model_name = model_config.get("name", "unknown")
    batch_size = workload_config["batch_size"]
    seq_len = workload_config["seq_len"]

    # ---- Hardware peaks ----
    peak_systolic = device.compute_module.total_systolic_array_flops
    peak_vector = device.compute_module.total_vector_flops
    hbm_bw = device.io_module.bandwidth
    l2_bw = (
        device.compute_module.l2_bandwidth_per_cycle
        * device.compute_module.clock_freq
    )
    effective_bw = min(hbm_bw, l2_bw)

    # ---- Build and run the transformer block ----
    if phase == "prefill":
        model = TransformerBlockInitComputationTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=fp16,
        )
        input_tensor = Tensor([batch_size, seq_len, d_model], fp16)
        _ = model(input_tensor)
    elif phase == "decode":
        model = TransformerBlockAutoRegressionTP(
            d_model=d_model,
            n_heads=n_heads,
            device_count=device_count,
            data_type=fp16,
        )
        input_tensor = Tensor([batch_size, 1, d_model], fp16)
        _ = model(input_tensor, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    # ---- Run roofline model (suppress noisy prints) ----
    buf = io.StringIO()
    with redirect_stdout(buf):
        total_latency = model.roofline_model(system)

    # ---- Parse per-operator latencies from roofline_log ----
    latencies_raw = [
        float(x.strip()) for x in model.roofline_log.split(",")
    ]
    assert len(latencies_raw) == 12, (
        f"Expected 12 entries in roofline_log, got {len(latencies_raw)}"
    )

    # ---- Collect per-operator flops / IO ----
    # Map logical operator index → (attribute_object, is_matmul_type)
    # QKV (index 0) represents Q_proj * 3; we report aggregated.
    op_attrs = [
        # 0: qkv = 3 * Q_proj
        ("Q_proj", "matmul", 3),
        # 1: Q_mul_K
        ("Q_mul_K", "batched_matmul", 1),
        # 2: A_mul_V
        ("A_mul_V", "batched_matmul", 1),
        # 3: H_matmul0
        ("H_matmul0", "matmul", 1),
        # 4: H_matmul1
        ("H_matmul1", "matmul", 1),
        # 5: H_matmul2
        ("H_matmul2", "matmul", 1),
        # 6: A_softmax
        ("A_softmax", "softmax", 1),
        # 7: layer_norm0
        ("layer_norm0", "layernorm", 1),
        # 8: layer_norm1 — roofline reuses layer_norm0 latency for both norms
        ("layer_norm1", "layernorm", 1),
        # 9: H_gelu
        ("H_gelu", "gelu", 1),
        # 10: allreduce_mha
        ("allreduce_mha", "allreduce", 1),
        # 11: allreduce_ffn
        ("allreduce_ffn", "allreduce", 1),
    ]

    operator_reports = []
    for idx, (attr_name, op_type, multiplier) in enumerate(op_attrs):
        op_obj = getattr(model, attr_name)
        latency = latencies_raw[idx]
        name = OPERATOR_NAMES[idx]

        # Determine flop_count and io_bytes
        if op_type == "matmul":
            flops_single, io_single = _get_matmul_flop_io(op_obj, word_size)
            flop_count = flops_single * multiplier
            io_bytes = io_single * multiplier
        elif op_type == "batched_matmul":
            flop_count, io_bytes = _get_batched_matmul_flop_io(op_obj, word_size)
            flop_count *= multiplier
            io_bytes *= multiplier
        elif op_type == "softmax":
            flop_count, io_bytes = _get_softmax_flop_io(op_obj, device)
        elif op_type == "layernorm":
            flop_count, io_bytes = _get_layernorm_flop_io(op_obj, device)
        elif op_type == "gelu":
            flop_count, io_bytes = _get_gelu_flop_io(op_obj, device)
        elif op_type == "allreduce":
            flop_count = 0
            io_bytes = _get_allreduce_io(op_obj, word_size)
        else:
            flop_count, io_bytes = 0, 0

        # Select the right hardware peak
        if name in SYSTOLIC_OPS:
            peak_compute = peak_systolic
        elif name in VECTOR_OPS:
            peak_compute = peak_vector
        else:
            # Communication ops — no compute peak; use vector as placeholder
            peak_compute = peak_vector

        # Compute theoretical times
        compute_time = flop_count / peak_compute if peak_compute > 0 else 0.0
        memory_time = io_bytes / effective_bw if effective_bw > 0 else 0.0

        # Arithmetic intensity
        arith_intensity = flop_count / io_bytes if io_bytes > 0 else float("inf")

        # Classification
        if name in COMM_OPS:
            # AllReduce is classified separately — it's interconnect-bound
            bottleneck_label = "memory_bound"
            compute_util = 0.0
            bw_util = (
                memory_time / latency if latency > 0 else 0.0
            )
        else:
            bottleneck_label = _classify(compute_time, memory_time)
            compute_util = compute_time / latency if latency > 0 else 0.0
            bw_util = memory_time / latency if latency > 0 else 0.0

        operator_reports.append(
            BottleneckReport(
                operator_name=name,
                latency_seconds=latency,
                flop_count=flop_count,
                io_bytes=io_bytes,
                arithmetic_intensity=arith_intensity,
                peak_compute_flops=peak_compute,
                peak_bandwidth_bytes=hbm_bw,
                compute_utilization=min(compute_util, 1.0),
                bandwidth_utilization=min(bw_util, 1.0),
                bottleneck=bottleneck_label,
                compute_time=compute_time,
                memory_time=memory_time,
            )
        )

    # ---- Build layer report ----
    layer_report = LayerBottleneckReport(
        layer_index=0,
        phase=phase,
        total_latency_seconds=total_latency,
        operator_reports=operator_reports,
    )

    # ---- Build workload report ----
    ridge_systolic = peak_systolic / effective_bw if effective_bw > 0 else 0.0
    ridge_vector = peak_vector / effective_bw if effective_bw > 0 else 0.0

    report = WorkloadBottleneckReport(
        task_id=task_id,
        model_name=model_name,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        phase=phase,
        device_count=device_count,
        config_path=config_path,
        total_latency_seconds=total_latency,
        layer_report=layer_report,
        dominant_bottleneck=layer_report.dominant_bottleneck,
        peak_systolic_flops=peak_systolic,
        peak_vector_flops=peak_vector,
        peak_hbm_bandwidth=hbm_bw,
        ridge_point_systolic=ridge_systolic,
        ridge_point_vector=ridge_vector,
    )

    return report
