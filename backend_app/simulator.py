import asyncio
import time
import re
from typing import Any, Dict

# import real software models and hardware/system descriptions
from software_model.utils import Tensor, data_type_dict
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.operators import Operator
from hardware_model.system import system_dict


def _map_dtype(dtype_str: str):
    s = dtype_str.lower()
    if "fp16" in s or "float16" in s:
        return data_type_dict.get("fp16")
    else:
        return None

def _make_tensor(shape, dtype_obj):
    try:
        return Tensor(list(shape), dtype_obj)
    except Exception:
        return None


def _simulate_matmul_sync(kernel_name: str, input_dim: list[list], dtype_str: list[str], system_key: str = None) -> Dict[str, Any]:
    # normalize dtype input
    dt = _map_dtype(dtype_str[0])
    A_shape = input_dim[0]
    B_shape = input_dim[1]
    
    if dt is None or A_shape is None or B_shape is None:
        return {
            "status": "failed",
            "output": None,
            "time_taken": None,
            "metadata": {
                "kernel_name": kernel_name,
                "error": "invalid dtype or input_dim - cannot simulate matmul",
                "error_code": "INVALID_INPUT",
                "hint": "Check dtype and input_dim in the request",
            },
        }

    op = Matmul(dt)
    A = _make_tensor(A_shape, dt)
    B = _make_tensor(B_shape, dt)
    _ = op(A, B)

    # require valid system_key and resolve device safely
    if not system_key:
        return {
            "status": "failed",
            "output": None,
            "time_taken": None,
            "metadata": {
                "kernel_name": kernel_name,
                "error": "no system_key provided - cannot simulate matmul",
                "error_code": "NO_SYSTEM",
                "hint": "Provide a valid system_key in the request",
            },
        }
    system = system_dict.get(system_key)
    if system is None:
        return {
            "status": "failed",
            "output": None,
            "time_taken": None,
            "metadata": {
                "kernel_name": kernel_name,
                "error": f"missing system configuration '{system_key}' - cannot simulate matmul",
                "error_code": "NO_SYSTEM",
                "hint": "Use a key from hardware_model.system.system_dict",
            },
        }

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "matmul simulated"},
        "time_taken": float(latency),
        "metadata": {"kernel_name": kernel_name, "input_dim": input_dim, "dtype": dtype_str},
    }

def _simulate_fail(kernel_name: str, _input_dim=None, _dtype_str: str = "") -> Dict[str, Any]:
    # generic fallback is not allowed per policy; return explicit failure
    return {
        "status": "failed",
        "output": None,
        "time_taken": None,
        "metadata": {
            "kernel_name": kernel_name,
            "error": "unsupported op - no generic simulator available",
            "error_code": "UNSUPPORTED_OP",
            "hint": "Implement a simulator for this kernel in software_model or submit with a supported op"
        },
    }


def _select_sync_simulator(kernel_name: str):
    if not kernel_name:
        return _simulate_fail
    kn = kernel_name.lower()
    if "matmul" in kn:
        return _simulate_matmul_sync
    # conv and other ops are not supported unless explicitly implemented
    return _simulate_fail


async def simulate_kernel_trace(kernel_name: str, op: str, input_dim: list[list], dtype: list[str], system_key: str) -> Dict[str, Any]:
    """
    Dispatch to real software_model simulations. Blocking compile_and_simulate calls are executed
    in a thread via asyncio.to_thread so the event loop is not blocked.
    Returns standardized result: {status, output, time_taken, metadata}
    """
    # prefer op if provided (more specific), else fall back to kernel_name
    selector = op if op else kernel_name
    simulator = _select_sync_simulator(selector)
    # run simulator in thread; pass system_key when simulator expects it
    result = await asyncio.to_thread(simulator, kernel_name, input_dim, dtype, system_key)

    # simulator returns dict; validate and propagate structured failure
    if not isinstance(result, dict):
        return {"status": "failed", "output": None, "time_taken": None, "metadata": {"kernel_name": kernel_name, "error": "simulator returned invalid result type", "error_code": "SIMULATOR_ERROR", "hint": "Check simulator implementation"}}
    # ensure failure results have error_code/hint when possible
    if result.get("status") == "failed":
        md = result.setdefault("metadata", {})
        md.setdefault("error_code", md.get("error_code", "SIMULATOR_FAILED"))
        md.setdefault("hint", md.get("hint", "See simulator logs for details"))
    return result


def get_supported_ops() -> list:
    """Return list of supported op keywords for routing/diagnostics."""
    return ["gelu", "layernorm", "matmul", "softmax"]


async def process_kernel_simulation_task(kernel_task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public entrypoint used by the worker. Calls simulate_kernel_trace and normalizes output.
    """
    kernel_name = kernel_task.get("kernel_name", "")
    op = kernel_task.get("op", "")
    input_dim = kernel_task.get("input_dim", [])
    dtype = kernel_task.get("dtype", [])
    system_key = kernel_task.get("system_key")
    start = time.time()
    res = await simulate_kernel_trace(kernel_name, op, input_dim, dtype, system_key=system_key)
    end = time.time()
    # normalize to expected schema
    out = {
        "kernel_name": kernel_name,
        "status": res.get("status", "failed"),
        "output": res.get("output"),
        "time_taken": res.get("time_taken", end - start),
        "metadata": res.get("metadata", {"op": op, "input_dim": input_dim, "dtype": dtype}),
    }
    return out
