from typing import Any, Dict

from backend_app.sim_utils import (
    _map_dtype,
    _make_tensor,
    _make_failure,
    _make_missing_system,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from hardware_model.system import system_dict


def _simulate_matmul_sync(
    kernel_name: str,
    input_dim: list[list],
    dtype_str: list[str],
    system_key: str = None,
) -> Dict[str, Any]:
    dt = _map_dtype(dtype_str[0])
    A_shape = input_dim[0]
    B_shape = input_dim[1]

    if dt is None:
        return _make_failure(
            kernel_name, "invalid or unsupported dtype", "INVALID_INPUT"
        )
    elif A_shape is None or B_shape is None:
        return _make_failure(kernel_name, "invalid input dimension", "INVALID_INPUT")

    op = Matmul(dt)
    A = _make_tensor(A_shape, dt)
    B = _make_tensor(B_shape, dt)
    _ = op(A, B)

    if not system_key:
        return _make_failure(kernel_name, "no valid system_key provided", "NO_SYSTEM")
    system = system_dict.get(system_key)
    if system is None:
        return _make_missing_system(kernel_name, system_key)

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "matmul simulated"},
        "time_taken": float(latency),
        "metadata": {
            "kernel_name": kernel_name,
            "input_dim": input_dim,
            "dtype": dtype_str,
        },
    }


def _simulate_bmm_sync(
    kernel_name: str,
    input_dim: list[list],
    dtype_str: list[str],
    system_key: str = None,
) -> Dict[str, Any]:
    dt = _map_dtype(dtype_str[0])
    A_shape = input_dim[0]
    B_shape = input_dim[1]

    if dt is None:
        return _make_failure(
            kernel_name, "invalid or unsupported dtype", "INVALID_INPUT"
        )
    elif A_shape is None or B_shape is None:
        return _make_failure(kernel_name, "invalid input dimension", "INVALID_INPUT")

    op = BatchedMatmul(dt)
    A = _make_tensor(A_shape, dt)
    B = _make_tensor(B_shape, dt)
    _ = op(A, B)

    if not system_key:
        return _make_failure(kernel_name, "no valid system_key provided", "NO_SYSTEM")
    system = system_dict.get(system_key)
    if system is None:
        return _make_missing_system(kernel_name, system_key)

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "matmul simulated"},
        "time_taken": float(latency),
        "metadata": {
            "kernel_name": kernel_name,
            "input_dim": input_dim,
            "dtype": dtype_str,
        },
    }


def _simulate_layernorm_sync(
    kernel_name: str,
    input_dim: list,
    dtype_str: str,
    system_key: str = None,
) -> Dict[str, Any]:
    dt = _map_dtype(dtype_str)
    A_shape = input_dim

    if dt is None:
        return _make_failure(
            kernel_name, "invalid or unsupported dtype", "INVALID_INPUT"
        )
    elif A_shape is None:
        return _make_failure(kernel_name, "invalid input dimension", "INVALID_INPUT")

    op = LayerNorm(dt)
    A = _make_tensor(A_shape, dt)
    _ = op(A)

    if not system_key:
        return _make_failure(kernel_name, "no valid system_key provided", "NO_SYSTEM")
    system = system_dict.get(system_key)
    if system is None:
        return _make_missing_system(kernel_name, system_key)

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "LayerNorm simulated"},
        "time_taken": float(latency),
        "metadata": {
            "kernel_name": kernel_name,
            "input_dim": input_dim,
            "dtype": dtype_str,
        },
    }


def _simulate_gelu_sync(
    kernel_name: str,
    input_dim: list,
    dtype_str: str,
    system_key: str = None,
) -> Dict[str, Any]:
    dt = _map_dtype(dtype_str)
    A_shape = input_dim

    if dt is None:
        return _make_failure(
            kernel_name, "invalid or unsupported dtype", "INVALID_INPUT"
        )
    elif A_shape is None:
        return _make_failure(kernel_name, "invalid input dimension", "INVALID_INPUT")

    op = GeLu(dt)
    A = _make_tensor(A_shape, dt)
    _ = op(A)

    if not system_key:
        return _make_failure(kernel_name, "no valid system_key provided", "NO_SYSTEM")
    system = system_dict.get(system_key)
    if system is None:
        return _make_missing_system(kernel_name, system_key)

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "GeLU simulated"},
        "time_taken": float(latency),
        "metadata": {
            "kernel_name": kernel_name,
            "input_dim": input_dim,
            "dtype": dtype_str,
        },
    }


def _simulate_softmax_sync(
    kernel_name: str,
    input_dim: list,
    dtype_str: str,
    system_key: str = None,
) -> Dict[str, Any]:
    dt = _map_dtype(dtype_str)
    A_shape = input_dim

    if dt is None:
        return _make_failure(
            kernel_name, "invalid or unsupported dtype", "INVALID_INPUT"
        )
    elif A_shape is None:
        return _make_failure(kernel_name, "invalid input dimension", "INVALID_INPUT")

    op = Softmax(dt)
    A = _make_tensor(A_shape, dt)
    _ = op(A)

    if not system_key:
        return _make_failure(kernel_name, "no valid system_key provided", "NO_SYSTEM")
    system = system_dict.get(system_key)
    if system is None:
        return _make_missing_system(kernel_name, system_key)

    device = system.device
    latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
    return {
        "status": "success",
        "output": {"summary": "Softmax simulated"},
        "time_taken": float(latency),
        "metadata": {
            "kernel_name": kernel_name,
            "input_dim": input_dim,
            "dtype": dtype_str,
        },
    }


def _simulate_fail(
    kernel_name: str, _input_dim=None, _dtype_str: str = "", system_key: str = None,
) -> Dict[str, Any]:
    return _make_failure(
        kernel_name, "unsupported op - no generic simulator available", "UNSUPPORTED_OP"
    )


def _select_sync_simulator(kernel_name: str):
    if not kernel_name:
        return _simulate_fail
    kn = kernel_name.lower()
    if kn == "matmul":
        return _simulate_matmul_sync
    elif kn == "bmm":
        return _simulate_bmm_sync
    elif kn == "layernorm" in kn:
        return _simulate_layernorm_sync
    elif kn == "gelu":
        return _simulate_gelu_sync
    elif kn == "softmax":
        return _simulate_softmax_sync
    # conv and other ops are not supported unless explicitly implemented
    return _simulate_fail
