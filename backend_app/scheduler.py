import asyncio
import time
from typing import Any, Dict

# top-level async dispatcher imports helpers and sync simulators from smaller modules
from backend_app.sim_utils import _make_failure
from backend_app.sync_simulators import _select_sync_simulator


async def simulate_kernel_trace(
    kernel_name: str, op: str, input_dim: list[list], dtype: list[str], system_key: str
) -> Dict[str, Any]:
    """
    Dispatch to real software_model simulations. Blocking compile_and_simulate calls are executed
    in a thread via asyncio.to_thread so the event loop is not blocked.
    Returns standardized result: {status, output, simulated_time}
    """
    # prefer op if provided (more specific), else fall back to kernel_name
    selector = op if op else kernel_name
    simulator = _select_sync_simulator(selector)
    # run simulator in thread; pass system_key when simulator expects it
    result = await asyncio.to_thread(
        simulator, kernel_name, input_dim, dtype, system_key
    )

    # simulator returns dict; validate and propagate structured failure
    if not isinstance(result, dict):
        return _make_failure(
            kernel_name, "simulator returned invalid result type", "SIMULATOR_ERROR"
        )
    # ensure failure results have error_code when possible
    if result.get("status") == "failed":
        md = result.setdefault("metadata", {})
        md.setdefault("error_code", md.get("error_code", "SIMULATOR_FAILED"))
    return result


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
    res = await simulate_kernel_trace(
        kernel_name, op, input_dim, dtype, system_key=system_key
    )
    end = time.time()
    # normalize to expected schema
    if res.get("status") == "failed":
        out = {
            "kernel_name": kernel_name,
            "status": "failed",
            "output": None,
            "simulated_time": None,
            "failure_reason": res.get("metadata", {}),
        }
    else:
        out = {
            "kernel_name": kernel_name,
            "status": res.get("status"),
            "output": res.get("output"),
            "simulated_time": res.get("simulated_time"),
        }
    return out
