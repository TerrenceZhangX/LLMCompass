from typing import Any

# import minimal software/hardware helpers used by simulators
from software_model.utils import Tensor, data_type_dict


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


# centralized failure helper to avoid repeated dict literals
def _make_failure(kernel_name: str, error: str, error_code: str):
    return {
        "status": "failed",
        "output": None,
        "time_taken": None,
        "metadata": {
            "kernel_name": kernel_name,
            "error": error,
            "error_code": error_code,
        },
    }


def _make_missing_system(kernel_name: str, system_key: str):
    return _make_failure(
        kernel_name, f"missing system configuration '{system_key}'", "NO_SYSTEM"
    )


def get_supported_ops() -> list:
    """Return list of supported op keywords for routing/diagnostics."""
    return ["gelu", "layernorm", "matmul", "softmax"]
