import os
import time
import subprocess
import sys
import signal
import json
from pathlib import Path

import pytest
import requests


# Server config: if API_URL is set we will target that and not start a server.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("API_PORT", "8000"))
BASE = os.environ.get("API_URL", f"http://{SERVER_HOST}:{SERVER_PORT}")

# artifacts directory for intermediate results
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))


def _ensure_artifacts_dir():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def server():
    """Start a uvicorn server for the test session when API_URL is not set.

    If API_URL is provided (pointing to an external service), the fixture does nothing.
    """
    _ensure_artifacts_dir()
    if os.environ.get("API_URL"):
        # External server provided, do not start local uvicorn.
        yield
        return

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend_app.main:app",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]

    # redirect uvicorn output to artifact files
    out_path = ARTIFACT_DIR / "uvicorn.out"
    err_path = ARTIFACT_DIR / "uvicorn.err"
    fout = open(out_path, "wb")
    ferr = open(err_path, "wb")
    proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr, preexec_fn=None)

    # wait for health endpoint
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        # failed to start in time; capture stderr for debugging
        try:
            ferr.flush()
            with open(err_path, "rb") as f:
                err = f.read()
        except Exception:
            proc.kill()
            err = b"(no stderr available)"
        fout.close()
        ferr.close()
        raise RuntimeError(
            f"uvicorn failed to start in time. stderr:\n{err.decode(errors='ignore')}"
        )

    try:
        yield
    finally:
        # terminate the server process
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
        fout.close()
        ferr.close()


def _url(path: str) -> str:
    return BASE.rstrip("/") + path


def test_health():
    r = requests.get(_url("/health"), timeout=5)
    with open(ARTIFACT_DIR / "health.json", "w") as f:
        json.dump({"status_code": r.status_code, "body": r.json()}, f, indent=2)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_supported_ops():
    r = requests.get(_url("/supported_ops"), timeout=5)
    with open(ARTIFACT_DIR / "supported_ops.json", "w") as f:
        json.dump({"status_code": r.status_code, "body": r.json()}, f, indent=2)
    assert r.status_code == 200
    j = r.json()
    assert isinstance(j.get("supported_ops"), list)


@pytest.mark.parametrize(
    "matmul_payload",
    [
        {
            "kernel_name": "itest_matmul_M_1",
            "op": "matmul",
            "input_dim": [[1, 2048], [2048, 7168]],
            "dtype": ["c10::BFloat16", "c10::BFloat16"],
            "system_key": "A100_4_fp16",
        },
        {
            "kernel_name": "itest_matmul_M_128",
            "op": "matmul",
            "input_dim": [[128, 128], [128, 128]],
            "dtype": ["c10::BFloat16", "c10::BFloat16"],
            "system_key": "A100_4_fp16",
        },
        {
            "kernel_name": "itest_matmul_fp8_unsupported",
            "op": "matmul",
            "input_dim": [[128, 128], [128, 128]],
            "dtype": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
            "system_key": "A100_4_fp16",
        },
    ],
)
def test_create_task_and_poll_matmul(matmul_payload):
    """Submit a matmul task and poll for completion; save artifacts for debugging."""
    payload = matmul_payload

    with open(ARTIFACT_DIR / "matmul_create_task_request.json", "w") as f:
        json.dump(
            {"url": _url("/tasks"), "payload": payload, "op": "matmul"}, f, indent=2
        )

    r = requests.post(_url("/tasks"), json=payload, timeout=5)
    with open(ARTIFACT_DIR / "matmul_create_task_response.json", "w") as f:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        json.dump({"status_code": r.status_code, "body": body}, f, indent=2)

    assert r.status_code == 200
    j = r.json()
    task_id = j.get("task_id")
    assert task_id

    # poll briefly for terminal status
    deadline = time.time() + 20
    last = None
    while time.time() < deadline:
        r = requests.get(_url(f"/tasks/{task_id}"), timeout=5)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status")
            with open(ARTIFACT_DIR / f"task_{task_id}_poll_matmul.json", "w") as f:
                json.dump({"status_code": r.status_code, "body": info}, f, indent=2)
            if status == "done":
                assert "result" in info
                simulated_time = info.get("result", {}).get("time_taken")
                if payload.get("kernel_name") == "itest_matmul_fp8_unsupported":
                    # this kernel is expected to be unsupported
                    assert (
                        info.get("result", {}).get("metadata").get("error_code")
                    ), "INVALID_INPUT"
                elif payload.get("kernel_name") == "itest_matmul_M_1":
                    # this kernel is expected to be very fast
                    assert (
                        simulated_time == 1.4408317802844531e-05
                    ), f"simulated_time={simulated_time}"
                elif payload.get("kernel_name") == "itest_matmul_M_128":
                    # this kernel is expected to be fast
                    assert (
                        simulated_time == 1.1276595744680851e-07
                    ), f"simulated_time={simulated_time}"
                break
            last = status
        time.sleep(1)

    assert last in ("done", "failed", "queued", None)


@pytest.mark.parametrize(
    "bmm_payload",
    [
        {
            "kernel_name": "itest_bmm_M_1",
            "op": "bmm",
            # batch=1, m=1, k=2048 ; batch=1, k=2048, n=7168
            "input_dim": [[1, 1, 2048], [1, 2048, 7168]],
            "dtype": ["c10::BFloat16", "c10::BFloat16"],
            "system_key": "A100_4_fp16",
        },
        {
            "kernel_name": "itest_bmm_fp8_unsupported",
            "op": "bmm",
            "input_dim": [[128, 128, 128], [128, 128, 128]],
            "dtype": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn"],
            "system_key": "A100_4_fp16",
        },
    ],
)
def test_create_task_and_poll_bmm(bmm_payload):
    """Submit a batched-matmul (bmm) task and poll for completion; save artifacts for debugging."""
    payload = bmm_payload

    with open(ARTIFACT_DIR / "bmm_create_task_request.json", "w") as f:
        json.dump({"url": _url("/tasks"), "payload": payload, "op": "bmm"}, f, indent=2)

    r = requests.post(_url("/tasks"), json=payload, timeout=5)
    with open(ARTIFACT_DIR / "bmm_create_task_response.json", "w") as f:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        json.dump({"status_code": r.status_code, "body": body}, f, indent=2)

    assert r.status_code == 200
    j = r.json()
    task_id = j.get("task_id")
    assert task_id

    # poll briefly for terminal status
    deadline = time.time() + 20
    last = None
    while time.time() < deadline:
        r = requests.get(_url(f"/tasks/{task_id}"), timeout=5)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status")
            with open(ARTIFACT_DIR / f"task_{task_id}_poll_bmm.json", "w") as f:
                json.dump({"status_code": r.status_code, "body": info}, f, indent=2)
            if status == "done":
                assert "result" in info
                simulated_time = info.get("result", {}).get("time_taken")
                if payload.get("kernel_name") == "itest_bmm_fp8_unsupported":
                    # this kernel is expected to be unsupported
                    assert (
                        info.get("result", {}).get("metadata", {}).get("error_code")
                        == "INVALID_INPUT"
                    )
                else:
                    # for supported kernels we expect a numeric simulation time
                    assert simulated_time is not None
                    assert isinstance(simulated_time, (int, float))
                break
            last = status
        time.sleep(1)

    assert last in ("done", "failed", "queued", None)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kernel_name": "itest_gelu_default",
            "op": "gelu",
            "input_dim": [1024],
            "dtype": "fp16",
            "system_key": "A100_4_fp16",
        },
    ],
)
def test_create_task_and_poll_gelu(payload):
    """Template test for gelu: submit and poll for completion."""
    with open(ARTIFACT_DIR / "gelu_create_task_request.json", "w") as f:
        json.dump(
            {"url": _url("/tasks"), "payload": payload, "op": "gelu"}, f, indent=2
        )

    r = requests.post(_url("/tasks"), json=payload, timeout=5)
    with open(ARTIFACT_DIR / "gelu_create_task_response.json", "w") as f:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        json.dump({"status_code": r.status_code, "body": body}, f, indent=2)

    assert r.status_code == 200
    j = r.json()
    task_id = j.get("task_id")
    assert task_id

    deadline = time.time() + 20
    last = None
    while time.time() < deadline:
        r = requests.get(_url(f"/tasks/{task_id}"), timeout=5)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status")
            with open(ARTIFACT_DIR / f"task_{task_id}_poll_gelu.json", "w") as f:
                json.dump({"status_code": r.status_code, "body": info}, f, indent=2)
            if status == "done":
                assert "result" in info
                simulated_time = info.get("result", {}).get("time_taken")
                assert simulated_time is not None
                break
            last = status
        time.sleep(1)
    assert last in ("done", "failed", "queued", None)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kernel_name": "itest_layernorm_default",
            "op": "layernorm",
            "input_dim": [1, 1024, 7168],
            "dtype": "fp16",
            "system_key": "A100_4_fp16",
        },
        {
            "kernel_name": "itest_layernorm_unsupported",
            "op": "layernorm",
            "input_dim": [1, 1024, 7168],
            "dtype": "fp8",
            "system_key": "A100_4_fp16",
        },
    ],
)
def test_create_task_and_poll_layernorm(payload):
    """Template test for layernorm: submit and poll for completion."""
    with open(ARTIFACT_DIR / "layernorm_create_task_request.json", "w") as f:
        json.dump(
            {"url": _url("/tasks"), "payload": payload, "op": "layernorm"}, f, indent=2
        )

    r = requests.post(_url("/tasks"), json=payload, timeout=5)
    with open(ARTIFACT_DIR / "layernorm_create_task_response.json", "w") as f:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        json.dump({"status_code": r.status_code, "body": body}, f, indent=2)

    assert r.status_code == 200
    j = r.json()
    task_id = j.get("task_id")
    assert task_id

    deadline = time.time() + 20
    last = None
    while time.time() < deadline:
        r = requests.get(_url(f"/tasks/{task_id}"), timeout=5)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status")
            with open(ARTIFACT_DIR / f"task_{task_id}_poll_layernorm.json", "w") as f:
                json.dump({"status_code": r.status_code, "body": info}, f, indent=2)
            if status == "done":
                assert "result" in info
                simulated_time = info.get("result", {}).get("time_taken")
                if payload.get("kernel_name") == "itest_layernorm_default":
                    assert simulated_time is not None
                elif payload.get("kernel_name") == "itest_layernorm_unsupported":
                    assert simulated_time is None
                break
            last = status
        time.sleep(1)
    assert last in ("done", "failed", "queued", None)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kernel_name": "itest_softmax_default",
            "op": "softmax",
            "input_dim": [64, 128],
            "dtype": "fp16",
            "system_key": "A100_4_fp16",
        },
    ],
)
def test_create_task_and_poll_softmax(payload):
    """Template test for softmax: submit and poll for completion."""
    with open(ARTIFACT_DIR / "softmax_create_task_request.json", "w") as f:
        json.dump(
            {"url": _url("/tasks"), "payload": payload, "op": "softmax"}, f, indent=2
        )

    r = requests.post(_url("/tasks"), json=payload, timeout=5)
    with open(ARTIFACT_DIR / "softmax_create_task_response.json", "w") as f:
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        json.dump({"status_code": r.status_code, "body": body}, f, indent=2)

    assert r.status_code == 200
    j = r.json()
    task_id = j.get("task_id")
    assert task_id

    deadline = time.time() + 20
    last = None
    while time.time() < deadline:
        r = requests.get(_url(f"/tasks/{task_id}"), timeout=5)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status")
            with open(ARTIFACT_DIR / f"task_{task_id}_poll_softmax.json", "w") as f:
                json.dump({"status_code": r.status_code, "body": info}, f, indent=2)
            if status == "done":
                assert "result" in info
                simulated_time = info.get("result", {}).get("time_taken")
                assert simulated_time is not None
                break
            last = status
        time.sleep(1)
    assert last in ("done", "failed", "queued", None)
