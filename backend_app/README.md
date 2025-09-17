# LLMCompass backend_app

This directory contains a minimal backend HTTP service (FastAPI + simulation scheduler).
This README explains how to build and run the backend (Docker-only), how to call the API,
and how to extend the codebase with new synchronous simulators.

## Prerequisites
- Docker (required runtime)
- Python 3.8+ (for development/testing inside the image)

## Docker (build & run)

The backend is supported to run only via Docker. Build the image from the repository root:

```bash
sudo docker build -t llmcompass-backend .
```

Run the docker container, which exposes 8000 to host for backend interaction:

```bash
sudo docker run --rm -p 8000:8000 llmcompass-backend
```

## Environment variables
- `API_PORT` — port used by tests/uvicorn inside the container (default: 8000)
- `API_URL` — if set in tests, the test suite will target this external URL instead of starting a local server
- `ARTIFACT_DIR` — directory where tests write artifacts (default: `artifacts/`)

## HTTP API endpoints
- `GET /health` — health check, returns `{status: "ok"}`
- `GET /supported_ops` — list of supported operations (e.g. `matmul`, `gelu`)
- `POST /tasks` — submit a simulation task (returns `task_id`)
- `GET /tasks/{task_id}` — query task status and result

Example task payload (matmul):

```json
{
  "kernel_name": "itest_matmul",
  "op": "matmul",
  "input_dim": [[1, 2048], [2048, 7168]],
  "dtype": ["c10::BFloat16", "c10::BFloat16"],
  "system_key": "A100_4_fp16"
}
```

Example finished task (matmul):
```json
{
  "status_code": 200,
  "body": {
    "task_id": "089d0b13-2ef9-43e1-bde3-44ed7219e959",
    "status": "done",
    "result": {
      "kernel_name": "itest_matmul_M_1",
      "status": "success",
      "output": {
        "summary": "matmul simulated"
      },
      "time_taken": 1.4408317802844531e-05,
      "metadata": {
        "kernel_name": "itest_matmul_M_1",
        "input_dim": [
          [
            1,
            2048
          ],
          [
            2048,
            7168
          ]
        ],
        "dtype": [
          "c10::BFloat16",
          "c10::BFloat16"
        ]
      }
    },
    "payload": {
      "kernel_name": "itest_matmul_M_1",
      "op": "matmul",
      "input_dim": [
        [
          1,
          2048
        ],
        [
          2048,
          7168
        ]
      ],
      "dtype": [
        "c10::BFloat16",
        "c10::BFloat16"
      ],
      "system_key": "A100_4_fp16"
    },
    "created_at": "2025-09-17T02:23:11.777675",
    "updated_at": "2025-09-17T02:23:11.778457"
  }
}
```


Task states may include `queued`, `running`, `done`, `failed` (scheduler/worker dependent).

## Code layout and runtime flow

Key modules:
- `backend_app/simulator.py` — async entry points and dispatcher (`simulate_kernel_trace`, `process_kernel_simulation_task`).
- `backend_app/sim_utils.py` — shared helpers: dtype mapping, tensor construction, unified failure response helper `_make_failure`.
- `backend_app/sync_simulators.py` — synchronous `_simulate_*` implementations (e.g. `_simulate_matmul_sync`) and `_select_sync_simulator`.

Runtime flow (simplified):
1. Worker receives a task, constructs `kernel_task` dict and calls `process_kernel_simulation_task`.
2. `process_kernel_simulation_task` calls `simulate_kernel_trace` (async).
3. `simulate_kernel_trace` selects a synchronous implementation via `_select_sync_simulator` and runs it in a thread using `asyncio.to_thread` to avoid blocking the event loop.
4. Synchronous implementations perform compile/simulate and return a standardized dict: `{status, output, time_taken, metadata}`.

All failure responses are created via `_make_failure(kernel_name, error, error_code)` to keep format consistent.

## Adding a new synchronous simulator

1. Implement a new `_simulate_<op>_sync` function in `backend_app/sync_simulators.py` with the same signature as existing ones:

```py
def _simulate_conv_sync(kernel_name, input_dim, dtype_str, system_key=None):
    # use backend_app.sim_utils helpers: _map_dtype, _make_tensor, _make_failure
    ...
    return {"status": "success", ...}
```

2. Update `_select_sync_simulator` in the same file to return your function when appropriate (e.g. `if "conv" in kn:`).

3. Optionally add the op keyword to `get_supported_ops()` in `backend_app/simulator.py`.

4. Add unit and/or integration tests to cover happy-path and failure cases.

5. If new dependencies are required, update `requirements.txt` / `pyproject.toml` and Dockerfile.

Important: keep the synchronous implementation's return schema consistent so the async wrapper can handle it uniformly.

## Error codes

Error codes are currently plain strings (e.g. `INVALID_INPUT`, `NO_SYSTEM`, `SIMULATOR_ERROR`).

## Tests

Run tests from the repository root inside the Docker container or a development image:

```bash
pytest tests/
```

Integration tests write artifacts to the directory specified by `ARTIFACT_DIR` to aid debugging.