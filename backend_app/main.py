import uuid
import asyncio
import json
import datetime
import os
from typing import List, Union, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend_app.scheduler import simulate_kernel_trace, process_kernel_simulation_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    # in-memory tasks store (not persisted): lost on restart
    app.state.tasks = {}
    app.state.tasks_lock = asyncio.Lock()

    # create queue and start background workers
    app.state.queue = asyncio.Queue()
    # number of concurrent background consumers (in-process). Use environment var WORKER_COUNT
    try:
        worker_count = int(os.environ.get("WORKER_COUNT", "32"))
    except Exception:
        worker_count = 1
    app.state.worker_tasks = []
    for i in range(max(1, worker_count)):
        app.state.worker_tasks.append(
            asyncio.create_task(worker_loop(app.state.queue, app.state.tasks, app.state.tasks_lock, worker_id=i))
        )

    try:
        yield
    finally:
        # shutdown: cancel background worker
        workers = getattr(app.state, "worker_tasks", None) or []
        for worker in workers:
            worker.cancel()
        for worker in workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass


app = FastAPI(title="LLMCompass Kernel Simulator", lifespan=lifespan)


class KernelTask(BaseModel):
    kernel_name: str
    op: str
    input_dim: Optional[Any] = None
    # some clients send a single dtype string, others send a list of dtype strings
    dtype: Optional[Union[str, List[str]]] = "fp32"
    # optional system key
    system_key: Optional[str] = None


async def worker_loop(queue: asyncio.Queue, tasks: dict, lock: asyncio.Lock, worker_id: int = 0):
    while True:
        task_id = await queue.get()
        try:
            async with lock:
                entry = tasks.get(task_id)
                if not entry:
                    queue.task_done()
                    continue
                payload = entry["payload"]
                # mark as running and record worker id / start time
                tasks[task_id]["status"] = "running"
                tasks[task_id]["worker"] = worker_id
                tasks[task_id]["started_at"] = datetime.datetime.utcnow().isoformat()
            # process (outside lock)
            result = await process_kernel_simulation_task(payload)
            async with lock:
                if task_id in tasks:
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["result"] = result
                    tasks[task_id][
                        "updated_at"
                    ] = datetime.datetime.utcnow().isoformat()
        except Exception as e:
            async with lock:
                if task_id in tasks:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["result"] = {"error": str(e)}
                    tasks[task_id][
                        "updated_at"
                    ] = datetime.datetime.utcnow().isoformat()
        finally:
            queue.task_done()


@app.post("/tasks")
async def create_task(t: KernelTask, wait: bool = False, timeout: float = 30.0):
    """
    Create a kernel simulation task.
    If `wait` is False (default) the task is queued and returns immediately with status queued.
    If `wait` is True the request will block up to `timeout` seconds and return the final status/result.
    """
    task_id = str(uuid.uuid4())
    payload = t.dict()
    created_at = datetime.datetime.utcnow().isoformat()

    # insert into in-memory store
    async with app.state.tasks_lock:
        app.state.tasks[task_id] = {
            "payload": payload,
            "status": "queued",
            "result": None,
            "created_at": created_at,
            "updated_at": created_at,
        }

    if not wait:
        # enqueue for background processing
        await app.state.queue.put(task_id)
        return {"task_id": task_id, "status": "queued"}

    # synchronous path: process inline with timeout
    try:
        # mark as running for synchronous (wait) path
        async with app.state.tasks_lock:
            if task_id in app.state.tasks:
                app.state.tasks[task_id]["status"] = "running"
                app.state.tasks[task_id]["worker"] = "inline"
                app.state.tasks[task_id]["started_at"] = datetime.datetime.utcnow().isoformat()

        result = await asyncio.wait_for(
            process_kernel_simulation_task(payload), timeout=timeout
        )
    except asyncio.TimeoutError:
        # leave as queued for background worker to pick up later
        return {
            "task_id": task_id,
            "status": "timeout",
            "message": f"processing did not finish within {timeout}s",
        }
    except Exception as e:
        # update in-memory store as failed
        async with app.state.tasks_lock:
            if task_id in app.state.tasks:
                app.state.tasks[task_id]["status"] = "failed"
                app.state.tasks[task_id]["result"] = {"error": str(e)}
                app.state.tasks[task_id][
                    "updated_at"
                ] = datetime.datetime.utcnow().isoformat()
        raise HTTPException(status_code=500, detail=str(e))

    # write result into in-memory store and return
    async with app.state.tasks_lock:
        if task_id in app.state.tasks:
            app.state.tasks[task_id]["status"] = "done"
            app.state.tasks[task_id]["result"] = result
            app.state.tasks[task_id][
                "updated_at"
            ] = datetime.datetime.utcnow().isoformat()

    return {"task_id": task_id, "status": "done", "result": result}


@app.get("/supported_ops")
async def supported_ops():
    from backend_app.sim_utils import get_supported_ops

    return {"supported_ops": get_supported_ops()}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    async with app.state.tasks_lock:
        entry = app.state.tasks.get(task_id)
        if not entry:
            raise HTTPException(status_code=404, detail="task not found")
        status = entry.get("status")
        result = entry.get("result")
        payload = entry.get("payload")
        created_at = entry.get("created_at")
        updated_at = entry.get("updated_at")

    return {
        "task_id": task_id,
        "status": status,
        "result": result,
        "payload": payload,
        "created_at": created_at,
        "updated_at": updated_at,
    }


@app.get("/health")
async def health():
    # provide richer health info: queue size, worker tasks status, and task counts
    queue = getattr(app.state, "queue", None)
    workers = getattr(app.state, "worker_tasks", None) or []
    tasks_store = getattr(app.state, "tasks", None) or {}

    # summarize task states
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for entry in tasks_store.values():
        st = entry.get("status")
        if st in counts:
            counts[st] += 1

    worker_info = []
    for w in workers:
        try:
            worker_info.append({"done": w.done(), "cancelled": w.cancelled()})
        except Exception:
            worker_info.append({"done": None, "cancelled": None})

    return {
        "status": "ok",
        "queue_length": queue.qsize() if queue is not None else None,
        "worker_count": len(workers),
        "workers": worker_info,
        "task_counts": counts,
    }
