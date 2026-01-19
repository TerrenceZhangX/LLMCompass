from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, log
import time
import statistics
import numpy as np
import torch


@torch.compile
def gelu_gpu(input: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(input, approximate="tanh")


# x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
class GeLU(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.shape = None

    def __call__(self, input: Tensor) -> Tensor:
        assert self.data_type == input.data_type
        self.shape = input.shape
        self.M = size(input.shape[:])
        self.computational_graph = self.ComputationalGraph(self.M, self.data_type)
        return input

    def roofline_model(self, pcb_module: Device):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        M = self.M
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / min(pcb_module.io_module.bandwidth
            , pcb_module.compute_module.l2_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq)
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def print_latency(self):
        print(f"{self.shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, data_type: DataType):            
            self.M = M
            self.data_type = data_type

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        self.computational_graph.data_type = (
            pcb_module.compute_module.core.vector_unit.data_type
        )
        parallelism = (
            pcb_module.compute_module.core_count
            * pcb_module.compute_module.core.vector_unit.vector_width
            * pcb_module.compute_module.core.vector_unit.vector_count
        )
        M = ceil(self.computational_graph.M / parallelism) * parallelism
        data_type = self.computational_graph.data_type
        total_io_count = M * 2 * data_type.word_size
        io_latency = (
            total_io_count / pcb_module.io_module.bandwidth
            + total_io_count
            / pcb_module.compute_module.l2_bandwidth_per_cycle
            / pcb_module.compute_module.clock_freq
        )
        total_flop_count = M * (
            10 + pcb_module.compute_module.core.vector_unit.flops_per_exp
        )
        compute_latency = (
            total_flop_count
            / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            / pcb_module.compute_module.core_count
            / pcb_module.compute_module.clock_freq
        )

        return max(compute_latency, io_latency)

    def run_on_gpu(self):
        assert self.shape is not None
        input = torch.randn(self.shape, dtype=torch.float16, device="cuda")
        latencies = []

        # warmup
        for _ in range(3):
            _ = gelu_gpu(input)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = gelu_gpu(input)
            torch.cuda.synchronize()
            end = time.time()
            assert output.shape == input.shape
            latencies.append(end - start)
        # print(latencies)
        self.latency_on_gpu = statistics.median(latencies)
        return self.latency_on_gpu

    def profile(self, pcb_module: Device):
        word_size = getattr(getattr(self, "data_type", None), "word_size", None)
        if not isinstance(word_size, int) or word_size <= 0:
            word_size = 2
        m = getattr(self, "M", None)
        try:
            m = int(m)
        except (TypeError, ValueError):
            return super().profile(pcb_module)
        if m <= 0:
            return super().profile(pcb_module)

        # Strict streaming model: elementwise kernel reads input once and writes output once.
        base = float(m * word_size)
        dram_bytes = 2.0 * base
        l2_bytes = 2.0 * base
        l1_bytes = 2.0 * base
        smem_bytes = 0.0
        reg_bytes = 0.0

        # Parallelism: derive how many cores are needed to cover the vector lanes.
        core_count = getattr(getattr(pcb_module, "compute_module", None), "core_count", None)
        vec = getattr(getattr(getattr(pcb_module, "compute_module", None), "core", None), "vector_unit", None)
        vector_width = getattr(vec, "vector_width", None)
        vector_count = getattr(vec, "vector_count", None)

        occ = 0.0
        act_cta = 0.0
        grid_size = 0.0

        try:
            core_count = int(core_count)
        except (TypeError, ValueError):
            core_count = None
        if isinstance(core_count, int) and core_count <= 0:
            core_count = None

        try:
            vector_width = int(vector_width)
        except (TypeError, ValueError):
            vector_width = None
        if isinstance(vector_width, int) and vector_width <= 0:
            vector_width = None

        try:
            vector_count = int(vector_count)
        except (TypeError, ValueError):
            vector_count = None
        if isinstance(vector_count, int) and vector_count <= 0:
            vector_count = None

        if isinstance(core_count, int) and core_count > 0 and isinstance(vector_width, int) and isinstance(vector_count, int) and vector_width > 0 and vector_count > 0:
            lanes_per_core = vector_width * vector_count
            cores_needed = (m + lanes_per_core - 1) // lanes_per_core
            active = min(int(cores_needed), int(core_count))
            occ = float(active) / float(core_count)
            act_cta = float(active)
            grid_size = float(int(cores_needed))

        out = super().profile(pcb_module)
        out["traffic_bytes"]["dram"] = dram_bytes
        out["traffic_bytes"]["l2"] = l2_bytes
        out["traffic_bytes"]["l1"] = l1_bytes
        out["traffic_bytes"]["smem"] = smem_bytes
        out["traffic_bytes"]["reg"] = reg_bytes

        l2_access = float(l2_bytes)
        l2_miss = min(float(dram_bytes), l2_access)
        l2_hit = l2_access - l2_miss
        out["cache"]["l2_accesses"] = l2_access
        out["cache"]["l2_hits"] = l2_hit
        out["cache"]["l2_hit_rate"] = (l2_hit / l2_access) if l2_access > 0 else 0.0

        l1_access = float(l1_bytes)
        l1_miss = min(float(l2_bytes), l1_access)
        l1_hit = l1_access - l1_miss
        out["cache"]["l1_accesses"] = l1_access
        out["cache"]["l1_hits"] = l1_hit
        out["cache"]["l1_hit_rate"] = (l1_hit / l1_access) if l1_access > 0 else 0.0

        out["parallelism"]["occupancy"] = occ
        out["parallelism"]["active_ctas"] = act_cta
        out["parallelism"]["grid_size"] = grid_size
        return out

    @staticmethod
    def gpu_kernel_launch_overhead():
        tensor_size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(tensor_size, tensor_size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            _ = gelu_gpu(a)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        print(latencies)
        return avg_overhead
