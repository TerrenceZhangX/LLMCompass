from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy


class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        # Profiling support (populated by compile_and_simulate)
        self._chosen_strategy = None  # "bmm" | "fused_k" | None
        self._matmul_best_mapping = None
        self._matmul_shape = None  # (M, N, K)

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
        self.bs = size(self.input1_shape[:-2])
        self.M = self.input1_shape[-2]
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        return output

    def roofline_model(self, pcb_module: Device):
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency = matmul.roofline_model(pcb_module)
        self.roofline_latency = matmul_latency * self.bs
        return self.roofline_latency

    # def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
    #     matmul = Matmul(self.data_type)
    #     _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
    #     matmul_latency = (
    #         matmul.compile_and_simulate(pcb_module, compile_mode)
    #         # - pcb_module.io_module.latency * 2
    #     )
    #     self.latency = matmul_latency * self.bs  # + pcb_module.io_module.latency * 2
    #     return self.latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
        # Strategy 1: bs independent GEMMs (bmm)
        matmul1 = Matmul(self.data_type)
        _ = matmul1(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
        matmul_latency1 = matmul1.compile_and_simulate(pcb_module, compile_mode) * self.bs

        # Strategy 2: fuse batch into K (model-specific heuristic) + reduction overhead
        matmul2 = Matmul(self.data_type)
        _ = matmul2(Tensor([self.M, self.K * self.bs]), Tensor([self.K * self.bs, self.N]))
        matmul_latency2 = (
            matmul2.compile_and_simulate(pcb_module, compile_mode)
            + (self.bs - 1)
            * self.M
            * self.N
            * self.data_type.word_size
            / pcb_module.io_module.bandwidth
        )

        if matmul_latency1 <= matmul_latency2:
            self._chosen_strategy = "bmm"
            self._matmul_best_mapping = copy.deepcopy(getattr(matmul1, "best_mapping", None))
            self._matmul_shape = (int(self.M), int(self.N), int(self.K))
            self.latency = float(matmul_latency1)
            # Propagate cycle breakdown (scale by batch count)
            bd = getattr(matmul1, '_latency_breakdown', None)
            if bd is not None:
                self._latency_breakdown = {}
                for k, v in bd.items():
                    if k.endswith('_sensitivity_pct'):
                        self._latency_breakdown[k] = v  # ratios don't scale
                    else:
                        self._latency_breakdown[k] = v * float(self.bs)
        else:
            self._chosen_strategy = "fused_k"
            self._matmul_best_mapping = copy.deepcopy(getattr(matmul2, "best_mapping", None))
            self._matmul_shape = (int(self.M), int(self.N), int(self.K * self.bs))
            self.latency = float(matmul_latency2)
            # Propagate cycle breakdown + add reduction overhead as DRAM cycles
            bd = getattr(matmul2, '_latency_breakdown', None)
            if bd is not None:
                freq = pcb_module.compute_module.clock_freq
                red_overhead_cycles = float(
                    (self.bs - 1) * self.M * self.N * self.data_type.word_size
                    / pcb_module.io_module.bandwidth * freq
                )
                self._latency_breakdown = {
                    "dram_cycles": bd["dram_cycles"] + red_overhead_cycles,
                    "l2_cycles": bd["l2_cycles"],
                    "compute_cycles": bd["compute_cycles"],
                    "total_cycles": bd["total_cycles"] + red_overhead_cycles,
                }
                # Copy sensitivity signals if present
                for sk in ('l2_sensitivity_pct', 'dram_sensitivity_pct', 'compute_sensitivity_pct'):
                    if sk in bd:
                        self._latency_breakdown[sk] = bd[sk]

        return self.latency

    def run_on_gpu(
        self,
    ):
        input1 = torch.randn(self.bs, self.M, self.K, dtype=torch.float16).cuda()
        input2 = torch.randn(self.bs, self.K, self.N, dtype=torch.float16).cuda()
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.bmm(input1, input2)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.bmm(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        latencies = []
        for _ in range(50):
            a = torch.randn(1, 1, 1, device="cuda")
            b = torch.randn(1, 1, 1, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.bmm(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        # print(latencies)
        return avg_overhead

    def profile(self, pcb_module: Device):
        # Strict profile: reuse the chosen Matmul mapping when available.
        bs = getattr(self, "bs", None)
        m = getattr(self, "M", None)
        n = getattr(self, "N", None)
        k = getattr(self, "K", None)
        if not all(isinstance(x, int) and x > 0 for x in (bs, m, n, k)):
            return super().profile(pcb_module)

        word_size = getattr(getattr(self, "data_type", None), "word_size", None)
        if not isinstance(word_size, int) or word_size <= 0:
            word_size = 2

        mapping = getattr(self, "_matmul_best_mapping", None)
        shape = getattr(self, "_matmul_shape", None)
        strategy = getattr(self, "_chosen_strategy", None)
        if mapping is not None and isinstance(shape, tuple) and len(shape) == 3:
            mm_m, mm_n, mm_k = shape
            mm = Matmul(self.data_type)
            mm.M, mm.N, mm.K = int(mm_m), int(mm_n), int(mm_k)
            mm.best_mapping = mapping
            base = mm.profile(pcb_module)

            if strategy == "bmm":
                scale = float(bs)
                for k0 in ("dram", "l2", "l1", "smem", "reg"):
                    base["traffic_bytes"][k0] = float(base["traffic_bytes"].get(k0, 0.0) or 0.0) * scale
                # Cache counters are in bytes; scale them similarly.
                for k0 in ("l2_hits", "l2_accesses", "l1_hits", "l1_accesses"):
                    base["cache"][k0] = float(base["cache"].get(k0, 0.0) or 0.0) * scale
                # Hit rates remain ratios; keep as-is.
                base["parallelism"]["grid_size"] = float(base["parallelism"].get("grid_size", 0.0) or 0.0) * scale

                # Derived matmul counters (grid_size_mn scales with batch; k_partitions_avg stays the same).
                d = base.get("derived") if isinstance(base.get("derived"), dict) else None
                dm = d.get("matmul") if isinstance(d, dict) and isinstance(d.get("matmul"), dict) else None
                if isinstance(dm, dict):
                    if isinstance(dm.get("grid_size_mn"), (int, float)):
                        dm["grid_size_mn"] = float(dm["grid_size_mn"]) * scale
                    d["matmul"]["l1_loop_order"] = str(getattr(mapping, "l1_loop_order", ""))
                    d["matmul"]["l2_loop_order"] = str(getattr(mapping, "l2_loop_order", ""))
                # Propagate cycle breakdown from BatchedMatmul
                bd = getattr(self, '_latency_breakdown', None)
                if bd is not None:
                    base["latency_breakdown"] = bd
                return base

            if strategy == "fused_k":
                # Model-specific reduction overhead is expressed using IO bandwidth in compile.
                # Treat it as additional streaming DRAM/L2/L1 traffic on the output matrix.
                extra = float((bs - 1) * m * n * word_size)
                for k0 in ("dram", "l2", "l1"):
                    base["traffic_bytes"][k0] = float(base["traffic_bytes"].get(k0, 0.0) or 0.0) + extra
                base["cache"]["l2_accesses"] = float(base["cache"].get("l2_accesses", 0.0) or 0.0) + extra
                base["cache"]["l1_accesses"] = float(base["cache"].get("l1_accesses", 0.0) or 0.0) + extra
                # Recompute hit rates (still conservative / best-effort).
                l2_a = float(base["cache"].get("l2_accesses", 0.0) or 0.0)
                l2_h = float(base["cache"].get("l2_hits", 0.0) or 0.0)
                base["cache"]["l2_hit_rate"] = (l2_h / l2_a) if l2_a > 0 else 0.0
                l1_a = float(base["cache"].get("l1_accesses", 0.0) or 0.0)
                l1_h = float(base["cache"].get("l1_hits", 0.0) or 0.0)
                base["cache"]["l1_hit_rate"] = (l1_h / l1_a) if l1_a > 0 else 0.0
                # Propagate cycle breakdown from BatchedMatmul
                bd = getattr(self, '_latency_breakdown', None)
                if bd is not None:
                    base["latency_breakdown"] = bd
                return base

            # Unknown strategy -> fall back to base matmul-derived profile.
            return base

        # Fallback: algorithmic global-memory bytes (still strict/non-null).
        # NOTE: This path has no tile model, so L1 metrics are estimated (not precise).
        elem_io = bs * (m * k + k * n + m * n)
        bytes_total = float(elem_io) * float(word_size)
        out = super().profile(pcb_module)
        out["traffic_bytes"]["dram"] = bytes_total
        out["traffic_bytes"]["l2"] = bytes_total
        out["traffic_bytes"]["l1"] = bytes_total
        out["cache"]["l2_accesses"] = bytes_total
        out["cache"]["l2_hits"] = 0.0
        out["cache"]["l2_hit_rate"] = 0.0
        # L1 fallback estimate: without tile model, use algorithmic access pattern.
        # This is an approximation; the main Matmul.profile() uses precise tile traversal.
        total_macs = float(bs * m * n * k)
        l1_access_bytes = 2.0 * total_macs * float(word_size)
        l1_hits_bytes = max(0.0, l1_access_bytes - bytes_total)
        l1_hit_rate = (l1_hits_bytes / l1_access_bytes) if l1_access_bytes > 0 else 0.0
        out["cache"]["l1_accesses"] = l1_access_bytes
        out["cache"]["l1_hits"] = l1_hits_bytes
        out["cache"]["l1_hit_rate"] = l1_hit_rate

        # Estimate parallelism for streaming batched matmul.
        core_count = getattr(getattr(pcb_module, "compute_module", None), "core_count", 0)
        if not isinstance(core_count, int) or core_count <= 0:
            core_count = 1

        # Grid size = output elements / threads_per_block.
        output_elems = bs * m * n
        threads_per_block = 256
        grid_size = float(max(1, (output_elems + threads_per_block - 1) // threads_per_block))
        active_ctas = float(min(grid_size, core_count))
        occupancy = min(1.0, active_ctas / core_count) if core_count > 0 else 0.0

        out["parallelism"] = {
            "occupancy": occupancy,
            "active_ctas": active_ctas,
            "grid_size": grid_size,
        }
        return out


class Matmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = max(
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops,
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
        )
        return self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
            )
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )
            print(
                f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
            )

    def profile(self, pcb_module: Device):
        """Expose best-effort hierarchy traffic + cache/parallelism proxies.

        - traffic_bytes.dram: DRAM<->L2 bytes, following the same reuse logic as simulate() for L2 tiles.
        - traffic_bytes.l2: L2<->core bytes (proxy) derived from inner L1-tile batching model.
        - traffic_bytes.smem/l1: on-core SRAM traffic derived from L1/L0 tiling loops.
        - traffic_bytes.reg: register traffic proxy derived from L0 K-partition reductions.
        - cache.l2_hit_rate: byte-based proxy derived from l2_accesses vs dram_reads.
        - parallelism.occupancy/active_ctas: proxy derived from inner batching vs core_count.
        """

        if self.best_mapping is None:
            # Streaming/GEMV mode: no tiled mapping, but still provide parallelism estimate.
            base_profile = super().profile(pcb_module)

            # Estimate parallelism for streaming matmul (M=1 or N=1).
            m = getattr(self, "M", None)
            n = getattr(self, "N", None)
            k = getattr(self, "K", None)
            if all(isinstance(x, int) and x > 0 for x in (m, n, k)):
                core_count = getattr(getattr(pcb_module, "compute_module", None), "core_count", 0)
                if not isinstance(core_count, int) or core_count <= 0:
                    core_count = 1

                # Streaming GEMV: typically one output element per thread block.
                # Grid size = output elements = M * N (typically one of them is 1).
                output_elems = m * n

                # Use a reasonable block size (e.g., 256 threads).
                threads_per_block = 256
                grid_size = float(max(1, (output_elems + threads_per_block - 1) // threads_per_block))

                # Active CTAs = min(grid_size, core_count).
                active_ctas = float(min(grid_size, core_count))

                # Occupancy = active_ctas / core_count, capped at 1.0.
                occupancy = min(1.0, active_ctas / core_count) if core_count > 0 else 0.0

                base_profile["parallelism"] = {
                    "occupancy": occupancy,
                    "active_ctas": active_ctas,
                    "grid_size": grid_size,
                }

            # Cycle breakdown from simulation (GEMV path)
            breakdown = getattr(self, '_latency_breakdown', None)
            if breakdown is not None:
                base_profile["latency_breakdown"] = breakdown

            return base_profile

        mapping = self.best_mapping
        m = getattr(self, "M", None)
        n = getattr(self, "N", None)
        k = getattr(self, "K", None)
        if not all(isinstance(x, int) and x > 0 for x in (m, n, k)):
            return super().profile(pcb_module)

        word_size = getattr(getattr(self, "data_type", None), "word_size", None)
        if not isinstance(word_size, int) or word_size <= 0:
            word_size = 2

        l2_tile_m = int(getattr(mapping, "l2_tile_M", 0) or 0)
        l2_tile_n = int(getattr(mapping, "l2_tile_N", 0) or 0)
        l2_tile_k = int(getattr(mapping, "l2_tile_K", 0) or 0)
        if min(l2_tile_m, l2_tile_n, l2_tile_k) <= 0:
            return super().profile(pcb_module)

        def ceil_div(a: int, b: int) -> int:
            return (a + b - 1) // b

        def dim_size(idx: int, tile: int, total: int) -> int:
            start = idx * tile
            if start >= total:
                return 0
            return min(tile, total - start)

        outer_m = ceil_div(m, l2_tile_m)
        outer_n = ceil_div(n, l2_tile_n)
        outer_k = ceil_div(k, l2_tile_k)

        def mk_elems(mi: int, ki: int) -> int:
            return dim_size(mi, l2_tile_m, m) * dim_size(ki, l2_tile_k, k)

        def kn_elems(ki: int, ni: int) -> int:
            return dim_size(ki, l2_tile_k, k) * dim_size(ni, l2_tile_n, n)

        def mn_elems(mi: int, ni: int) -> int:
            return dim_size(mi, l2_tile_m, m) * dim_size(ni, l2_tile_n, n)

        dram_read_bytes = 0.0
        dram_write_bytes = 0.0

        first = True
        prev_m = prev_n = prev_k = 0
        for mi, ni, ki in Matmul.generate_tile_loops(outer_m, outer_n, outer_k, mapping.l2_loop_order):
            if first:
                dram_read_bytes += float(mk_elems(mi, ki) + kn_elems(ki, ni)) * float(word_size)
                prev_m, prev_n, prev_k = mi, ni, ki
                first = False
                continue

            # current tile read
            if mi == prev_m and ki == prev_k:
                dram_read_bytes += float(kn_elems(ki, ni)) * float(word_size)
            elif ni == prev_n and ki == prev_k:
                dram_read_bytes += float(mk_elems(mi, ki)) * float(word_size)
            else:
                dram_read_bytes += float(mk_elems(mi, ki) + kn_elems(ki, ni)) * float(word_size)
            if ki > 0 and not (mi == prev_m and ni == prev_n):
                dram_read_bytes += float(mn_elems(mi, ni)) * float(word_size)

            # previous tile write
            if not (mi == prev_m and ni == prev_n):
                dram_write_bytes += float(mn_elems(prev_m, prev_n)) * float(word_size)

            prev_m, prev_n, prev_k = mi, ni, ki

        if not first:
            dram_write_bytes += float(mn_elems(prev_m, prev_n)) * float(word_size)

        # L2 bytes (L2<->core) proxy: re-use the L1-tile batching logic (see L2TileSimulator.simulate_l2_tile_compute_cycle_count)
        l1_tile_m = int(getattr(mapping, "l1_tile_M", 0) or 0)
        l1_tile_n = int(getattr(mapping, "l1_tile_N", 0) or 0)
        l1_tile_k = int(getattr(mapping, "l1_tile_K", 0) or 0)
        if min(l1_tile_m, l1_tile_n, l1_tile_k) <= 0:
            out = super().profile(pcb_module)
            out["traffic_bytes"]["dram"] = float(dram_read_bytes + dram_write_bytes)
            return out

        input_word_size = getattr(getattr(pcb_module.compute_module.core, "systolic_array", None), "input_word_size", word_size)
        output_word_size = getattr(getattr(pcb_module.compute_module.core, "systolic_array", None), "output_word_size", word_size)
        if not isinstance(input_word_size, (int, float)):
            input_word_size = word_size
        if not isinstance(output_word_size, (int, float)):
            output_word_size = word_size

        core_count = int(getattr(pcb_module.compute_module, "core_count", 1) or 1)

        # Count repeated outer tile shapes to reduce work.
        shape_counts = {}
        for mi in range(outer_m):
            tm = dim_size(mi, l2_tile_m, m)
            if tm <= 0:
                continue
            for ni in range(outer_n):
                tn = dim_size(ni, l2_tile_n, n)
                if tn <= 0:
                    continue
                for ki in range(outer_k):
                    tk = dim_size(ki, l2_tile_k, k)
                    if tk <= 0:
                        continue
                    shape_counts[(tm, tn, tk)] = shape_counts.get((tm, tn, tk), 0) + 1

        def l2_bytes_for_tile(tile_m: int, tile_n: int, tile_k: int):
            inner_m = ceil_div(tile_m, l1_tile_m)
            inner_n = ceil_div(tile_n, l1_tile_n)
            inner_k = ceil_div(tile_k, l1_tile_k)

            def dsz(idx: int, tile: int, total: int) -> int:
                start = idx * tile
                if start >= total:
                    return 0
                return min(tile, total - start)

            def mk_sz(bm: int, bk: int) -> int:
                return dsz(bm, l1_tile_m, tile_m) * dsz(bk, l1_tile_k, tile_k)

            def kn_sz(bk: int, bn: int) -> int:
                return dsz(bk, l1_tile_k, tile_k) * dsz(bn, l1_tile_n, tile_n)

            def mn_sz(bm: int, bn: int) -> int:
                return dsz(bm, l1_tile_m, tile_m) * dsz(bn, l1_tile_n, tile_n)

            prev_mk = set()
            prev_kn = set()
            prev_mn_read = set()
            prev_mn_write = set()

            batch = []
            read_bytes = 0.0
            write_bytes = 0.0
            smem_bytes = 0.0
            reg_bytes = 0.0
            batch_count = 0
            active_sum = 0
            # L1 (SRAM) access tracking: compute unit reads from SRAM to feed systolic array.
            # For each L1 tile, the systolic array performs tm*tn*tk MACs.
            # Each MAC logically reads one A element and one B element.
            # The actual SRAM read pattern depends on dataflow (OS/WS/IS), but we can
            # precisely track: A tile loaded once, read tn times (broadcast to columns);
            # B tile loaded once, read tm times (broadcast to rows).
            l1_read_bytes = 0.0  # Total bytes read from SRAM by compute
            l1_write_bytes = 0.0  # Total bytes written to SRAM by compute (C output)

            for bm, bn, bk in Matmul.generate_tile_loops(inner_m, inner_n, inner_k, mapping.l1_loop_order):
                # On-core traffic: each L1 tile streams A/B from core SRAM into compute,
                # and produces/updates C in core SRAM. When bk > 0, C participates in
                # a reduction/update (vector-unit) and incurs extra read+write.
                tm = dsz(bm, l1_tile_m, tile_m)
                tn = dsz(bn, l1_tile_n, tile_n)
                tk = dsz(bk, l1_tile_k, tile_k)
                a_e = tm * tk
                b_e = tk * tn
                c_e = tm * tn

                smem_bytes += float(a_e + b_e) * float(input_word_size)
                if bk > 0:
                    smem_bytes += float(c_e) * float(output_word_size)
                    reg_bytes += float(2 * c_e) * float(output_word_size)
                smem_bytes += float(c_e) * float(output_word_size)

                # L1 access: systolic array reads A and B from SRAM for compute.
                # For output-stationary dataflow:
                # - A[tm,tk] is read once per output column -> read tn times total
                # - B[tk,tn] is read once per output row -> read tm times total
                # This captures the data reuse within the systolic array.
                l1_read_bytes += float(a_e * tn) * float(input_word_size)  # A reuse
                l1_read_bytes += float(b_e * tm) * float(input_word_size)  # B reuse
                # C is written once per tile (accumulated in registers, then stored)
                l1_write_bytes += float(c_e) * float(output_word_size)
                if bk > 0:
                    # For k > 0, need to read previous C for accumulation
                    l1_read_bytes += float(c_e) * float(output_word_size)

                batch.append((bm, bn, bk))
                is_last = bm == inner_m - 1 and bn == inner_n - 1 and bk == inner_k - 1
                if not is_last and len(batch) < core_count:
                    continue

                current_mk = {(x_m, x_k) for (x_m, _x_n, x_k) in batch}
                current_kn = {(x_k, x_n) for (_x_m, x_n, x_k) in batch}
                current_mn_read = {(x_m, x_n) for (x_m, x_n, x_k) in batch if x_k > 0}
                current_mn_write = {(x_m, x_n) for (x_m, x_n, _x_k) in batch}

                mk_new = current_mk - prev_mk
                kn_new = current_kn - prev_kn
                mn_new = current_mn_read - (prev_mn_read | prev_mn_write)
                mk_e = sum(mk_sz(x_m, x_k) for (x_m, x_k) in mk_new)
                kn_e = sum(kn_sz(x_k, x_n) for (x_k, x_n) in kn_new)
                mn_e = sum(mn_sz(x_m, x_n) for (x_m, x_n) in mn_new)

                mn_write_prev = prev_mn_write - current_mn_read
                mn_w_e = sum(mn_sz(x_m, x_n) for (x_m, x_n) in mn_write_prev)

                read_bytes += float(mk_e + kn_e + mn_e) * float(input_word_size)
                write_bytes += float(mn_w_e) * float(output_word_size)

                batch_count += 1
                active_sum += min(len(batch), core_count)

                prev_mk = current_mk
                prev_kn = current_kn
                prev_mn_read = current_mn_read
                prev_mn_write = current_mn_write
                batch = []

            if prev_mn_write:
                mn_w_last = sum(mn_sz(x_m, x_n) for (x_m, x_n) in prev_mn_write)
                write_bytes += float(mn_w_last) * float(output_word_size)

            total_ctas = inner_m * inner_n * inner_k
            total_out_tiles = inner_m * inner_n
            return read_bytes, write_bytes, active_sum, batch_count, smem_bytes, reg_bytes, total_ctas, total_out_tiles, inner_k, l1_read_bytes, l1_write_bytes

        l2_read = 0.0
        l2_write = 0.0
        smem_total = 0.0
        reg_total = 0.0
        total_active = 0.0
        total_batches = 0.0
        grid_size = 0.0
        grid_size_mn = 0.0
        k_partitions_weighted = 0.0
        l1_read_total = 0.0
        l1_write_total = 0.0
        for (tm, tn, tk), cnt in shape_counts.items():
            r_b, w_b, active_sum, batch_count, sm_b, rg_b, ctas, out_tiles, inner_k, l1_r, l1_w = l2_bytes_for_tile(tm, tn, tk)
            l2_read += r_b * cnt
            l2_write += w_b * cnt
            smem_total += sm_b * cnt
            reg_total += rg_b * cnt
            total_active += float(active_sum) * cnt
            total_batches += float(batch_count) * cnt
            grid_size += float(ctas) * cnt
            grid_size_mn += float(out_tiles) * cnt
            l1_read_total += l1_r * cnt
            l1_write_total += l1_w * cnt
            # Average K partitions per output tile (weighted by output tile count).
            k_partitions_weighted += float(inner_k) * float(out_tiles) * cnt

        l2_access = float(l2_read)
        l2_miss = min(float(dram_read_bytes), l2_access) if l2_access > 0 else 0.0
        l2_hit = max(l2_access - l2_miss, 0.0)
        l2_hit_rate = (l2_hit / l2_access) if l2_access > 0 else 0.0

        occupancy = (float(total_active) / float(total_batches * core_count)) if total_batches > 0 else 0.0
        active_ctas = (float(total_active) / float(total_batches)) if total_batches > 0 else 0.0

        out = super().profile(pcb_module)
        out["traffic_bytes"]["dram"] = float(dram_read_bytes + dram_write_bytes)
        l2_bytes = float(l2_read + l2_write)
        out["traffic_bytes"]["l2"] = l2_bytes
        # On-core traffic derived from L1/L0 tiles.
        out["traffic_bytes"]["smem"] = float(smem_total)
        # LLMCompass doesn't separate HW L1 cache vs shared SRAM; treat them as the same level.
        out["traffic_bytes"]["l1"] = float(smem_total)
        out["traffic_bytes"]["reg"] = float(reg_total)
        out["cache"]["l2_hits"] = float(l2_hit)
        out["cache"]["l2_accesses"] = float(l2_access)
        out["cache"]["l2_hit_rate"] = float(l2_hit_rate)
        # L1 (SRAM) access metrics derived from precise tile model traversal.
        # l1_read_total: bytes read from SRAM by systolic array (includes A/B reuse within tiles)
        # l1_write_total: bytes written to SRAM (C output tiles)
        # smem_total: bytes loaded from L2 into SRAM (L1 misses / cold loads)
        l1_access_bytes = float(l1_read_total + l1_write_total)
        # L1 hits = accesses served by SRAM (reuse) = total accesses - cold loads from L2
        # Cold loads = A/B loads from L2 to SRAM (part of smem_total, excluding C writes)
        # For simplicity: l1_hits = l1_reads - (A+B loads from L2)
        # smem_total includes A, B loads and C read/writes; extract A+B portion
        l1_cold_load_bytes = float(smem_total)  # Conservative: all smem traffic is "miss"
        l1_hits_bytes = max(0.0, l1_access_bytes - l1_cold_load_bytes)
        l1_hit_rate = (l1_hits_bytes / l1_access_bytes) if l1_access_bytes > 0 else 0.0
        out["cache"]["l1_accesses"] = float(l1_access_bytes)
        out["cache"]["l1_hits"] = float(l1_hits_bytes)
        out["cache"]["l1_hit_rate"] = float(l1_hit_rate)
        out["parallelism"]["occupancy"] = float(occupancy)
        out["parallelism"]["active_ctas"] = float(active_ctas)
        out["parallelism"]["grid_size"] = float(grid_size)

        # Extra interpretability counters specific to matmul-like kernels.
        # - grid_size_mn: total number of output tiles (mn) across all L2 tiles.
        # - k_partitions_avg: average split-K partitions per output tile.
        k_partitions_avg = (k_partitions_weighted / grid_size_mn) if grid_size_mn > 0 else 0.0
        out["derived"] = {
            "matmul": {
                "grid_size_mn": float(grid_size_mn),
                "k_partitions_avg": float(k_partitions_avg),
                # Loop order hints for interpreting whether k-partitions are scheduled as slices (k-major)
                # or interleaved with m/n traversal. These are descriptive strings, not fixed keys.
                "l1_loop_order": str(getattr(mapping, "l1_loop_order", "") or ""),
                "l2_loop_order": str(getattr(mapping, "l2_loop_order", "") or ""),
            }
        }

        # Cycle breakdown from simulation
        breakdown = getattr(self, '_latency_breakdown', None)
        if breakdown is not None:
            out["latency_breakdown"] = breakdown

        return out

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        if (M == 1 or N == 1) and (
            compile_mode == "heuristic-GPU"
            or compile_mode == "heuristic-our-throughput"
        ):
            working_set_size = M * K + N * K + M * N
            total_io_count = working_set_size * self.data_type.word_size
            io_latency = total_io_count / pcb_module.io_module.bandwidth
            total_flop_count = 2 * M * N * K
            compute_latency = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
            )
            self.latency = max(
                compute_latency, io_latency
            )  # + pcb_module.io_module.latency * 2
            # Store cycle breakdown for bottleneck classification
            freq = pcb_module.compute_module.clock_freq
            dram_cyc = float(io_latency * freq)
            compute_cyc = float(compute_latency * freq)
            total_cyc = float(self.latency * freq)
            self._latency_breakdown = {
                "dram_cycles": dram_cyc,
                "l2_cycles": 0.0,
                "compute_cycles": compute_cyc,
                "total_cycles": total_cyc,
            }
            # GEMV sensitivity: simple roofline → 2x resource only helps if it's the bottleneck
            io_lat_2x = total_io_count / (pcb_module.io_module.bandwidth * 2)
            comp_lat_2x = compute_latency / 2  # 2x cores
            base_lat = self.latency
            self._latency_breakdown['l2_sensitivity_pct'] = 0.0  # no L2 in GEMV
            lat_dram2x = max(compute_latency, io_lat_2x)
            self._latency_breakdown['dram_sensitivity_pct'] = (1 - lat_dram2x / base_lat) * 100 if base_lat > 0 else 0.0
            lat_comp2x = max(comp_lat_2x, io_latency)
            self._latency_breakdown['compute_sensitivity_pct'] = (1 - lat_comp2x / base_lat) * 100 if base_lat > 0 else 0.0
            return self.latency
        if compile_mode == "exhaustive":
            for l2_tile_M_log2 in range(5, ceil(log2(self.computational_graph.M)) + 1):
                l2_tile_M = 2**l2_tile_M_log2
                for l2_tile_N_log2 in range(
                    5, ceil(log2(self.computational_graph.N)) + 1
                ):
                    l2_tile_N = 2**l2_tile_N_log2
                    for l2_tile_K_log2 in range(
                        5, ceil(log2(self.computational_graph.K)) + 1
                    ):
                        l2_tile_K = 2**l2_tile_K_log2
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False
                        for l1_tile_M_log2 in range(5, l2_tile_M_log2 + 1):
                            l1_tile_M = 2**l1_tile_M_log2
                            for l1_tile_N_log2 in range(5, l2_tile_N_log2 + 1):
                                l1_tile_N = 2**l1_tile_N_log2
                                for l1_tile_K_log2 in range(5, l2_tile_K_log2 + 1):
                                    l1_tile_K = 2**l1_tile_K_log2
                                    if (
                                        l1_tile_M * l1_tile_N
                                        + l1_tile_N * l1_tile_K
                                        + l1_tile_M * l1_tile_K
                                        > pcb_module.compute_module.core.SRAM_size
                                        // self.data_type.word_size
                                        // 2
                                    ):
                                        continue
                                    for l2_loop_order in [
                                        "mkn",
                                        "mnk",
                                        "nkm",
                                        "nmk",
                                        "knm",
                                        "kmn",
                                    ]:
                                        for l1_loop_order in [
                                            "mkn",
                                            "mnk",
                                            "nkm",
                                            "nmk",
                                            "knm",
                                            "kmn",
                                        ]:
                                            for (
                                                l0_M_tiling_factor,
                                                l0_N_tiling_factor,
                                                l0_K_tiling_factor,
                                            ) in self.find_permutations(
                                                pcb_module.compute_module.core.systolic_array_count
                                            ):
                                                mapping = self.Mapping(
                                                    l2_tile_M,
                                                    l2_tile_N,
                                                    l2_tile_K,
                                                    is_l2_double_buffering,
                                                    l1_tile_M,
                                                    l1_tile_N,
                                                    l1_tile_K,
                                                    l2_loop_order,
                                                    l1_loop_order,
                                                    l0_M_tiling_factor,
                                                    l0_N_tiling_factor,
                                                    l0_K_tiling_factor,
                                                )
                                                cycle_count = self.simulate(
                                                    self.computational_graph,
                                                    mapping,
                                                    pcb_module,
                                                )
                                                if cycle_count < min_cycle_count:
                                                    min_cycle_count = cycle_count
                                                    best_mapping = mapping
        elif compile_mode == "heuristic-our-throughput":
            i = 0
            for l2_tile_M in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                for l2_tile_N in [
                    l2_tile_M // 4,
                    l2_tile_M // 2,
                    l2_tile_M,
                    l2_tile_M * 2,
                    l2_tile_M * 4,
                    l2_tile_M * 8,
                    l2_tile_M * 16,
                    l2_tile_M * 32,
                    
                ]:
                    l2_tile_K_max = (
                        pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                        - l2_tile_M * l2_tile_N
                    ) // (l2_tile_M + l2_tile_N)
                    if l2_tile_K_max < 1:
                        continue
                    l2_tile_K = min(l2_tile_K_max, K)
                    l2_tile_K = floor(log2(l2_tile_K))
                    l2_tile_K = 2**l2_tile_K
                    working_set_size = (
                        l2_tile_N * l2_tile_K
                        + l2_tile_M * l2_tile_K
                        + l2_tile_M * l2_tile_N
                    )
                    if (
                        working_set_size
                        > pcb_module.compute_module.l2_size // self.data_type.word_size
                    ):
                        continue
                    elif (
                        working_set_size
                        <= pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                    ):
                        is_l2_double_buffering = True
                    else:
                        is_l2_double_buffering = False

                    assert is_l2_double_buffering

                    for l1_tile_M in [32, 64, 128, 256]:
                        l1_tile_M = min(l1_tile_M, l2_tile_M, l2_tile_N)
                        # if l1_tile_M > min(l2_tile_M, l2_tile_N):
                        #     continue
                        l1_tile_N = l1_tile_M
                        l1_tile_K_max = (
                            pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                            - l1_tile_M * l1_tile_N
                        ) // (l1_tile_M + l1_tile_N)
                        if l1_tile_K_max < 1:
                            continue
                        l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                        l1_tile_K = floor(log2(l1_tile_K))
                        l1_tile_K = 2**l1_tile_K

                        if (
                            l1_tile_M * l1_tile_N
                            + l1_tile_N * l1_tile_K
                            + l1_tile_M * l1_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue
                        l2_loop_order = "knm"
                        l1_loop_order = "knm"
                        for (
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        ) in [(2, 2, 1)]:
                            # self.find_permutations(
                            #     pcb_module.compute_module.core.systolic_array_count
                            # ):
                            i += 1
                            # start = time.time()
                            mapping = self.Mapping(
                                l2_tile_M,
                                l2_tile_N,
                                l2_tile_K,
                                is_l2_double_buffering,
                                l1_tile_M,
                                l1_tile_N,
                                l1_tile_K,
                                l2_loop_order,
                                l1_loop_order,
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                            )
                            cycle_count = self.simulate(
                                self.computational_graph,
                                mapping,
                                pcb_module,
                            )
                            # end = time.time()
                            # if i % 1000 == 0:
                            #     print(f"{i} simulation time: {end-start}")
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                best_mapping = mapping
        elif compile_mode == "heuristic-GPU":
            i = 0
            for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
                for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                    if K <= 12288:
                        l2_K_tiling_factor_list = [1, 2, 4, 8]
                    else:
                        l2_K_tiling_factor_list = [
                            K // 1024,
                            K // 2048,
                            K // 4096,
                            K // 8192,
                        ]
                    for l2_K_tiling_factor in l2_K_tiling_factor_list:
                        l2_tile_K = ceil(
                            self.computational_graph.K / l2_K_tiling_factor
                        )
                        l2_tile_K = 2 ** floor(log2(l2_tile_K))
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False

                        for l1_tile_M in [32, 64, 128, 256]:
                            if l1_tile_M > min(l2_tile_M, l2_tile_N):
                                continue
                            l1_tile_N = l1_tile_M
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
                                l2_loop_order = "knm"
                                l1_loop_order = "knm"
                                for (
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    i += 1
                                    start = time.time()
                                    mapping = self.Mapping(
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        is_l2_double_buffering,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
                                    end = time.time()
                                    # if i % 1000 == 0:
                                    #     print(f"{i} simulation time: {end-start}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
            # print("total dse times:", i)
        elif compile_mode == "heuristic-TPU":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 2, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        elif compile_mode == "heuristic-TPU-new":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 1, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # Re-simulate best mapping to capture cycle breakdown
        if best_mapping is not None:
            self.simulate(
                self.computational_graph,
                best_mapping,
                pcb_module,
            )
            # Save original breakdown before sensitivity runs
            saved_breakdown = dict(self._latency_breakdown)
            # --- Sensitivity analysis: re-simulate with 2x resources ---
            cm = pcb_module.compute_module
            im = pcb_module.io_module
            base = float(min_cycle_count)
            # 2x L2 BW
            orig_l2bw = cm.l2_bandwidth_per_cycle
            cm.l2_bandwidth_per_cycle = orig_l2bw * 2
            c2 = self.simulate(self.computational_graph, best_mapping, pcb_module)
            cm.l2_bandwidth_per_cycle = orig_l2bw
            # 2x DRAM BW
            orig_dram = im.bandwidth
            im.bandwidth = orig_dram * 2
            c3 = self.simulate(self.computational_graph, best_mapping, pcb_module)
            im.bandwidth = orig_dram
            # Restore original breakdown from saved copy (avoids extra re-simulate)
            self._latency_breakdown = saved_breakdown
            # Store sensitivity (% speedup from 2x resource)
            self._latency_breakdown['l2_sensitivity_pct'] = (1 - c2 / base) * 100 if base > 0 else 0.0
            self._latency_breakdown['dram_sensitivity_pct'] = (1 - c3 / base) * 100 if base > 0 else 0.0
            self._latency_breakdown['compute_sensitivity_pct'] = 0.0  # matmul: can't simply 2x cores without re-tiling
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        if self.look_up_table is None:
            _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.look_up_table = pd.read_csv(
                os.path.join(_base_dir, "systolic_array_model", f"look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv"),
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            )
        else:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        _acc_dram_io = 0.0
        initial_read = (
            l2_tiles[0, 0, 0].M_K_io_cycle_count + l2_tiles[0, 0, 0].K_N_io_cycle_count
        )
        total_cycle_count += initial_read
        _acc_dram_io += initial_read

        previous_m = 0
        previous_n = 0
        previous_k = 0

        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            l2_tile = l2_tiles[m, n, k]
            previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]

            # current tile read latency
            if m == previous_m and k == previous_k:
                current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
            elif n == previous_n and k == previous_k:
                current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
            else:
                current_tile_read_cycle_count = (
                    l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
                )
            if k > 0 and not (m == previous_m and n == previous_n):
                current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if k > 0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if m == previous_m and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count

            # Accumulate raw sub-component cycles for breakdown
            _acc_dram_io += current_tile_read_cycle_count + previous_tile_write_cycle_count

            # read current tile, compute previous tile, write previous tile
            if mapping.is_l2_double_buffering:  # pipelined
                total_cycle_count += (
                    max(
                        current_tile_read_cycle_count, previous_tile_compute_cycle_count
                    )
                    + previous_tile_write_cycle_count
                )
            else:  # non-pipelined
                total_cycle_count += (
                    current_tile_read_cycle_count
                    + previous_tile_compute_cycle_count
                    + previous_tile_write_cycle_count
                )

            previous_m = m
            previous_n = n
            previous_k = k

        # compute and write last tile
        last_write = l2_tiles[-1, -1, -1].M_N_io_cycle_count
        last_compute = l2_tiles[-1, -1, -1].compute_cycle_count
        if previous_k > 0:
            last_compute += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
        total_cycle_count += last_write + last_compute
        _acc_dram_io += last_write

        # Accumulate L2↔SRAM IO and systolic cycles from all L2 tiles
        _acc_l2_io = 0.0
        _acc_systolic = 0.0
        for m_i in range(l2_tiles.shape[0]):
            for n_i in range(l2_tiles.shape[1]):
                for k_i in range(l2_tiles.shape[2]):
                    tile = l2_tiles[m_i, n_i, k_i]
                    _acc_l2_io += getattr(tile, '_l2_io_cycles', 0.0)
                    _acc_systolic += getattr(tile, '_systolic_cycles', 0.0)

        self._latency_breakdown = {
            "dram_cycles": float(_acc_dram_io),
            "l2_cycles": float(_acc_l2_io),
            "compute_cycles": float(_acc_systolic),
            "total_cycles": float(total_cycle_count),
        }

        return total_cycle_count

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L2 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.K_reduction_cycle_count = ceil(
                M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )
            self.K_reduction_io_count = 2 * M * N * data_type.word_size
            self.M_K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, K, data_type, pcb_module
            )
            self.K_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                K, N, data_type, pcb_module
            )
            self.M_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table
            )

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )

            M_K_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
            )
            M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain

            K_N_tile_size = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
            )
            K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain

            M_N_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
            )
            M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain

            total_cycle_count = 0
            previous_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            )
            previous_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_compute_cycle_count = 0
            _acc_l2_read = 0.0
            _acc_l2_write = 0.0
            _acc_systolic = 0.0
            active_l1_tile_list = []
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ):
                active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
                if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
                ):
                    pass
                elif (
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                ):
                    continue

                assert (
                    len(active_l1_tile_list) <= chiplet_module.compute_module.core_count
                )
                current_batch_Read_M_K = np.zeros(
                    [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                )
                current_batch_Read_K_N = np.zeros(
                    [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Read_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Write_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )

                current_batch_compute_cycle_count = 0
                for i in range(len(active_l1_tile_list)):
                    temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                    current_batch_Read_M_K[temp_m, temp_k] = 1
                    current_batch_Read_K_N[temp_k, temp_n] = 1
                    current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                    current_batch_Write_M_N[temp_m, temp_n] = 1
                    temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                    if temp_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            temp_l1_tile.M
                            * temp_l1_tile.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count,
                        temp_l1_tile_compute_cycle_count,
                    )

                # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
                # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
                current_batch_M_K_read_count = np.sum(
                    (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                    * M_K_tile_size
                )
                current_batch_K_N_read_count = np.sum(
                    (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                    * K_N_tile_size
                )
                current_batch_M_N_read_count = np.sum(
                    (
                        current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                    )
                    * M_N_tile_size
                )
                previous_batch_M_N_write_count = np.sum(
                    (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                    * M_N_tile_size
                )

                # read current batch while compute and write previous batch
                current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
                )
                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                prvious_batch_write_cycle_count = ceil(
                    previous_batch_M_N_write_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )
                # Accumulate raw sub-component cycles
                _acc_l2_read += current_batch_read_cycle_count
                _acc_l2_write += prvious_batch_write_cycle_count
                _acc_systolic += previous_batch_compute_cycle_count

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write
            last_write = ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )
            total_cycle_count += previous_batch_compute_cycle_count + last_write
            _acc_systolic += previous_batch_compute_cycle_count
            _acc_l2_write += last_write

            self._l2_io_cycles = float(_acc_l2_read + _acc_l2_write)
            self._systolic_cycles = float(_acc_systolic)
            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

            return compute_cycle_count

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock)

    def run_on_gpu(
        self,
    ):
        # import subprocess
        # subprocess.run(['nvidia-smi', '-q', '–d', 'CLOCK'])
        input1 = torch.randn(
            self.computational_graph.M,
            self.computational_graph.K,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.randn(
            self.computational_graph.K,
            self.computational_graph.N,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        latencies = []
        input1_dummy = torch.ones(4096, 4096).cuda()
        input2_dummy = torch.ones(4096, 4096).cuda()
        # warmup
        for _ in range(3):
            torch.matmul(input1_dummy, input2_dummy)
            torch.cuda.synchronize()
            time.sleep(1)
        for _ in range(self.iterations):
            # x = torch.matmul(input1_dummy, input2_dummy)  # flush the cache
            # torch.cuda.synchronize()
            start = time.time()
            output = torch.matmul(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert list(output.shape) == [
                self.computational_graph.M,
                self.computational_graph.N,
            ]
            latencies.append(end - start)
            # time.sleep(1)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # min(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print("GPU kernel launch overhead: ", avg_overhead * 1e3, "ms")
        print(latencies)
        return avg_overhead
