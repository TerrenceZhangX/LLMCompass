from utils import size
from typing import List
from hardware_model.device import Device
from software_model.utils import Tensor, DataType


class Operator:
    def __init__(
        self,
        flop_count,
        load_count,
        store_count,
        peak_memory_usage,
        data_type: DataType,
        gpu_device=None,
        verbose=True,
    ):
        self.flop_count = flop_count
        self.load_count = load_count
        self.store_count = store_count
        self.io_count = load_count + store_count
        self.peak_memory_usage = peak_memory_usage
        self.data_type = data_type
        self.gpu_device = gpu_device
        self.verbose = verbose
        self.log = ""
        self.comment = ""
        # simulation results
        self.latency = 0
        self.latency_on_gpu = 1
        self.is_io_bound = None
        # run on gpu
        self.iterations = 50

    def profile(self, pcb_module: Device):
        """Return structured profiling stats for this operator.

        This is intentionally schema-agnostic and can be consumed by external
        integrations (e.g., ArchCopilot Counter Contract) to populate stable keys.

        Keys are best-effort; values may be None when not modeled.
        """
        _ = pcb_module
        word_size = getattr(self.data_type, "word_size", None)
        if not isinstance(word_size, int) or word_size <= 0:
            word_size = 2

        # Most operators use load/store counts as *element counts*.
        # Convert to bytes as a conservative DRAM traffic estimate.
        io_count = getattr(self, "io_count", None)
        bytes_total = None
        if isinstance(io_count, (int, float)) and io_count >= 0:
            bytes_total = float(io_count) * float(word_size)

        if bytes_total is None:
            bytes_total = 0.0

        # Default strict semantics for operators without an explicit tiling model:
        # treat them as streaming/global-memory kernels.
        dram_bytes = float(bytes_total)
        l2_bytes = float(bytes_total)
        l1_bytes = float(bytes_total)
        smem_bytes = 0.0
        reg_bytes = 0.0

        # Conservative cache proxy: streaming access -> treat as misses.
        l2_access = float(l2_bytes)
        l2_hit = 0.0
        l1_access = float(l1_bytes)
        l1_hit = 0.0

        return {
            "traffic_bytes": {
                "dram": dram_bytes,
                "l2": l2_bytes,
                "l1": l1_bytes,
                "smem": smem_bytes,
                "reg": reg_bytes,
            },
            "cache": {
                "l2_hits": l2_hit,
                "l2_accesses": l2_access,
                "l2_hit_rate": (l2_hit / l2_access) if l2_access > 0 else 0.0,
                "l1_hits": l1_hit,
                "l1_accesses": l1_access,
                "l1_hit_rate": (l1_hit / l1_access) if l1_access > 0 else 0.0,
            },
            "parallelism": {
                "occupancy": 0.0,
                "active_ctas": 0.0,
                "grid_size": 0.0,
            },
            "latency_breakdown": None,
        }

    class mapping:
        pass


# auxilary functions


class Reshape(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input_shape = None
        self.output_shape = None

    def __call__(self, inp: Tensor, output_shape: List[int]) -> Tensor:
        assert inp.size == size(output_shape)
        self.flop_count = 0
        self.load_count = 0
        self.store_count = 0
        self.io_count = 0
        self.peak_memory_usage = 0
        self.input_shape = inp.shape
        self.output_shape = output_shape
        output = Tensor(output_shape, self.data_type)
        return output


class Concat(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.concat_dim = None
        self.output_shape = None

    def __call__(self, input1: Tensor, input2: Tensor, concat_dim: int) -> Tensor:
        assert len(input1.shape) == len(input2.shape)
        for i in range(len(input1.shape)):
            if i != concat_dim:
                assert input1.shape[i] == input2.shape[i]
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.concat_dim = concat_dim
        self.flop_count = 0
        self.load_count = input1.size + input2.size
        self.store_count = input1.size + input2.size
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = (input1.size + input2.size) * 2
        self.output_shape = (
            input1.shape[:concat_dim]
            + [input1.shape[concat_dim] + input2.shape[concat_dim]]
            + input1.shape[concat_dim + 1 :]
        )
        output = Tensor(self.output_shape, self.data_type)
        return output


class Transpose(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input_shape = None
        self.output_shape = None
        self.permute = None

    def __call__(self, inp: Tensor, permute: List[int]) -> Tensor:
        assert len(inp.shape) == len(permute)
        self.input_shape = inp.shape
        self.permute = permute

        self.flop_count = 0
        self.load_count = size(inp.shape)
        self.store_count = self.load_count
        self.io_count = self.load_count + self.store_count
        self.peak_memory_usage = inp.size * 2

        self.output_shape = [self.input_shape[i] for i in permute]
        output = Tensor(self.output_shape, self.data_type)
        return output
