from hardware_model.device import Device
from hardware_model.interconnect import (
    InterConnectModule,
    TopologyType,
)
from software_model.utils import Tensor, DataType
from typing import Any
from utils import size
from math import ceil


class CommunicationPrimitive:
    def __init__(self, data_type: DataType) -> None:
        self.data_type = data_type
        # simulation results
        self.latency = None

    def profile(self, pcb_module: Device):
        _ = pcb_module
        # Contract v0.2 requires stable keys; for communication primitives we model
        # buffer movement but do not model cache/occupancy in detail.
        return {
            "traffic_bytes": {
                "dram": 0.0,
                "l2": 0.0,
                "l1": 0.0,
                "smem": 0.0,
                "reg": 0.0,
            },
            "cache": {
                "l2_hits": 0.0,
                "l2_accesses": 0.0,
                "l2_hit_rate": 0.0,
                "l1_hits": 0.0,
                "l1_accesses": 0.0,
                "l1_hit_rate": 0.0,
            },
            "parallelism": {
                "occupancy": 0.0,
                "active_ctas": 0.0,
                "grid_size": 0.0,
            },
        }


class AllReduceMultiPCB(CommunicationPrimitive):
    def __init__(self, data_type: DataType) -> None:
        super().__init__(data_type)
        self.input_shape = None

    def __call__(self, tensor: Tensor) -> Any:
        assert tensor.data_type == self.data_type
        self.input_shape = tensor.shape
        return tensor

    def simulate(self, interconnect_module: InterConnectModule) -> None:
        device_count = interconnect_module.device_count
        link_bandwidth_per_direction = (
            interconnect_module.link_module.bandwidth_per_direction
        )
        link_bandwidth_both_direction = (
            interconnect_module.link_module.bandwidth_both_direction
        )
        link_latency = interconnect_module.link_module.latency
        _flit_size = interconnect_module.link_module.flit_size
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        link_count_per_device = interconnect_module.link_count_per_device
        data_size = size(self.input_shape) * self.data_type.word_size
        if interconnect_module.topology == TopologyType.FC:
            edge_bandwidth_per_direction = (
                link_bandwidth_per_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_bandwidth_both_direction = (
                link_bandwidth_both_direction
                * link_count_per_device
                / (device_count - 1)
            )
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            # stage 1: ring reduce
            latency = (
                edge_latency
                + effective_data_size_per_device / edge_bandwidth_both_direction
            ) * (device_count - 1)
            # stage 2: broadcast
            latency += effective_data_size_per_device / edge_bandwidth_per_direction
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
            return latency
        elif interconnect_module.topology == TopologyType.RING:
            edge_bandwidth = link_bandwidth_per_direction * link_count_per_device
            edge_latency = link_latency
            data_size_per_device = data_size / device_count
            effective_data_size_per_device = (
                header_size
                + ceil(data_size_per_device / max_payload_size) * header_size
                + data_size_per_device
            )
            per_transmission_latency = effective_data_size_per_device / edge_bandwidth
            latency = (edge_latency + per_transmission_latency) * (
                (device_count - 1) * 2
            )
            latency += (
                data_size / interconnect_module.internal_link_bandwidth_per_direction
            )
            self.latency = latency
        else:
            raise NotImplementedError
        return self.latency

    def profile(self, pcb_module: Device):
        out = super().profile(pcb_module)
        input_shape = getattr(self, "input_shape", None)
        word_size = getattr(getattr(self, "data_type", None), "word_size", None)
        if not isinstance(word_size, int) or word_size <= 0:
            word_size = 2
        shape = input_shape if isinstance(input_shape, (list, tuple)) else None
        if shape is not None:
            total = 1
            for d in list(shape):
                if not isinstance(d, int) or d < 0:
                    total = None
                    break
                total *= d
            if total is not None:
                payload = float(total) * float(word_size)
                # Strict per-device memory traffic: read input buffer + write output buffer.
                # (Network traffic is modeled separately by interconnect timing, not as cache traffic.)
                bytes_moved = 2.0 * payload

                out["traffic_bytes"]["dram"] = bytes_moved
                out["traffic_bytes"]["l2"] = bytes_moved
                out["traffic_bytes"]["l1"] = bytes_moved
                out["traffic_bytes"]["smem"] = 0.0
                out["traffic_bytes"]["reg"] = 0.0

                out["cache"]["l2_accesses"] = bytes_moved
                out["cache"]["l2_hits"] = 0.0
                out["cache"]["l2_hit_rate"] = 0.0
                out["cache"]["l1_accesses"] = bytes_moved
                out["cache"]["l1_hits"] = 0.0
                out["cache"]["l1_hit_rate"] = 0.0
        return out


# class P2P:
#     def __init__(self):
#         self.src = None
#         self.dst = None
#         self.tensor = None

#     def __call__(self, src: int, dst: int, tensor: Tensor):
#         self.src = src
#         self.dst = dst
#         self.tensor = tensor

#     def __simulate__(self, src: ChipletModule, dst: ChipletModule, link: LinkModule):
#         pass


class Broadcast:
    def __init__(self):
        self.src = None
        self.tensor = None

    def __call__(self, src: int, tensor: Tensor):
        self.src = src
        self.tensor = tensor

