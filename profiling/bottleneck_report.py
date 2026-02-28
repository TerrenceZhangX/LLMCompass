"""Bottleneck analysis data structures for LLMCompass profiling."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class BottleneckReport:
    """Per-operator bottleneck analysis."""

    operator_name: str
    latency_seconds: float
    flop_count: int
    io_bytes: int
    arithmetic_intensity: float  # flops/byte
    peak_compute_flops: float  # hardware peak for this op type
    peak_bandwidth_bytes: float  # HBM bandwidth (bytes/sec)
    compute_utilization: float  # 0-1, actual / peak compute time
    bandwidth_utilization: float  # 0-1, actual / peak bandwidth time
    bottleneck: str  # "compute_bound" | "memory_bound" | "balanced"
    compute_time: float  # time if only compute-limited (seconds)
    memory_time: float  # time if only memory-limited (seconds)

    def to_dict(self) -> dict:
        return {
            "operator_name": self.operator_name,
            "latency_seconds": self.latency_seconds,
            "flop_count": self.flop_count,
            "io_bytes": self.io_bytes,
            "arithmetic_intensity": self.arithmetic_intensity,
            "peak_compute_flops": self.peak_compute_flops,
            "peak_bandwidth_bytes": self.peak_bandwidth_bytes,
            "compute_utilization": self.compute_utilization,
            "bandwidth_utilization": self.bandwidth_utilization,
            "bottleneck": self.bottleneck,
            "compute_time": self.compute_time,
            "memory_time": self.memory_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BottleneckReport":
        return cls(**d)


@dataclass
class LayerBottleneckReport:
    """Aggregated bottleneck analysis for one transformer layer."""

    layer_index: int
    phase: str  # "prefill" | "decode"
    total_latency_seconds: float
    operator_reports: List[BottleneckReport] = field(default_factory=list)
    dominant_bottleneck: str = ""  # overall layer classification
    compute_fraction: float = 0.0  # fraction of latency from compute-bound ops
    memory_fraction: float = 0.0  # fraction of latency from memory-bound ops

    def __post_init__(self):
        if self.operator_reports and not self.dominant_bottleneck:
            self._classify()

    def _classify(self):
        """Classify the overall layer bottleneck based on operator breakdown."""
        compute_latency = sum(
            r.latency_seconds
            for r in self.operator_reports
            if r.bottleneck == "compute_bound"
        )
        memory_latency = sum(
            r.latency_seconds
            for r in self.operator_reports
            if r.bottleneck == "memory_bound"
        )
        total = self.total_latency_seconds if self.total_latency_seconds > 0 else 1.0
        self.compute_fraction = compute_latency / total
        self.memory_fraction = memory_latency / total
        if self.compute_fraction > 0.6:
            self.dominant_bottleneck = "compute_bound"
        elif self.memory_fraction > 0.6:
            self.dominant_bottleneck = "memory_bound"
        else:
            self.dominant_bottleneck = "balanced"

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "phase": self.phase,
            "total_latency_seconds": self.total_latency_seconds,
            "dominant_bottleneck": self.dominant_bottleneck,
            "compute_fraction": self.compute_fraction,
            "memory_fraction": self.memory_fraction,
            "operator_reports": [r.to_dict() for r in self.operator_reports],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerBottleneckReport":
        ops = [BottleneckReport.from_dict(o) for o in d.get("operator_reports", [])]
        return cls(
            layer_index=d["layer_index"],
            phase=d["phase"],
            total_latency_seconds=d["total_latency_seconds"],
            operator_reports=ops,
            dominant_bottleneck=d.get("dominant_bottleneck", ""),
            compute_fraction=d.get("compute_fraction", 0.0),
            memory_fraction=d.get("memory_fraction", 0.0),
        )


@dataclass
class WorkloadBottleneckReport:
    """Full workload bottleneck analysis report."""

    task_id: str
    model_name: str
    d_model: int
    n_heads: int
    n_layers: int
    batch_size: int
    seq_len: int
    phase: str  # "prefill" | "decode"
    device_count: int
    config_path: str

    # Aggregate results
    total_latency_seconds: float = 0.0
    layer_report: Optional[LayerBottleneckReport] = None
    dominant_bottleneck: str = ""

    # Hardware specs (for roofline context)
    peak_systolic_flops: float = 0.0
    peak_vector_flops: float = 0.0
    peak_hbm_bandwidth: float = 0.0
    ridge_point_systolic: float = 0.0  # flops/byte where compute = memory time
    ridge_point_vector: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "phase": self.phase,
            "device_count": self.device_count,
            "config_path": self.config_path,
            "total_latency_seconds": self.total_latency_seconds,
            "dominant_bottleneck": self.dominant_bottleneck,
            "peak_systolic_flops": self.peak_systolic_flops,
            "peak_vector_flops": self.peak_vector_flops,
            "peak_hbm_bandwidth": self.peak_hbm_bandwidth,
            "ridge_point_systolic": self.ridge_point_systolic,
            "ridge_point_vector": self.ridge_point_vector,
            "layer_report": self.layer_report.to_dict() if self.layer_report else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkloadBottleneckReport":
        layer = None
        if d.get("layer_report"):
            layer = LayerBottleneckReport.from_dict(d["layer_report"])
        report = cls(
            task_id=d["task_id"],
            model_name=d["model_name"],
            d_model=d["d_model"],
            n_heads=d["n_heads"],
            n_layers=d["n_layers"],
            batch_size=d["batch_size"],
            seq_len=d["seq_len"],
            phase=d["phase"],
            device_count=d["device_count"],
            config_path=d["config_path"],
            total_latency_seconds=d.get("total_latency_seconds", 0.0),
            layer_report=layer,
            dominant_bottleneck=d.get("dominant_bottleneck", ""),
            peak_systolic_flops=d.get("peak_systolic_flops", 0.0),
            peak_vector_flops=d.get("peak_vector_flops", 0.0),
            peak_hbm_bandwidth=d.get("peak_hbm_bandwidth", 0.0),
            ridge_point_systolic=d.get("ridge_point_systolic", 0.0),
            ridge_point_vector=d.get("ridge_point_vector", 0.0),
        )
        return report

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "WorkloadBottleneckReport":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
