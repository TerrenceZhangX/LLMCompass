"""
Unified Tile Traversal Model for Operator Profiling.

This module provides a common abstraction for modeling tile-based execution
patterns across different operators (Matmul, Softmax, LayerNorm, etc.).

Key concepts:
- TileTraversal: Models the hierarchical tile traversal (L2 -> L1 -> Compute)
- ReductionTree: Models intra-L1-tile reduction patterns for vector core utilization
- CacheTracker: Tracks cache residency and computes precise hit/miss metrics

The goal is to provide fine-grained metrics that align with actual hardware execution.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Iterator, Any
from enum import Enum
from math import ceil, log2
from abc import ABC, abstractmethod


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class TileDimensions:
    """Describes a tile's dimensions and position in the iteration space."""
    shape: Tuple[int, ...]  # (dim0, dim1, ...) - actual element counts
    index: Tuple[int, ...]  # (idx0, idx1, ...) - tile index in each dimension
    
    @property
    def elements(self) -> int:
        """Total number of elements in this tile."""
        result = 1
        for s in self.shape:
            result *= s
        return result
    
    def bytes(self, word_size: int) -> float:
        """Total bytes for this tile."""
        return float(self.elements) * float(word_size)


@dataclass
class TileEvent:
    """Represents a single tile access event in the execution timeline."""
    pass_id: int              # Which pass (0 for single-pass ops)
    l2_tile_idx: Tuple[int, ...]  # L2 tile index
    l1_tile_idx: Tuple[int, ...]  # L1 tile index within L2 tile
    tile_dims: TileDimensions     # Actual tile dimensions
    access_type: str          # 'read', 'write', 'read_write', 'reduction'
    
    @property
    def global_id(self) -> Tuple:
        """Unique identifier for this tile across all passes."""
        return (self.pass_id, self.l2_tile_idx, self.l1_tile_idx)


class ReductionType(Enum):
    """Types of reduction operations."""
    NONE = "none"           # No reduction (element-wise)
    ROW_MAX = "row_max"     # Softmax: find max per row
    ROW_SUM = "row_sum"     # Softmax/LayerNorm: sum per row
    ROW_MEAN = "row_mean"   # LayerNorm: mean per row
    ROW_VAR = "row_var"     # LayerNorm: variance per row
    TREE = "tree"           # General tree reduction


@dataclass
class ReductionTree:
    """
    Models intra-L1-tile reduction execution on vector cores.
    
    For a reduction over N elements with V vector lanes:
    - Tree depth = ceil(log2(N / V))
    - At each level, active threads halve
    - Vector core utilization decreases as tree progresses
    
    Example: N=1024, V=32 (warp size)
    - Level 0: 32 active threads, each reduces 32 elements -> 32 partial results
    - Level 1: 16 active threads, each reduces 2 partials -> 16 results  
    - Level 2: 8 active threads -> 8 results
    - Level 3: 4 active threads -> 4 results
    - Level 4: 2 active threads -> 2 results
    - Level 5: 1 active thread -> 1 final result
    """
    reduction_dim: int        # Size of dimension being reduced
    vector_lanes: int         # Vector unit width (e.g., 32 for warp)
    reduction_type: ReductionType
    
    @property
    def tree_depth(self) -> int:
        """Number of reduction levels in the tree."""
        if self.reduction_type == ReductionType.NONE:
            return 0
        # First level: each lane reduces (reduction_dim / vector_lanes) elements
        elements_per_lane = max(1, self.reduction_dim // self.vector_lanes)
        # Subsequent levels: binary tree reduction across lanes
        cross_lane_depth = int(ceil(log2(min(self.vector_lanes, self.reduction_dim)))) if self.reduction_dim > 1 else 0
        return cross_lane_depth
    
    @property
    def total_ops(self) -> int:
        """Total reduction operations (add/max/etc)."""
        return self.reduction_dim - 1 if self.reduction_dim > 1 else 0
    
    def utilization_per_level(self) -> List[float]:
        """
        Vector core utilization at each tree level.
        Returns list of utilization fractions [0, 1].
        """
        if self.tree_depth == 0:
            return [1.0]  # No reduction, full utilization
        
        utils = []
        active = min(self.vector_lanes, self.reduction_dim)
        for level in range(self.tree_depth):
            utils.append(float(active) / float(self.vector_lanes))
            active = max(1, active // 2)
        return utils
    
    @property
    def average_utilization(self) -> float:
        """Average vector core utilization across all reduction levels."""
        utils = self.utilization_per_level()
        if not utils:
            return 1.0
        return sum(utils) / len(utils)
    
    def cycles_at_level(self, level: int, flops_per_cycle: float) -> float:
        """Cycles for reduction at a specific tree level."""
        if level >= self.tree_depth:
            return 0.0
        active = min(self.vector_lanes, self.reduction_dim) >> level
        ops_at_level = max(1, active)
        return float(ops_at_level) / flops_per_cycle

    def to_contract_impl(self, pass_name: str = "reduction") -> Dict[str, Any]:
        """
        Convert to Counter Contract v0.2 impl.reduction structure.
        
        This provides L2-level detail for bottleneck attribution.
        Maps to 'ComputeBound' when vector utilization is low.
        """
        utils = self.utilization_per_level()
        avg_util = self.average_utilization
        
        # Determine bottleneck attribution
        # Low vector utilization during reduction indicates compute inefficiency
        maps_to = None
        if avg_util < 0.5:
            maps_to = "ComputeBound"  # Severe underutilization
        elif avg_util < 0.75:
            maps_to = "ComputeBound"  # Moderate underutilization
        
        return {
            "pass_name": pass_name,
            "reduction_type": self.reduction_type.value,
            "reduction_dim": self.reduction_dim,
            "vector_lanes": self.vector_lanes,
            "tree_depth": self.tree_depth,
            "total_ops": self.total_ops,
            "utilization_per_level": utils,
            "average_utilization": avg_util,
            "maps_to": maps_to,
        }


@dataclass
class CacheState:
    """Tracks cache residency for a single cache level."""
    capacity_bytes: float
    resident_tiles: Dict[Tuple, float] = field(default_factory=dict)  # tile_id -> bytes
    current_usage: float = 0.0
    
    # Metrics
    total_accesses: float = 0.0
    total_hits: float = 0.0
    total_bytes_accessed: float = 0.0
    total_bytes_hit: float = 0.0
    
    def access(self, tile_id: Tuple, tile_bytes: float) -> bool:
        """
        Access a tile. Returns True if hit, False if miss.
        Uses simple LRU-like eviction when capacity exceeded.
        """
        self.total_accesses += 1.0
        self.total_bytes_accessed += tile_bytes
        
        if tile_id in self.resident_tiles:
            # Cache hit
            self.total_hits += 1.0
            self.total_bytes_hit += tile_bytes
            return True
        
        # Cache miss - need to load
        # Simple eviction: if over capacity, evict oldest (FIFO approximation)
        while self.current_usage + tile_bytes > self.capacity_bytes and self.resident_tiles:
            oldest = next(iter(self.resident_tiles))
            evicted_bytes = self.resident_tiles.pop(oldest)
            self.current_usage -= evicted_bytes
        
        # Insert new tile
        self.resident_tiles[tile_id] = tile_bytes
        self.current_usage += tile_bytes
        return False
    
    def evict(self, tile_id: Tuple) -> bool:
        """Explicitly evict a tile from cache."""
        if tile_id in self.resident_tiles:
            evicted_bytes = self.resident_tiles.pop(tile_id)
            self.current_usage -= evicted_bytes
            return True
        return False
    
    def clear(self):
        """Clear all tiles from cache (e.g., between passes if no persistence)."""
        self.resident_tiles.clear()
        self.current_usage = 0.0
    
    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_hits / self.total_accesses
    
    @property
    def byte_hit_rate(self) -> float:
        if self.total_bytes_accessed == 0:
            return 0.0
        return self.total_bytes_hit / self.total_bytes_accessed


# =============================================================================
# Tile Traversal Base Class
# =============================================================================

class TileTraversal(ABC):
    """
    Abstract base class for tile-based operator execution modeling.
    
    Subclasses implement specific traversal patterns for different operators:
    - MatmulTraversal: 3D tiling with K-dimension reduction
    - ReductionTraversal: Multi-pass with row-wise reduction (Softmax, LayerNorm)
    - ElementwiseTraversal: Single-pass streaming (GeLU, ReLU)
    """
    
    def __init__(
        self,
        problem_dims: Tuple[int, ...],
        l2_tile_dims: Tuple[int, ...],
        l1_tile_dims: Tuple[int, ...],
        word_size: int = 2,
        l2_capacity: float = float('inf'),
        l1_capacity: float = float('inf'),
        vector_lanes: int = 32,
        core_count: int = 1,
    ):
        self.problem_dims = problem_dims
        self.l2_tile_dims = l2_tile_dims
        self.l1_tile_dims = l1_tile_dims
        self.word_size = word_size
        self.vector_lanes = vector_lanes
        self.core_count = core_count
        
        # Initialize cache state
        self.l2_cache = CacheState(capacity_bytes=l2_capacity)
        self.l1_cache = CacheState(capacity_bytes=l1_capacity)
        
        # Traffic counters (bytes)
        self.dram_read_bytes = 0.0
        self.dram_write_bytes = 0.0
        self.l2_read_bytes = 0.0
        self.l2_write_bytes = 0.0
        self.l1_read_bytes = 0.0
        self.l1_write_bytes = 0.0
        self.smem_bytes = 0.0
        self.reg_bytes = 0.0
        
        # Compute metrics
        self.total_flops = 0.0
        self.vector_util_samples: List[float] = []
        
        # Parallelism metrics
        self.total_tiles = 0
        self.total_waves = 0
        self.peak_active_ctas = 0
    
    @abstractmethod
    def num_passes(self) -> int:
        """Number of passes over the data."""
        pass
    
    @abstractmethod
    def generate_events(self) -> Iterator[TileEvent]:
        """Generate tile access events in execution order."""
        pass
    
    @abstractmethod
    def reduction_tree_for_tile(self, event: TileEvent) -> Optional[ReductionTree]:
        """Return the reduction tree for a given tile event, if applicable."""
        pass
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the tile traversal and collect all metrics.
        Returns a dictionary compatible with operator.profile() output.
        """
        # Reset state
        self._reset_metrics()
        
        # Process all tile events
        current_wave: List[TileEvent] = []
        
        for event in self.generate_events():
            self.total_tiles += 1
            
            # Accumulate into wave (batch of tiles executed in parallel)
            current_wave.append(event)
            
            # Process wave when full or at end
            if len(current_wave) >= self.core_count:
                self._process_wave(current_wave)
                current_wave = []
        
        # Process remaining tiles
        if current_wave:
            self._process_wave(current_wave)
        
        return self._build_profile()
    
    def _reset_metrics(self):
        """Reset all metrics for a fresh execution."""
        self.l2_cache = CacheState(capacity_bytes=self.l2_cache.capacity_bytes)
        self.l1_cache = CacheState(capacity_bytes=self.l1_cache.capacity_bytes)
        
        self.dram_read_bytes = 0.0
        self.dram_write_bytes = 0.0
        self.l2_read_bytes = 0.0
        self.l2_write_bytes = 0.0
        self.l1_read_bytes = 0.0
        self.l1_write_bytes = 0.0
        self.smem_bytes = 0.0
        self.reg_bytes = 0.0
        
        self.total_flops = 0.0
        self.vector_util_samples = []
        
        self.total_tiles = 0
        self.total_waves = 0
        self.peak_active_ctas = 0
    
    def _process_wave(self, wave: List[TileEvent]):
        """Process a wave of tiles executed in parallel."""
        self.total_waves += 1
        self.peak_active_ctas = max(self.peak_active_ctas, len(wave))
        
        for event in wave:
            self._process_tile(event)
    
    def _process_tile(self, event: TileEvent):
        """Process a single tile event."""
        tile_bytes = event.tile_dims.bytes(self.word_size)
        
        # L2 cache access
        l2_tile_id = (event.pass_id, event.l2_tile_idx)
        l2_hit = self.l2_cache.access(l2_tile_id, tile_bytes)
        
        if 'read' in event.access_type:
            self.l2_read_bytes += tile_bytes
            if not l2_hit:
                self.dram_read_bytes += tile_bytes
        
        # L1 cache access (simplified: each SM has its own L1)
        l1_tile_id = event.global_id
        l1_hit = self.l1_cache.access(l1_tile_id, tile_bytes)
        
        if 'read' in event.access_type:
            self.l1_read_bytes += tile_bytes
            self.smem_bytes += tile_bytes
        
        if 'write' in event.access_type:
            self.l1_write_bytes += tile_bytes
            self.l2_write_bytes += tile_bytes
            self.dram_write_bytes += tile_bytes
            self.smem_bytes += tile_bytes
        
        # Process reduction tree if applicable
        reduction = self.reduction_tree_for_tile(event)
        if reduction is not None:
            self.vector_util_samples.append(reduction.average_utilization)
            self.total_flops += float(reduction.total_ops)
    
    def _build_profile(self) -> Dict[str, Any]:
        """Build the profile output dictionary."""
        # Calculate cache metrics
        l2_accesses = self.l2_read_bytes + self.l2_write_bytes
        l2_misses = self.dram_read_bytes + self.dram_write_bytes
        l2_hits = max(0.0, l2_accesses - l2_misses)
        
        l1_accesses = self.l1_read_bytes + self.l1_write_bytes
        # L1 hits = accesses served from L1 without going to L2
        # For now, model as: L1 miss = L2 traffic
        l1_misses = self.l2_read_bytes
        l1_hits = max(0.0, l1_accesses - l1_misses)
        
        # Occupancy
        total_slots = self.total_waves * self.core_count
        occupancy = float(self.total_tiles) / float(total_slots) if total_slots > 0 else 0.0
        
        # Average vector utilization
        avg_vector_util = (
            sum(self.vector_util_samples) / len(self.vector_util_samples)
            if self.vector_util_samples else 1.0
        )
        
        return {
            "traffic_bytes": {
                "dram": self.dram_read_bytes + self.dram_write_bytes,
                "l2": l2_accesses,
                "l1": l1_accesses,
                "smem": self.smem_bytes,
                "reg": self.reg_bytes,
            },
            "cache": {
                "l2_accesses": l2_accesses,
                "l2_hits": l2_hits,
                "l2_hit_rate": l2_hits / l2_accesses if l2_accesses > 0 else 0.0,
                "l1_accesses": l1_accesses,
                "l1_hits": l1_hits,
                "l1_hit_rate": l1_hits / l1_accesses if l1_accesses > 0 else 0.0,
            },
            "parallelism": {
                "occupancy": occupancy,
                "active_ctas": float(self.peak_active_ctas),
                "grid_size": float(self.total_tiles),
            },
            "compute": {
                "total_flops": self.total_flops,
                "avg_vector_utilization": avg_vector_util,
                "reduction_tree_samples": len(self.vector_util_samples),
            },
        }

    def to_contract_metrics(self, time_total_s: float = 0.0) -> Dict[str, Any]:
        """
        Convert TileTraversal profile to Counter Contract v0.2 format.
        
        This method produces metrics compatible with the performance counter
        contract schema, mapping internal fields to the standardized structure.
        
        Args:
            time_total_s: Total execution time in seconds (from simulator).
                          TileTraversal doesn't model time; this must be provided.
        
        Returns:
            Dict in Counter Contract v0.2 format with L0/L1/L2 metrics.
        """
        profile = self._build_profile()
        
        # L0: Mandatory fields
        dram_bytes = profile["traffic_bytes"]["dram"]
        flop_total = profile["compute"]["total_flops"]
        
        # Derived L0 metrics
        ai = flop_total / dram_bytes if dram_bytes > 0 else 0.0
        gbps = dram_bytes / time_total_s / 1e9 if time_total_s > 0 else 0.0
        tflops = flop_total / time_total_s / 1e12 if time_total_s > 0 else 0.0
        
        def metric(value, unit: str, method: str = "simulated", confidence: float = 0.8):
            """Helper to create a metric structure."""
            return {
                "value": value,
                "unit": unit,
                "method": method,
                "confidence": confidence,
                "source": "tile_model",
            }
        
        return {
            # L0: Mandatory
            "time": {
                "total": metric(time_total_s, "s", "simulated"),
            },
            "work": {
                "flop_total": metric(flop_total, "FLOP", "simulated"),
            },
            "traffic": {
                "bytes": {
                    "total": metric(dram_bytes, "bytes", "derived", 0.9),
                    "dram": metric(profile["traffic_bytes"]["dram"], "bytes"),
                    "l2": metric(profile["traffic_bytes"]["l2"], "bytes"),
                    "l1": metric(profile["traffic_bytes"]["l1"], "bytes"),
                    "smem": metric(profile["traffic_bytes"]["smem"], "bytes"),
                    "reg": metric(profile["traffic_bytes"]["reg"], "bytes"),
                },
            },
            # L0 Derived
            "derived": {
                "ai_flop_per_byte": metric(ai, "FLOP/byte", "derived"),
                "effective_gbps": metric(gbps, "GB/s", "derived"),
                "effective_tflops": metric(tflops, "TFLOP/s", "derived"),
            },
            # L1: Cache
            "cache": {
                "hits": {
                    "l2": metric(profile["cache"]["l2_hits"], "count"),
                    "l1": metric(profile["cache"]["l1_hits"], "count"),
                },
                "accesses": {
                    "l2": metric(profile["cache"]["l2_accesses"], "count"),
                    "l1": metric(profile["cache"]["l1_accesses"], "count"),
                },
                "hit_rate": {
                    "l2": metric(profile["cache"]["l2_hit_rate"], "ratio", "derived"),
                    "l1": metric(profile["cache"]["l1_hit_rate"], "ratio", "derived"),
                },
            },
            # L1: Parallelism
            "parallelism": {
                "occupancy": metric(profile["parallelism"]["occupancy"], "ratio"),
                "active_ctas": metric(profile["parallelism"]["active_ctas"], "ctas"),
                "grid_size": metric(profile["parallelism"]["grid_size"], "ctas"),
            },
            # L2: Implementation details (reduction tree)
            "impl": {
                "reduction": {
                    "avg_vector_utilization": metric(
                        profile["compute"]["avg_vector_utilization"], "ratio"
                    ),
                    "tree_samples": metric(
                        profile["compute"]["reduction_tree_samples"], "count"
                    ),
                    "maps_to": "ComputeBound" if profile["compute"]["avg_vector_utilization"] < 0.8 else None,
                },
                "tile_model": {
                    "total_tiles": metric(float(self.total_tiles), "count"),
                    "total_waves": metric(float(self.total_waves), "count"),
                    "num_passes": metric(float(self.num_passes()), "count"),
                },
                # NCU-alignable metrics (新增)
                "memory": {
                    # Sector efficiency: L2 有效利用率
                    # NCU 对应: lts__t_sector_op_*_utilization.pct
                    "sector_efficiency": metric(
                        1.0 if dram_bytes == 0 else min(1.0, profile["traffic_bytes"]["l2"] / (dram_bytes * 4)),
                        "ratio"
                    ),
                    # Coalescing proxy: 假设完美合并=1.0
                    # NCU 对应: l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
                    "coalescing_efficiency": metric(1.0, "ratio"),
                    # L2 write-back bytes (驱逐)
                    # NCU 对应: lts__t_sectors_op_write.sum * 32
                    "l2_writeback_bytes": metric(self.dram_write_bytes, "bytes"),
                },
                "parallelism_detail": {
                    # Waves = ceil(grid_size / active_sms)
                    # NCU 对应: 可通过 kernel duration / sm_cycles 推导
                    "waves": metric(float(self.total_waves), "count"),
                    # Tail effect: 最后一波的 SM 利用率损失
                    "tail_effect": metric(
                        1.0 - (float(self.total_tiles % max(1, self.core_count)) / float(max(1, self.core_count)))
                        if self.total_tiles > 0 else 0.0,
                        "ratio"
                    ),
                    # Theoretical occupancy (根据资源)
                    # NCU 对应: launch__occupancy_per_sm
                    "theoretical_occupancy": metric(
                        min(1.0, float(self.total_tiles) / float(max(1, self.core_count))),
                        "ratio"
                    ),
                },
            },
            # NCU Counter 对照表 (用于文档/调试)
            "ncu_mapping": {
                "traffic.bytes.dram": "dram__bytes.sum",
                "traffic.bytes.l2": "lts__t_bytes.sum",
                "traffic.bytes.l1": "l1tex__t_bytes.sum",
                "cache.hit_rate.l2": "lts__t_sector_hit_rate.pct",
                "cache.hit_rate.l1": "l1tex__t_sector_hit_rate.pct",
                "parallelism.occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
                "parallelism.grid_size": "launch__grid_size",
                "impl.memory.sector_efficiency": "lts__t_sector_op_read_hit_rate.pct",
                "impl.memory.coalescing_efficiency": "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            },
        }

    def to_contract_node(
        self,
        node_id: str,
        node_type: str = "kernel",
        time_total_s: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
        parent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete Contract Node per Counter Contract v0.2.
        
        Args:
            node_id: Unique node identifier (e.g., "phase/prefill/layer_0/softmax")
            node_type: Node type ("kernel", "layer", "phase", etc.)
            time_total_s: Total execution time in seconds
            tags: Optional tags (e.g., {"op_type": "Softmax", "phase": "prefill"})
            parent: Optional parent node id
        
        Returns:
            Complete Contract Node with id, type, metrics, and optional fields.
        """
        node = {
            "id": node_id,
            "type": node_type,
            "metrics": self.to_contract_metrics(time_total_s),
        }
        
        if tags:
            node["tags"] = tags
        
        if parent:
            node["parent"] = parent
        
        return node


def create_contract(
    nodes: List[Dict[str, Any]],
    backend: str = "LLMCompass",
    backend_version: str = "tile_model_v1",
    workload: Optional[Dict[str, Any]] = None,
    arch: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a complete Counter Contract v0.2 document.
    
    Args:
        nodes: List of contract nodes (from TileTraversal.to_contract_node())
        backend: Backend name
        backend_version: Backend version string
        workload: Workload parameters
        arch: Architecture parameters (peak values for normalization)
        artifacts: List of artifact paths (logs, configs, etc.)
    
    Returns:
        Complete Contract document per Counter Contract v0.2 schema.
    
    Example:
        >>> softmax = ReductionTraversal(M=4096, N=768, ...)
        >>> softmax.execute()
        >>> node = softmax.to_contract_node("layer_0/softmax", tags={"op_type": "Softmax"})
        >>> contract = create_contract([node], workload={"batch": 32, "seq": 128})
    """
    return {
        "schema_version": "0.2",
        "backend": {
            "name": backend,
            "version": backend_version,
        },
        "workload": workload or {},
        "arch": arch or {},
        "artifacts": artifacts or [],
        "nodes": nodes,
    }


# =============================================================================
# Concrete Traversal Implementations
# =============================================================================

class ReductionTraversal(TileTraversal):
    """
    Tile traversal for reduction operators (Softmax, LayerNorm).
    
    Execution pattern:
    - Multiple passes over the same data
    - Each pass performs a different operation (max, sum, normalize)
    - Reduction within each row (N dimension)
    - Parallel across rows (M dimension)
    
    Pass structure for Softmax:
        Pass 0: Read x, compute row_max -> write max (reduction)
        Pass 1: Read x and max, compute exp(x-max), sum -> write sum (reduction)  
        Pass 2: Read x, max, sum, compute x_norm = exp(x-max)/sum -> write output
    
    Pass structure for LayerNorm:
        Pass 0: Read x, compute row_mean -> write mean (reduction)
        Pass 1: Read x and mean, compute (x-mean)^2, sum -> write var (reduction)
        Pass 2: Read x, mean, var, compute (x-mean)/sqrt(var) -> write output
    """
    
    def __init__(
        self,
        M: int,  # Number of rows
        N: int,  # Row length (reduction dimension)
        l2_tile_M: int,
        l1_tile_M: int,
        l1_tile_N: int,
        num_reduction_passes: int = 3,
        word_size: int = 2,
        l2_capacity: float = float('inf'),
        l1_capacity: float = float('inf'),
        vector_lanes: int = 32,
        core_count: int = 1,
        persist_across_passes: bool = True,  # Does L2 retain data between passes?
    ):
        super().__init__(
            problem_dims=(M, N),
            l2_tile_dims=(l2_tile_M, N),
            l1_tile_dims=(l1_tile_M, l1_tile_N),
            word_size=word_size,
            l2_capacity=l2_capacity,
            l1_capacity=l1_capacity,
            vector_lanes=vector_lanes,
            core_count=core_count,
        )
        self.M = M
        self.N = N
        self.l2_tile_M = l2_tile_M
        self.l1_tile_M = l1_tile_M
        self.l1_tile_N = l1_tile_N
        self._num_passes = num_reduction_passes
        self.persist_across_passes = persist_across_passes
        
        # Reduction types per pass (default: Softmax pattern)
        self.pass_reductions = [
            ReductionType.ROW_MAX,   # Pass 0: max reduction
            ReductionType.ROW_SUM,   # Pass 1: sum reduction
            ReductionType.NONE,      # Pass 2: normalize (element-wise)
        ]
    
    def num_passes(self) -> int:
        return self._num_passes
    
    def generate_events(self) -> Iterator[TileEvent]:
        """Generate tile events for multi-pass reduction."""
        outer_m = ceil(self.M / self.l2_tile_M)
        inner_m = ceil(self.l2_tile_M / self.l1_tile_M)
        inner_n = ceil(self.N / self.l1_tile_N)
        
        for pass_id in range(self._num_passes):
            # Clear L2 if no persistence between passes
            if pass_id > 0 and not self.persist_across_passes:
                self.l2_cache.clear()
            
            for l2_m in range(outer_m):
                l2_tile_m_actual = min(self.l2_tile_M, self.M - l2_m * self.l2_tile_M)
                
                for l1_m in range(inner_m):
                    m_start = l1_m * self.l1_tile_M
                    if m_start >= l2_tile_m_actual:
                        continue
                    l1_tile_m_actual = min(self.l1_tile_M, l2_tile_m_actual - m_start)
                    
                    for l1_n in range(inner_n):
                        n_start = l1_n * self.l1_tile_N
                        if n_start >= self.N:
                            continue
                        l1_tile_n_actual = min(self.l1_tile_N, self.N - n_start)
                        
                        # Determine access type based on pass
                        if pass_id < self._num_passes - 1:
                            access_type = 'read'  # Intermediate passes just read
                        else:
                            access_type = 'read_write'  # Final pass writes output
                        
                        yield TileEvent(
                            pass_id=pass_id,
                            l2_tile_idx=(l2_m,),
                            l1_tile_idx=(l1_m, l1_n),
                            tile_dims=TileDimensions(
                                shape=(l1_tile_m_actual, l1_tile_n_actual),
                                index=(l2_m * inner_m + l1_m, l1_n),
                            ),
                            access_type=access_type,
                        )
    
    def reduction_tree_for_tile(self, event: TileEvent) -> Optional[ReductionTree]:
        """Return reduction tree based on pass type."""
        pass_id = event.pass_id
        if pass_id >= len(self.pass_reductions):
            return None
        
        reduction_type = self.pass_reductions[pass_id]
        if reduction_type == ReductionType.NONE:
            return None
        
        # Reduction is over the N dimension within the L1 tile
        reduction_dim = event.tile_dims.shape[1]  # N dimension
        
        return ReductionTree(
            reduction_dim=reduction_dim,
            vector_lanes=self.vector_lanes,
            reduction_type=reduction_type,
        )


class ElementwiseTraversal(TileTraversal):
    """
    Tile traversal for element-wise operators (GeLU, ReLU, etc.).
    
    Single pass, no reduction, full vector utilization.
    Simple streaming pattern: read once, write once.
    """
    
    def __init__(
        self,
        total_elements: int,
        l1_tile_size: int = 4096,
        word_size: int = 2,
        vector_lanes: int = 32,
        core_count: int = 1,
    ):
        super().__init__(
            problem_dims=(total_elements,),
            l2_tile_dims=(total_elements,),
            l1_tile_dims=(l1_tile_size,),
            word_size=word_size,
            vector_lanes=vector_lanes,
            core_count=core_count,
        )
        self.total_elements = total_elements
        self.l1_tile_size = l1_tile_size
    
    def num_passes(self) -> int:
        return 1
    
    def generate_events(self) -> Iterator[TileEvent]:
        """Generate tile events for streaming element-wise op."""
        num_tiles = ceil(self.total_elements / self.l1_tile_size)
        
        for tile_idx in range(num_tiles):
            start = tile_idx * self.l1_tile_size
            actual_size = min(self.l1_tile_size, self.total_elements - start)
            
            yield TileEvent(
                pass_id=0,
                l2_tile_idx=(tile_idx,),  # Each L1 tile is its own L2 tile (no reuse)
                l1_tile_idx=(0,),         # Single L1 tile within each L2 tile
                tile_dims=TileDimensions(
                    shape=(actual_size,),
                    index=(tile_idx,),
                ),
                access_type='read_write',
            )
    
    def reduction_tree_for_tile(self, event: TileEvent) -> Optional[ReductionTree]:
        """Element-wise ops have no reduction."""
        return None


# =============================================================================
# Helper Functions
# =============================================================================

def create_softmax_traversal(
    M: int,
    N: int,
    mapping: Any,  # Softmax.Mapping
    device: Any,   # Device
) -> ReductionTraversal:
    """Create a ReductionTraversal configured for Softmax."""
    l2_tile_M = getattr(mapping, 'l2_tile_M', M)
    l1_tile_M = getattr(mapping, 'l1_tile_M', 1)
    l1_tile_N = getattr(mapping, 'l1_tile_N', N)
    
    compute = getattr(device, 'compute_module', None)
    core_count = getattr(compute, 'core_count', 1) if compute else 1
    l2_size = getattr(compute, 'l2_size', float('inf')) if compute else float('inf')
    
    core = getattr(compute, 'core', None) if compute else None
    l1_size = getattr(core, 'SRAM_size', float('inf')) if core else float('inf')
    vector_unit = getattr(core, 'vector_unit', None) if core else None
    vector_count = getattr(vector_unit, 'vector_count', 32) if vector_unit else 32
    
    traversal = ReductionTraversal(
        M=M,
        N=N,
        l2_tile_M=l2_tile_M,
        l1_tile_M=l1_tile_M,
        l1_tile_N=l1_tile_N,
        num_reduction_passes=3,
        word_size=2,
        l2_capacity=float(l2_size),
        l1_capacity=float(l1_size),
        vector_lanes=vector_count,
        core_count=core_count,
        persist_across_passes=True,  # L2 retains data between passes
    )
    
    # Softmax pass reductions: max, sum, normalize
    traversal.pass_reductions = [
        ReductionType.ROW_MAX,
        ReductionType.ROW_SUM,
        ReductionType.NONE,
    ]
    
    return traversal


def create_layernorm_traversal(
    M: int,
    N: int,
    mapping: Any,
    device: Any,
) -> ReductionTraversal:
    """Create a ReductionTraversal configured for LayerNorm."""
    traversal = create_softmax_traversal(M, N, mapping, device)
    
    # LayerNorm pass reductions: mean, var, normalize
    traversal.pass_reductions = [
        ReductionType.ROW_MEAN,
        ReductionType.ROW_VAR,
        ReductionType.NONE,
    ]
    
    return traversal


def create_gelu_traversal(
    total_elements: int,
    device: Any,
) -> ElementwiseTraversal:
    """Create an ElementwiseTraversal configured for GeLU."""
    compute = getattr(device, 'compute_module', None)
    core_count = getattr(compute, 'core_count', 1) if compute else 1
    
    core = getattr(compute, 'core', None) if compute else None
    l1_size = getattr(core, 'SRAM_size', 32768) if core else 32768
    vector_unit = getattr(core, 'vector_unit', None) if core else None
    vector_count = getattr(vector_unit, 'vector_count', 32) if vector_unit else 32
    
    # L1 tile size: fit in SRAM, aligned to vector lanes
    tile_size = min(l1_size // 4, total_elements)  # 4 bytes margin
    tile_size = max(vector_count, (tile_size // vector_count) * vector_count)
    
    return ElementwiseTraversal(
        total_elements=total_elements,
        l1_tile_size=tile_size,
        word_size=2,
        vector_lanes=vector_count,
        core_count=core_count,
    )
