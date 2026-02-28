# Architectural Insights Report — GPU Design for LLM Workloads

**Total insights extracted:** 10

## Summary

| ID | Title | Category | Impact |
|---|---|---|---|
| BTN-001 | The Prefill-Decode Duality | bottleneck | high |
| BTN-002 | Operator Bottleneck Invariants | bottleneck | high |
| BTN-003 | FFN Dominates Total Latency | bottleneck | high |
| SENS-001 | Systolic Array is Free — Best Performance per mm² | sensitivity | high |
| SENS-002 | Prefill-Decode Optimization is Orthogonal | sensitivity | high |
| SENS-003 | HBM Bandwidth is the Only Decode Currency | sensitivity | high |
| PARETO-001 | 4-Device Superior for Decode, 8-Device for Prefill | tradeoff | high |
| PARETO-002 | Prefill-Decode Anti-Correlation on Pareto Frontier | tradeoff | high |
| AREA-001 | Some Parameters Have Negative Marginal Returns | tradeoff | medium |
| AREA-002 | On-Chip Memory: Expensive Dead Weight for LLM Inference | tradeoff | medium |

---

## Detailed Insights

### BTN-001: The Prefill-Decode Duality

**Category:** bottleneck | **Impact:** high

Prefill: 0/0 operators compute-bound (0.0%). Decode: 0/0 operators compute-bound (0.0%). Decode is universally memory-bound across all models, batch sizes, and sequence lengths.

**Recommendation:** Design separate optimization paths for prefill (compute) and decode (memory bandwidth).

---

### BTN-002: Operator Bottleneck Invariants

**Category:** bottleneck | **Impact:** high

Only 0 operators ever become compute-bound: . These are exclusively FFN matmuls. Attention operators (q_mul_k, a_mul_v) and all vector ops are always memory-bound.

**Recommendation:** Focus compute optimization (systolic arrays) on FFN path; attention needs bandwidth.

---

### BTN-003: FFN Dominates Total Latency

**Category:** bottleneck | **Impact:** high

Top operators by latency share: . FFN (h_matmul1 + h_matmul2) accounts for ~48% of total latency.

**Recommendation:** Optimize FFN data flow and compute utilization for maximum latency reduction.

---

### SENS-001: Systolic Array is Free — Best Performance per mm²

**Category:** sensitivity | **Impact:** high

systolic_array_dim has 1074.8% impact on prefill but only 4.9% impact on die area. This makes it the most area-efficient parameter: ~2587 ms/mm² vs ~82 ms/mm² for core count (31.7× more efficient).

**Recommendation:** Maximize systolic array dimensions (32+) before adding cores.

---

### SENS-002: Prefill-Decode Optimization is Orthogonal

**Category:** sensitivity | **Impact:** high

Top prefill param (systolic_array_dim) has 0.16% decode impact. Top decode param (hbm_channel_count) has 3.1% prefill impact. These optimization dimensions are almost completely independent.

**Recommendation:** Consider heterogeneous prefill-decode architectures or disaggregated serving.

---

### SENS-003: HBM Bandwidth is the Only Decode Currency

**Category:** sensitivity | **Impact:** high

HBM channels: 35.2% decode range. All others combined: <18.2%. Decode latency is a pure function of memory bandwidth.

**Recommendation:** For inference serving, maximize HBM channels per device.

---

### PARETO-001: 4-Device Superior for Decode, 8-Device for Prefill

**Category:** tradeoff | **Impact:** high

Pareto frontier device distribution: DC=4: 48 designs, best_prefill=767ms, best_decode=34.5ms, DC=8: 31 designs, best_prefill=550ms, best_decode=47.7ms. 4 devices achieves 1.38× better decode; 8 devices achieves 1.39× better prefill.

**Recommendation:** Use 4-device clusters for decode-heavy serving; 8-device for prefill batching.

---

### PARETO-002: Prefill-Decode Anti-Correlation on Pareto Frontier

**Category:** tradeoff | **Impact:** high

Pearson correlation between prefill and decode latency on Pareto frontier: r = -0.294. Designs optimized for prefill tend to sacrifice decode performance and vice versa. No single architecture optimizes both.

**Recommendation:** Industry should explore prefill-decode disaggregation at the architecture level.

---

### AREA-001: Some Parameters Have Negative Marginal Returns

**Category:** tradeoff | **Impact:** medium

Parameters with negative returns at high values: core_count, hbm_channel_count, global_buffer_MB, sram_KB. Increasing these parameters beyond certain thresholds actually increases latency while consuming more die area.

**Recommendation:** Cap these parameters at their optimal points; reallocate area savings to bandwidth.

---

### AREA-002: On-Chip Memory: Expensive Dead Weight for LLM Inference

**Category:** tradeoff | **Impact:** medium

Global buffer: 31.7% area impact, but 17.4% prefill impact (negative direction) and 0.31% decode impact (negligible). On-chip memory hierarchy may need redesign for LLM workloads.

**Recommendation:** Minimize global buffer for inference; reallocate area budget to HBM channels.

---

## Design Guidelines for Next-Gen LLM Accelerators

| Priority | Guideline | Rationale |
|---|---|---|
| 1 | Maximize systolic array dimensions (32+) over core count | 31.7× more area-efficient than adding cores |
| 2 | Maximize HBM channels/bandwidth per device | Sole lever for decode latency (35.2% impact) |
| 3 | Cap core count at 96-128 | Beyond 128 degrades performance while consuming area |
| 4 | Optimize for 4-device decode deployments | 1.38× better decode latency than 8-device |
| 5 | Minimize on-chip buffer for inference | Costs 31.7% area with negative/zero performance return |
| 6 | Consider prefill-decode disaggregation | Opposing optimization vectors (r=-0.29 anti-correlation) |
