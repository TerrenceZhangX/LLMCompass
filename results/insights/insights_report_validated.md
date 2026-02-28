# Architectural Insights Report — GPU Design for LLM Workloads
## (Critical-Thinking Validated Edition)

**Date:** 2025-07-24
**Methodology:** LLMCompass analytical model, 5082 DSE design points, 61 profiling tasks
**Validation:** Every claim verified against raw data with adversarial scrutiny

---

## Executive Summary

Through systematic design space exploration and critical validation, we identify
**three architectural dimensions** that fundamentally determine LLM inference
performance, and discover that **two commonly swept parameters have zero effect**
in this model — an important finding about both architecture and modeling fidelity.

| Finding | Confidence | Impact |
|---|---|---|
| Decode is 100% memory-bound with 25ms compute floor | HIGH | Critical |
| Systolic array dimension has 30.5× better prefill ROI than core count | HIGH | High |
| SRAM and Global Buffer have zero/minimal latency effect | HIGH (model limitation) | High |
| Prefill-Decode optimization vectors are orthogonal | HIGH | High |
| HBM-decode relationship follows 65.16/BW + 25.20 (not linear) | HIGH | High |
| On-chip memory as dead weight for LLM inference | MEDIUM (needs real validation) | Medium |

---

## Insight 1: The Prefill-Decode Architectural Schism

**Status:** ✅ VALIDATED | **Confidence:** HIGH

### Evidence
- **Decode**: 0/384 operators compute-bound across ALL configurations (compute_fraction = 0.0000)
- **Prefill**: 114/348 operators compute-bound (32.8%), exclusively FFN matmuls
- **Core count has ZERO effect on decode**: 64.96ms at 32, 48, 64, 96, and 128 cores (HBM=4)
- **Systolic dim has ZERO effect on decode**: 64.96ms at dim=8, 16, 32 (with same HBM)

### Deep Analysis
At batch_size=64 decode (the most compute-intensive decode scenario tested), the
highest arithmetic intensity operator (qkv_proj) has AI=62.06 vs ridge point=180.48
— only 34.4% of the way to compute-bound. Even aggressive batching cannot break
decode out of memory-bound regime.

### Implication
Prefill and decode need fundamentally different optimization strategies. Adding
compute resources (cores, larger systolic arrays) benefits only prefill. Adding
memory bandwidth benefits only decode.

---

## Insight 2: Systolic Array Dimension — The Free Lunch

**Status:** ✅ VALIDATED | **Confidence:** HIGH

### Evidence (controlled single-parameter sweep, all else fixed at baseline)
| Systolic Dim | Prefill (ms) | Decode (ms) | Die Area (mm²) | Peak TFLOPS |
|---|---|---|---|---|
| 8 | 47,140 | 64.96 | 501.56 | 46.20 |
| 16 | 12,158 | 64.96 | 514.50 | 184.81 |
| 32 | 3,484 | 64.96 | 566.24 | 739.25 |

- **8→32 speedup**: 13.53× prefill, 0× decode impact
- **Area cost**: 12.9% total die area increase
- **ROI**: 104.9× (speedup per unit area fraction)
- **vs Core count ROI**: 30.5× better (core 32→128: 3.77× speedup for 109.8% area)

### Why This Works
Systolic array dimension scales TFLOPS quadratically (dim²) while area scales
linearly. Doubling the systolic dimension gives 4× compute at ~2× area. Core
count, by contrast, gives linear compute at linear area.

### Caveat
At very large systolic dimensions, utilization may drop for operators with
dimensions smaller than the array. This model does not capture utilization loss.

---

## Insight 3: Decode Latency Formula — Not Linear in Bandwidth

**Status:** ✅ VALIDATED (corrects previous claim) | **Confidence:** HIGH

### Evidence
| HBM Channels | Bandwidth (TB/s) | Decode (ms) | BW × Latency |
|---|---|---|---|
| 3 | 1.229 | 78.22 | 96.11 |
| 4 | 1.638 | 64.96 | 106.43 |
| 5 | 2.048 | 57.01 | 116.76 |
| 6 | 2.458 | 51.71 | 127.08 |
| 8 | 3.277 | 45.08 | 147.73 |

### Fitted Formula
```
decode_latency = 65.16 / BW + 25.20 ms    (R² > 0.9999)
```

- **BW-dependent component**: 65.16 TB·ms (memory transfer time)
- **BW-independent floor**: 25.20 ms (compute/control overhead)
- The floor represents **55.9%** of the fastest decode config
- Max residual: **0.02%** — near-perfect fit

### Correction
Previous analysis claimed "HBM bandwidth perfectly determines decode latency"
and "decode is a pure function of memory bandwidth." This is **incorrect**.
The relationship is `A/BW + B`, not `A/BW`. At high bandwidths, the 25ms floor
dominates, creating severe diminishing returns.

### Implication
Beyond ~4 TB/s bandwidth, further HBM investment yields minimal decode improvement.
The 25ms compute floor becomes the binding constraint. This argues for balanced
BW investment rather than maximizing channels.

---

## Insight 4: SRAM Has ZERO Effect on Latency (Model Limitation)

**Status:** ✅ VALIDATED | **Confidence:** HIGH (for model; uncertain for reality)

### Evidence
- **349 out of 349** configuration groups show ZERO latency variation across SRAM
  sizes (64KB to 512KB)
- Prefill = 12,157.82ms and decode = 64.96ms for ALL SRAM values (fixed other params)
- Only die area changes: 508.50mm² (64KB) → 574.10mm² (512KB)

### Critical Assessment
This is almost certainly a **model limitation**, not an architectural truth.
In real GPUs, SRAM (register files, shared memory) critically affects:
- Data reuse and tiling efficiency
- Reduction of HBM traffic for attention computation
- Fusion opportunities across operators

The LLMCompass roofline model computes operator latency as
`max(compute_time, memory_time)` using peak rates, without modeling the effect
of on-chip data staging on effective bandwidth utilization.

### Implication
**For the paper:** Present this as a model limitation. Do NOT claim SRAM is
unimportant for LLM inference — this contradicts known results (e.g., FlashAttention
relies on SRAM tiling for 2-4× speedup).

---

## Insight 5: Global Buffer — Threshold Effect Only

**Status:** ✅ CORRECTED | **Confidence:** HIGH

### Evidence
- **292 of 361** config groups: ZERO latency effect from global buffer changes
- **69 of 361** groups: threshold effect where GB < 40MB causes slight degradation
  - Example: GB=20MB → 2257ms vs GB≥40MB → 2230ms (1.2% worse)
  - Affects both prefill and decode: GB=20MB → 46.86ms decode vs GB≥40 → 45.08ms
  - Effect appears only at HBM=8 with high compute (cores≥48, systolic≥32)

### Previous Sensitivity Analysis Bug
The sensitivity analysis reported 17.4% prefill range for global buffer. This was
an artifact of the **averaging methodology**: computing means across ALL configs
at each GB level mixes the effects of other parameters (core count, systolic dim)
into the GB sensitivity metric.

### Implication
Global buffer should be sized at a modest threshold (~40MB) to avoid degradation.
Beyond that, additional buffer area is pure waste for LLM workloads.

---

## Insight 6: Prefill Scaling — Sub-Linear with Compute Ceiling

**Status:** ✅ VALIDATED | **Confidence:** HIGH

### Evidence (TFLOPS × Prefill Latency product)
| Config | TFLOPS | Prefill (ms) | Product |
|---|---|---|---|
| sdim=8, cores=32 | 23.10 | 93,885 | 2,168,879 |
| sdim=16, cores=64 | 184.81 | 12,158 | 2,246,905 |
| sdim=32, cores=128 | 1,478.49 | 2,056 | 3,039,749 |

If prefill were purely compute-bound, the product would be constant. The 40%
increase from lowest to highest compute indicates a **memory bandwidth floor**
emerging as compute capacity grows.

### Implication
At very high compute levels (1000+ TFLOPS), prefill transitions from compute-bound
to mixed, and HBM bandwidth becomes relevant even for prefill. This explains why
modern accelerators (H100, B200) scale both compute AND memory bandwidth together.

---

## Insight 7: Operator Bottleneck Invariants

**Status:** ✅ VALIDATED | **Confidence:** HIGH

### Evidence
- Only 4 operators ever become compute-bound: `h_matmul0`, `h_matmul1`,
  `h_matmul2`, `qkv_proj` (all FFN matrix multiplications)
- Attention operators (`q_mul_k`, `a_mul_v`) and all vector ops (`softmax`,
  `layer_norm`, `gelu`) are **always** memory-bound regardless of configuration
- FFN accounts for ~48% of total prefill latency (`h_matmul1` 21.1% + `h_matmul2` 21.1%)

### Implication
Systolic array investment disproportionately benefits FFN. Attention mechanism
improvements must come from bandwidth and algorithmic optimization (e.g.,
FlashAttention, GQA) rather than more compute.

---

## Insight 8: Prefill-Decode Orthogonality

**Status:** ✅ VALIDATED | **Confidence:** HIGH

### Evidence
- Top prefill parameter (systolic_array_dim): **1074.8% prefill range**, 0.16% decode range
- Top decode parameter (hbm_channel_count): **35.2% decode range**, 3.1% prefill range
- These two optimization vectors are effectively independent

### Implication
GPU architects face an inherent tension: die area spent on compute does not help
decode, and area spent on HBM does not help prefill. This motivates:
1. **Disaggregated serving**: Separate prefill and decode to different hardware
2. **Chiplet architectures**: Configurable compute-to-bandwidth ratios
3. **Right-sizing**: Match hardware to the dominant phase of the workload mix

---

## Corrected Design Guidelines

| Priority | Guideline | Evidence |
|---|---|---|
| 1 | Maximize systolic array dim (32+) before adding cores | 30.5× better prefill ROI per mm² |
| 2 | Target ~3-4 TB/s HBM bandwidth; beyond this, 25ms floor limits returns | decode = 65.16/BW + 25.20ms |
| 3 | Core count sweet spot: 64-96 | Linear scaling up to ~128 but ROI is low |
| 4 | Set global buffer at modest threshold (40MB+) | Below threshold: ~1.2% penalty. Above: zero benefit |
| 5 | SRAM sizing: model doesn't capture — use real benchmarks | Zero modeled effect is a model limitation |
| 6 | Consider prefill-decode disaggregation | Orthogonal optimization vectors |

---

## Sensitivity Analysis Methodology Correction

The original sensitivity analysis computed **parameter-level means** across all
designs at each parameter value. This methodology confounds the effect of the
target parameter with the distribution of other parameters, producing misleading
results:

| Parameter | Reported Range | Actual Effect |
|---|---|---|
| SRAM | 8.1% | **0% (zero)** — pure averaging artifact |
| Global Buffer | 17.4% | **0-1.2%** — threshold effect only |
| Core Count & 192 | "non-monotonic regression" | **monotonic improvement** — averaging artifact from only 3 192-core designs |

**Corrected approach:** Use controlled single-parameter sweeps with all other
parameters held constant to isolate true parameter sensitivity.

---

## Model Limitations Identified

1. **No SRAM locality modeling**: On-chip memory has zero latency effect; roofline
   uses peak bandwidth regardless of data reuse
2. **No utilization modeling**: Large systolic arrays may underutilize for small
   operators (e.g., layer_norm with small hidden dim)
3. **Decode compute floor unexplained**: The 25.20ms floor in `decode = 65.16/BW + 25.20`
   may represent real compute overhead or a modeling artifact
4. **No inter-device communication modeling**: Multi-GPU decode degradation may
   be understated or overstated
