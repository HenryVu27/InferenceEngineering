# LLM Quantization Techniques: Comprehensive Research Summary

*Research compiled February 2026. Focus on NVIDIA Blackwell architecture (RTX 5080) relevance.*

---

## Table of Contents

1. [FP8 Inference](#1-fp8-inference)
2. [FP4 on Blackwell (NVFP4)](#2-fp4-on-blackwell-nvfp4)
3. [GPTQ](#3-gptq)
4. [AWQ (Activation-Aware Weight Quantization)](#4-awq-activation-aware-weight-quantization)
5. [SmoothQuant](#5-smoothquant)
6. [QuIP# and Lattice-Based Quantization](#6-quip-and-lattice-based-quantization)
7. [Mixed-Precision Strategies](#7-mixed-precision-strategies)
8. [Calibration Techniques](#8-calibration-techniques)
9. [Comparison Summary Table](#9-comparison-summary-table)

---

## 1. FP8 Inference

### Core Algorithm / Mathematical Insight

FP8 replaces the standard FP16/BF16 number format with an 8-bit floating-point representation, halving memory
footprint and doubling effective bandwidth. Two variants exist:

**E4M3 (4 exponent bits, 3 mantissa bits)**:
- Range: approximately +/-448
- Higher precision (more mantissa bits), narrower dynamic range
- Best for: **forward pass** (weights and activations) where precision matters most
- Can represent NaN but not +/-inf

**E5M2 (5 exponent bits, 2 mantissa bits)**:
- Range: approximately +/-57,344
- Lower precision, much wider dynamic range
- Best for: **backward pass** (gradients) where values vary significantly in magnitude
- Can represent NaN and +/-inf

The quantization formula:
```
Q(x) = clamp(round(x / scale), min_fp8, max_fp8)
x_dequant = Q(x) * scale
```

Where `scale` is computed from the absolute maximum (amax) of the tensor:
```
scale = amax / max_representable_fp8_value
```

### Scaling Strategies

Three scaling granularities exist, in order of increasing accuracy and overhead:

**Per-Tensor Scaling**:
- One FP32 scaling factor per entire tensor
- Simplest but least accurate for tensors with wide dynamic range
- Compatible with all hardware

**Per-Channel Scaling**:
- One scaling factor per output channel (row of weight matrix)
- Better handles per-channel variance
- Moderate overhead

**Per-Block Scaling (MXFP8 -- Microscaling FP8)**:
- Tensors divided into contiguous blocks of **32 values**
- Each block gets a dedicated scaling factor in **E8M0 format** (8-bit exponent only, power-of-2 multiplier)
- Native hardware support on Blackwell 5th-gen Tensor Cores
- Key detail: scaling is directional (row-wise or column-wise); transposing a MXFP8 tensor
  requires automatic requantization from the high-precision input to avoid compounding errors

**Delayed vs. Current Scaling**:
- *Delayed scaling*: uses amax from a window of previous iterations (smoother but can be destabilized by outliers)
- *Current scaling*: computes scale from the current batch (more adaptive, recommended for inference)

### How Tensor Cores Accelerate FP8 Matmuls

Blackwell 5th-gen Tensor Cores perform FP8 matrix multiply-accumulate (MMA) natively:
1. Load FP8 weight tile and FP8 activation tile into Tensor Core registers
2. Perform FP8 x FP8 multiply with FP16 or FP32 accumulation (higher precision accumulator prevents error buildup)
3. The scaling factors are applied before/after the matmul operation
4. For MXFP8: hardware automatically handles per-block scale application and requantization on transpose

This gives roughly **2x throughput** vs FP16 Tensor Core operations at the same clock rate, because twice
as many values fit in the same register/bandwidth.

### Quality vs Speed Trade-off

- **Perplexity impact**: Near-zero. FP8 preserves 99-100% of full-precision benchmark performance even at
  405B+ parameter scale. Some benchmarks show FP8 perplexity marginally *lower* than FP16 (within noise).
- **Speedup**: ~33% improvement in tokens/sec, ~8.5% reduction in TTFT (Mistral 7B benchmarks)
- **Memory**: 2x reduction vs FP16 (model weights only; KV cache can also be FP8)

### Implementation Complexity

**Low-Medium**. FP8 is the easiest quantization format to implement correctly:
- No weight reconstruction or Hessian computation needed
- Simple per-tensor or per-channel scaling via amax computation
- Calibration: run a small set of samples, collect amax statistics, compute scales
- Well-supported in PyTorch (torch.float8_e4m3fn, torch.float8_e5m2), NVIDIA Transformer Engine, TensorRT-LLM

### RTX 5080 Relevance

- **Native FP8 Tensor Core support** via 5th-gen Tensor Cores
- **MXFP8 per-block scaling** is hardware-accelerated (E8M0 scale factors handled natively)
- 960 GB/s GDDR7 bandwidth means FP8 effectively doubles usable bandwidth to ~1920 GB/s equivalent
  for the same model quality
- FP8 is the **recommended baseline** quantization for this hardware -- trivial to implement, minimal quality loss

### Key Papers and References

- [Floating-Point 8: An Introduction (NVIDIA Blog)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [Per-Tensor and Per-Block Scaling Strategies (NVIDIA Blog)](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Using FP8 and FP4 with Transformer Engine (NVIDIA Docs)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [An Investigation of FP8 Across Accelerators for LLM Inference (arXiv)](https://arxiv.org/html/2502.01070v1)
- [33% Faster LLM Inference with FP8 (Baseten)](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/)
- [Unified FP8: Moving Beyond Mixed Precision (LMSYS)](https://lmsys.org/blog/2025-11-25-fp8-rl/)

---

## 2. FP4 on Blackwell (NVFP4)

### Core Algorithm / Mathematical Insight

NVFP4 uses the **E2M1 format**: 1 sign bit, 2 exponent bits, 1 mantissa bit. This yields exactly
**16 representable values** (8 positive, 8 negative including zero):

```
Representable magnitudes: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
Full set: {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6} + negative zero
```

With only 16 values, the key to making FP4 work is a **dual-level scaling system**:

**Level 1 -- Per-Block Scaling (fine-grained)**:
- Tensor is partitioned into micro-blocks of **16 elements** (vs 32 for MXFP4)
- Each block gets an **FP8 E4M3 scaling factor** (1 byte per 16 elements = 0.5 bits overhead per element)
- The E4M3 scale is optimized to minimize collective block error

**Level 2 -- Per-Tensor Scaling (global)**:
- A single **FP32 scaling factor** normalizes the entire tensor's distribution
- Prevents the per-block E4M3 scales from overflowing

The dequantization formula:
```
x_reconstructed = x_fp4 * s_block_fp8 * s_tensor_fp32
```

**Why 16 elements per block (vs MXFP4's 32)?**:
NVFP4 halves the block size compared to MXFP4, providing "twice as many opportunities to match the
local dynamic range of the data." This is critical because with only 16 representable values, the
scale factor must very precisely match the local distribution.

### How It Works with Only 16 Values

The insight is that neural network weight distributions are approximately Gaussian and relatively
smooth. Within a block of 16 consecutive weights, the values cluster tightly enough that mapping
them to 16 discrete levels (after proper scaling) introduces tolerable error. The dual scaling
ensures the 16 levels are positioned optimally for each local region of the tensor.

### Quality vs Speed Trade-off

- **Memory reduction**: 3.5x vs FP16, 1.8x vs FP8
- **Perplexity impact**: ~1% or less accuracy degradation on key benchmarks (DeepSeek-R1-0528,
  quantized from FP8 to NVFP4). Some benchmarks (AIME 2024) show 2% *better* accuracy.
- **Speed**: Up to 4x performance improvement over FP8 inference on Blackwell GPUs.
  Layer-wise speedups up to 6x on RTX 5090. End-to-end 2.2-4x speedup depending on model/hardware.
- **Energy**: Up to 50x more energy efficient per token vs Hopper (Blackwell Ultra)

**Degradation patterns for 4-bit quantization generally**:
- 8-bit: <2% perplexity increase
- 4-bit: 2-8% quality degradation (NVFP4's dual scaling keeps it at the lower end)
- Qwen2.5-8B MMLU score drops from 74.7 to ~69.3 at 4-bit (method-dependent)

### Implementation Complexity

**Medium-High**. More complex than FP8:
- Requires implementing the dual-level scaling system
- Block partitioning logic (16-element blocks)
- FP8 E4M3 scale computation per block
- Calibration via post-training quantization (PTQ)
- Tool support: NVIDIA TensorRT Model Optimizer, LLM Compressor

**Software maturity caveat (as of early 2026)**: Software support for FP4 on consumer GPUs is still
maturing. TensorRT-LLM supports NVFP4, but ecosystem integration with frameworks like vLLM and
llama.cpp is still in progress. This is an area of active development.

### RTX 5080 Relevance

- **Native FP4 Tensor Core support**: 5th-gen Tensor Cores handle NVFP4 natively, including
  element grouping, dynamic scaling, and 4-bit matrix operations
- **1,801 AI TOPS** (FP4 with sparsity) -- the headline compute number
- **16 GB GDDR7**: FP4 reduces a 7B model from ~14 GB (FP16) to ~4 GB (FP4 + scales), leaving
  ample room for KV cache, batch scheduling, and other runtime needs
- **Blackwell-exclusive feature**: This is unavailable on Ada Lovelace (RTX 40-series) or Hopper,
  making it a differentiator for your RTX 5080

### Key Papers and References

- [Introducing NVFP4 (NVIDIA Blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 Trains with Precision of 16-Bit (NVIDIA Blog)](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [Bridging the Gap Between Promise and Performance for FP4 (arXiv)](https://arxiv.org/html/2509.23202v1)
- [NVIDIA Blackwell: The Impact of NVFP4 for LLM Inference](https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/)
- [Scaling NVFP4 Inference for FLUX.2 (NVIDIA Blog)](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus)

---

## 3. GPTQ

### Core Algorithm / Mathematical Insight

GPTQ (Generalized Post-Training Quantization) is a **one-shot, weight-only** post-training quantization
method based on approximate second-order information (the Hessian matrix). It is derived from the
Optimal Brain Quantizer (OBQ) but scaled to handle billions of parameters.

**The core optimization objective**: For each layer with weight matrix W and calibration input X,
minimize the output reconstruction error:

```
argmin_Q ||WX - Q(W)X||_2^2
```

This is equivalent to minimizing the quadratic form:

```
argmin_Q (W - Q(W))^T H (W - Q(W))
```

where H = 2XX^T is the Hessian of the layer's squared error with respect to weights.

**Algorithm steps (layer-by-layer)**:

1. Compute the Hessian H = 2XX^T + lambda*I from calibration data (lambda for numerical stability)
2. Compute the inverse Hessian H^{-1} using Cholesky decomposition
3. Process columns in blocks of size B (typically B=128):
   - For each column q in the block:
     a. Quantize: w_q_hat = quantize(w_q) (round to nearest grid point)
     b. Compute error: delta = w_q - w_q_hat
     c. Compensate: update remaining columns in the block:
        W[:, q+1:] -= delta * H^{-1}[q, q+1:] / H^{-1}[q, q]
   - After the block: update all remaining columns using accumulated block errors

The key insight: **error compensation**. When quantizing one weight introduces error, GPTQ
redistributes that error to unquantized weights in a way that minimizes the output perturbation,
guided by the Hessian (which encodes which weight directions matter most for the output).

**Recent mathematical insight (2025)**: GPTQ's column-by-column procedure is mathematically
equivalent to **Babai's nearest-plane algorithm** for the Closest Vector Problem (CVP) in a lattice
defined by the Hessian. Each update projects the error onto the nearest lattice hyperplane.

### Group Size

Weights are quantized in groups sharing a single scale and zero-point:
- **group_size=128**: Most common default. Good balance of accuracy and overhead.
- **group_size=64**: Better accuracy, more scale parameters (higher memory overhead)
- **group_size=32**: Highest accuracy, most overhead
- **group_size=-1** (per-column): Fastest but lowest accuracy

Smaller group sizes mean each group of weights has a more precisely fitted quantization range.

### Act-Order

The `--act-order` flag quantizes columns in order of **decreasing activation magnitude** rather
than sequential order. This ensures the most important weight columns (those multiplied by the
largest activations) are quantized first, when compensation capacity is highest.

Impact: Dramatically improves accuracy on outlier-heavy models:
- OPT-7B: Wiki2 perplexity improved from 7.15 to 6.09
- OPT-66B 4-bit: 9.55 to 9.34; 3-bit: 14.16 to 9.95

### Quality vs Speed Trade-off

- **Perplexity**: At 4-bit, GPTQ achieves ~6.90 Wiki2 perplexity on Llama-2-7B (vs ~5.47 FP16).
  At 3-bit, significant degradation occurs for smaller models.
- **Quantization time**: Slowest of the major methods -- hours for large models due to Hessian
  computation and iterative optimization
- **Inference speed**: With Marlin kernel backend, GPTQ achieves ~712 tok/s
  (vs ~461 tok/s FP16 baseline on comparable hardware)
- **Model size**: 4-bit reduces model to ~25% of FP16 size (plus group scale overhead)

### Implementation Complexity

**High**. GPTQ requires:
- Hessian computation from calibration data
- Cholesky decomposition of inverse Hessian
- Block-wise column processing with error compensation
- Group quantization with scale/zero-point management
- Libraries: AutoGPTQ (mature, supports Marlin kernel for fast inference)

### RTX 5080 Relevance

- GPTQ is a **weight-only INT4** method -- the RTX 5080 has INT8 Tensor Core support but not
  native INT4 Tensor Cores. GPTQ inference relies on **dequantize-then-matmul** kernels
  (e.g., Marlin) that unpack INT4 to FP16 on-the-fly
- Still highly relevant for memory reduction (fitting 7B models in ~4 GB)
- Consider GPTQ as a baseline to compare against NVFP4 (which has native hardware support)

### Key Papers and References

- [GPTQ: Accurate Post-Training Quantization (ICLR 2023)](https://arxiv.org/abs/2210.17323)
- [AutoGPTQ (GitHub)](https://github.com/AutoGPTQ/AutoGPTQ)
- [The Geometry of LLM Quantization: GPTQ as Babai's Algorithm (2025)](https://arxiv.org/pdf/2507.18553)
- [4-bit Quantization with GPTQ (Towards Data Science)](https://towardsdatascience.com/4-bit-quantization-with-gptq-36b0f4f02c34/)

---

## 4. AWQ (Activation-Aware Weight Quantization)

### Core Algorithm / Mathematical Insight

AWQ's key observation: **not all weights are equally important**. Protecting only **1% of salient
weight channels** can drastically reduce quantization error. Critically, salient channels are
identified by **activation magnitude**, not weight magnitude.

**Why activations, not weights?** A weight channel that consistently sees large activation values
has an outsized effect on the output. Even if the weight values themselves are small, the product
w * x is large, so quantization error in w gets amplified by x.

**Algorithm**:

1. Run calibration data through the model to collect activation statistics
2. For each linear layer, compute per-channel activation magnitudes:
   ```
   avg_activation_j = mean(|X[:, j]|)   # average magnitude per input channel j
   ```
3. Identify salient channels: those with the highest avg_activation_j
4. Apply per-channel scaling to protect salient weights:
   ```
   Y = (X * diag(s^{-1})) * (diag(s) * W)
   ```
   where s_j is derived from activation magnitude, modulated by a tunable hyperparameter

   This is mathematically equivalent to Y = X * W, but the scaling makes the salient
   weight channels larger (easier to represent precisely after quantization) while
   shrinking the corresponding activation channels (compensating)

5. Quantize the scaled weights using standard round-to-nearest

**The mathematical trick**: For a quantization function Q with rounding error delta:
```
Error = ||Q(w) * x - w * x|| = ||delta * x||
```
If x is large for channel j, delta_j contributes disproportionately to error.
By scaling w_j up by s_j before quantization, the relative rounding error delta_j/w_j
shrinks, and the total error is reduced.

### How It Differs from GPTQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Error compensation | Yes (Hessian-based redistribution) | No (scaling only) |
| Backpropagation needed | No | No |
| Computation | Heavy (Hessian, Cholesky) | Light (activation statistics) |
| Quantization speed | Hours for large models | Minutes |
| Accuracy (4-bit) | ~6.90 Wiki2 PPL (Llama-2-7B) | ~6.84 Wiki2 PPL |
| Approach | Minimize layer output error | Protect important channels via scaling |

AWQ is generally **faster to quantize** and **slightly more accurate** than GPTQ, while being
simpler to implement.

### Quality vs Speed Trade-off

- **Perplexity**: 6.84 Wiki2 PPL on Llama-2-7B at 4-bit (slightly better than GPTQ's 6.90)
- **Quantization time**: Minutes (vs hours for GPTQ) -- just a forward pass for statistics
- **Inference speed**: With Marlin kernel, AWQ achieves ~741 tok/s (best among INT4 methods),
  ITL of 12.6ms
- **Memory**: Same as GPTQ -- 4-bit weight-only, ~25% of FP16 size

### Implementation Complexity

**Low-Medium**. Simpler than GPTQ:
- Forward pass to collect activation statistics
- Per-channel scale computation (search for optimal scale)
- Standard round-to-nearest quantization with pre-applied scales
- No Hessian, no iterative optimization
- Libraries: llm-awq (MIT Han Lab), integrated into HuggingFace Transformers, TensorRT-LLM,
  vLLM, DirectML

### RTX 5080 Relevance

- Same as GPTQ: weight-only INT4, uses dequant-on-the-fly kernels (Marlin)
- AWQ's per-channel scaling is compatible with NVFP4's per-block scaling; in principle, AWQ's
  salient channel identification could inform which channels need higher FP8 block scales
- **Recommended over GPTQ** for RTX 5080 due to faster quantization and slightly better accuracy
- Consider comparing AWQ-INT4 against NVFP4 to see which gives better quality-per-bit on your
  specific model

### Key Papers and References

- [AWQ: Activation-aware Weight Quantization (MLSys 2024, Best Paper)](https://arxiv.org/abs/2306.00978)
- [llm-awq (GitHub)](https://github.com/mit-han-lab/llm-awq)
- [AWQ Project Page (MIT Han Lab)](https://hanlab.mit.edu/projects/awq)

---

## 5. SmoothQuant

### Core Algorithm / Mathematical Insight

SmoothQuant tackles a fundamental asymmetry: **weights are easy to quantize (smooth distribution),
but activations are hard (outlier channels)**. Activation outliers can be 100x larger than typical
values, making INT8 quantization of activations destructive.

**The core insight**: Instead of quantizing the hard activation distribution directly, **migrate
the quantization difficulty from activations to weights** using a mathematically equivalent
per-channel transformation.

**The transformation**:
```
Y = X * W = (X * diag(s)^{-1}) * (diag(s) * W) = X_hat * W_hat
```

Where s is a per-channel smoothing vector. This is mathematically equivalent -- the output Y is
identical. But X_hat has smaller outliers (easier to quantize), and W_hat absorbs the difficulty
(weights can handle it because they start smooth).

**Computing the smoothing factors**:
```
s_j = max(|X_j|)^alpha / max(|W_j|)^{1-alpha}
```

where:
- j is the channel index
- alpha in [0, 1] is the **migration strength hyperparameter**
- alpha = 0: no smoothing (all difficulty on activations)
- alpha = 1: all difficulty migrated to weights
- alpha = 0.5: balanced split (default, works well for most models like OPT and BLOOM)
- Larger alpha for models with more severe activation outliers (e.g., GLM-130B uses alpha=0.75)

**Why this works**: Channels with large activation outliers get divided by a large s_j, shrinking
the outlier. The corresponding weight channel gets multiplied by the same s_j, becoming larger.
Since the weight distribution started smooth, it can absorb this scaling without becoming hard
to quantize.

### W8A8 Quantization

SmoothQuant enables **W8A8** (8-bit weights AND 8-bit activations), which is more powerful than
weight-only quantization because:
- Both the matmul inputs are INT8, so Tensor Cores can do INT8 x INT8 -> INT32 accumulate
- This is ~2x faster than FP16 x FP16 on Tensor Cores
- Previous methods (LLM.int8()) kept activations in FP16, losing this speedup

### Quality vs Speed Trade-off

- **Perplexity**: Minimal degradation at W8A8. No accuracy loss on OPT-175B.
- **Speedup**: Faster than FP16 in both PyTorch and FasterTransformer backends (unlike LLM.int8()
  which was often *slower* than FP16). Up to 1.56x speedup with FasterTransformer integration.
- **Memory**: ~50% reduction (both weights and activations are INT8)
- **Unique advantage**: Can halve GPU count -- achieves faster inference using half the GPUs
  compared to FP16 (demonstrated on OPT-175B)

### Implementation Complexity

**Medium**. Requires:
- Forward pass to collect per-channel activation statistics (max absolute values)
- Per-channel smoothing factor computation
- Offline weight transformation (multiply by s)
- Online activation transformation (divide by s, can be fused with previous layer's normalization)
- The smoothing can be folded into LayerNorm parameters for zero runtime overhead
- Libraries: smoothquant (MIT Han Lab), Intel Neural Compressor, AMD Quark

### RTX 5080 Relevance

- **Highly relevant**: RTX 5080 has INT8 Tensor Core support; W8A8 gives true INT8 x INT8 matmuls
- SmoothQuant + INT8 Tensor Cores = significant speedup with minimal quality loss
- Can be combined with other techniques: SmoothQuant the activations, then apply GPTQ/AWQ to
  weights for further compression (e.g., W4A8)
- For memory-constrained scenarios (16 GB GDDR7), W8A8 uses ~50% of FP16 memory,
  fitting a 7B model in ~7 GB with room for KV cache

### Key Papers and References

- [SmoothQuant: Accurate and Efficient PTQ for LLMs (ICML 2023)](https://arxiv.org/abs/2211.10438)
- [smoothquant (GitHub)](https://github.com/mit-han-lab/smoothquant)
- [SmoothQuant Project Page (MIT Han Lab)](https://hanlab.mit.edu/projects/smoothquant)
- [SmoothRot: Combining Channel-Wise Scaling and Rotation (2025)](https://arxiv.org/html/2506.05413)

---

## 6. QuIP# and Lattice-Based Quantization

### Core Algorithm / Mathematical Insight

QuIP# (Quantization with Incoherence Processing, Sharp) achieves state-of-the-art results at
extreme compression (2-3 bits per weight) through three techniques:

**Technique 1: Hadamard Incoherence Processing**

The problem: weight matrices have structure (non-uniform distributions, correlations) that makes
quantization harder. The solution: apply a **randomized Hadamard transform (RHT)** to make weights
approximately i.i.d. Gaussian.

```
W_incoherent = U * W * V^T
```

where U and V are random orthogonal matrices constructed from Hadamard matrices. After
quantization, the inverse transform recovers the original structure:
```
W_reconstructed = U^T * Q(W_incoherent) * V
```

The Hadamard transform is O(n log n) (fast) and has strong theoretical guarantees for
creating incoherence (spreading information uniformly across entries).

**Technique 2: E8 Lattice Codebook (Vector Quantization)**

After incoherence processing, weights are approximately i.i.d. Gaussian -- they form a
"ball-shaped" distribution in high dimensions. The optimal codebook for quantizing such
distributions comes from **lattice theory**.

The **E8 lattice** achieves the densest sphere packing in 8 dimensions. QuIP# uses the
**E8P codebook**: a variant that maps to a 256-entry lookup table (256 x 8 codebook).
Each entry in the codebook is an 8-dimensional vector, and groups of 8 weights are
vector-quantized to the nearest codebook entry.

At 2 bits per weight: 256 codewords = 8 bits per 8 weights = 1 byte per 8 weights.

Why this beats scalar quantization: by quantizing 8 weights jointly, the codebook can
exploit correlations between nearby weights to achieve lower distortion.

**Technique 3: Fine-Tuning**

After initial quantization, QuIP# fine-tunes the quantized model end-to-end to recover
accuracy, capturing inter-layer interactions that layer-wise methods miss.

### QTIP: The Successor

QTIP (Quantization with Trellises and Incoherence Processing, NeurIPS 2024 Spotlight) extends
QuIP# by replacing the E8 lattice codebook with **trellis coded quantization (TCQ)**:

- TCQ uses a **stateful decoder** that separates codebook size from bitrate and effective dimension
- Introduces a "bitshift trellis" structure that is hardware-efficient
- Novel compute-based codes that trade memory for compute (no large lookup table needed)
- QTIP strongly outperforms QuIP#, AQLM, and GPTVQ at all tested bitrates

### Quality vs Speed Trade-off

**QuIP# Perplexity Results**:
- 2-bit Llama models: **usable quality** (unlike AWQ which "falls apart" at 2 bits, and OmniQuant
  which "produces unusable models")
- 3-bit QuIP#: comparable to or better than OmniQuant 4-bit
- QuIP# is the first PTQ method where 3-bit models scale better than "theoretically lossless" 4-bit

**Inference speed**: Slower than GPTQ/AWQ due to vector quantization decode overhead.
The E8P codebook requires a lookup per 8 weights. QTIP improves this with compute-based codes.

### Implementation Complexity

**Very High**. The most complex quantization method:
- Hadamard transform implementation (randomized, with inverse for inference)
- E8 lattice codebook construction and nearest-neighbor search
- Vector quantization (groups of 8 weights)
- Fine-tuning loop
- Custom CUDA kernels for the codebook lookup during inference
- Libraries: quip-sharp, qtip (Cornell RelaxML)

### RTX 5080 Relevance

- QuIP#/QTIP are primarily interesting for **extreme compression** (2-3 bits) where you need
  maximum quality-per-bit
- On the RTX 5080 with 16 GB, 2-bit quantization would reduce a 7B model to ~2 GB, enabling
  very large batch sizes or much larger models
- However, inference kernels are less optimized than Marlin (GPTQ/AWQ) -- expect lower tok/s
- Best use case: fitting models that otherwise would not fit in 16 GB (e.g., 13B+ models at 2-bit)

### Key Papers and References

- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (ICML 2024)](https://arxiv.org/abs/2402.04396)
- [QTIP: Quantization with Trellises and Incoherence Processing (NeurIPS 2024)](https://arxiv.org/abs/2406.11235)
- [quip-sharp (GitHub)](https://github.com/Cornell-RelaxML/quip-sharp)
- [qtip (GitHub)](https://github.com/Cornell-RelaxML/qtip)
- [Even Better, Even Faster Quantized LLMs with QTIP (Together AI Blog)](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip)

---

## 7. Mixed-Precision Strategies

### Core Insight

Not all layers and components in a transformer are equally sensitive to quantization. A mixed-precision
strategy assigns different bit-widths to different parts of the model based on their sensitivity,
optimizing the accuracy-efficiency trade-off.

### Which Layers to Keep at Higher Precision (and Why)

**Highly sensitive (keep at FP16/BF16 or FP8 minimum)**:

1. **Embedding layers (first layer)**: The embedding table maps discrete tokens to continuous
   vectors. Errors here propagate through every subsequent layer. Quantization can shift token
   representations enough to change semantic meaning.

2. **Output/LM head (last layer)**: The final linear projection maps hidden states to vocabulary
   logits. Small perturbations here directly change token probabilities, potentially flipping
   the predicted token entirely.

3. **Attention QKV projections**: Attention computes softmax(QK^T/sqrt(d)), which is sensitive
   to small perturbations in Q and K. Quantization noise in QKV projections can shift attention
   patterns, causing the model to "look at" the wrong tokens.

4. **Attention softmax computation**: Must remain in higher precision (FP16+) to avoid numerical
   instability. The exponential function amplifies small errors.

5. **LayerNorm / RMSNorm parameters**: Normalization layers have very few parameters but
   control the scale of all activations. Errors here are multiplied across the entire hidden
   dimension.

**Moderately sensitive (FP8 or INT8)**:

6. **Attention output projection**: Less sensitive than QKV because it operates on already-computed
   attention outputs.

7. **First and last transformer blocks**: Empirically more sensitive than middle blocks.

**Least sensitive (INT4/FP4 candidates)**:

8. **FFN/MLP layers**: The bulk of model parameters (2/3 of total). The up-projection, gate, and
   down-projection are relatively robust to quantization, especially with per-group scaling.

9. **Middle transformer blocks**: The middle layers tend to be more redundant and can tolerate
   aggressive quantization.

### Sensitivity Detection Methods

1. **First-order Taylor approximation** (LLM-MQ): Measure how model output changes when a layer
   is perturbed. Layer sensitivity = ||gradient * perturbation||. Assign higher precision to
   more sensitive layers.

2. **Hessian trace approximation** (MoPEQ): Use the trace of the Hessian to estimate curvature.
   Layers with higher curvature are more sensitive to weight changes.

3. **Output variance analysis**: Measure the variance of layer outputs when weights are quantized.
   Higher variance = more sensitive.

4. **Empirical sweep**: Quantize each layer individually, measure perplexity change. Simple but
   requires N forward passes for N layers.

### Practical Mixed-Precision Recipes

**Recipe 1: Conservative (minimal quality loss)**
```
Embedding/LM Head:    FP16
Attention (all):      FP8
FFN/MLP:              INT4 (AWQ/GPTQ, group_size=128)
LayerNorm:            FP16
```
Memory: ~30-35% of FP16. Good quality.

**Recipe 2: Aggressive (maximum compression)**
```
Embedding/LM Head:    FP8
Attention QKV:        FP8
Attention output:     FP4 (NVFP4)
FFN/MLP:              FP4 (NVFP4)
LayerNorm:            FP16
```
Memory: ~20-25% of FP16. Some quality degradation.

**Recipe 3: Format hybridization (2025 trend)**
```
Attention layers:     FP8 (handles dynamic range of attention)
MLP layers:           INT4 (better for static weight distributions)
```
This mixes floating-point and integer formats based on layer characteristics.

### Implementation Complexity

**Medium-High**. Requires:
- Per-layer sensitivity analysis (automated or heuristic)
- Different quantization configs per layer type
- Heterogeneous kernel dispatch (FP8 matmul for some layers, INT4 for others)
- Memory layout management for mixed formats
- Libraries: LLM Compressor (vLLM), TensorRT-LLM, LLM-MQ

### RTX 5080 Relevance

- The RTX 5080 supports FP16, BF16, FP8, FP4, INT8 on Tensor Cores -- ideal for mixed-precision
- A practical approach: FP8 attention + NVFP4 MLP, leveraging native hardware support for both
- The 16 GB memory budget makes mixed-precision attractive: keep critical layers at FP8,
  aggressively compress MLP layers to FP4

### Key Papers and References

- [Mixed-Precision Quantization for Language Models: Techniques and Prospects (2025)](https://arxiv.org/html/2510.16805v1)
- [LLM-MQ: Mixed-precision Quantization for Efficient LLM Deployment](https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/5c805adc-b555-499f-9882-5ca35ce674b5.pdf)
- [ATOM: Low-Bit Quantization for Efficient and Accurate LLM Serving (MLSys 2024)](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf)
- [Efficient Mixed-Precision LLM Inference with TurboMind (2025)](https://arxiv.org/html/2508.15601v1)

---

## 8. Calibration Techniques

### Overview

Calibration determines the quantization parameters (scale and zero-point) by running representative
data through the model. The quality of calibration directly impacts quantization accuracy.

### MinMax Calibration

**Algorithm**: Use the observed minimum and maximum values to set the quantization range.

```
scale = (max_val - min_val) / (qmax - qmin)
zero_point = round(qmin - min_val / scale)
```

For symmetric quantization (recommended for weights):
```
scale = max(|max_val|, |min_val|) / qmax
zero_point = 0
```

**Pros**: Simple, fast, deterministic, no hyperparameters
**Cons**: Sensitive to outliers. A single extreme value can expand the range, wasting precision
for the majority of values.

**Variants**:
- `memoryless_minmax`: Recomputes min/max per observation (recommended for PTQ weight quantization)
- `static_minmax`: Tracks running min/max across calibration batches (recommended for PTQ
  activation quantization)

### Percentile Calibration

**Algorithm**: Instead of using the absolute min/max, use the p-th and (100-p)-th percentiles.

```
scale = (percentile_high - percentile_low) / (qmax - qmin)
```

**Typical percentile values**: 99.99% or 99.999%. Lower percentiles (99.9%) are considered
"too aggressive" and can clip too many values.

**Pros**: Robust to outliers. Better precision for the bulk of the distribution.
**Cons**: Clips values beyond the percentile range (acceptable if outliers are rare).
Requires sorting or histogram computation.

**When to use**: When the data has heavy-tailed distributions with rare outliers
(common in LLM activations).

### MSE-Based Calibration

**Algorithm**: Search for the scale factor that minimizes the mean squared error between
the original and quantized values:

```
scale* = argmin_s ||x - dequantize(quantize(x, s), s)||_2^2
```

This is typically solved via grid search over candidate scale values, or golden-section search.

**Pros**: Directly optimizes for reconstruction quality. Often yields the best accuracy.
**Cons**: Most computationally intensive. Requires iterative optimization per tensor.

**Variant**: `memoryless_mse` -- minimizes MSE per observation independently (recommended
for PTQ weight quantization when highest accuracy is needed).

### Cross-Entropy Calibration

**Algorithm**: Instead of minimizing element-wise error, minimize the cross-entropy loss
of the quantized model on calibration data. This directly optimizes for the end task
(next-token prediction).

**Pros**: Task-aware, can find better quantization parameters than MSE
**Cons**: Most expensive, requires full model inference per candidate parameter set

### How Many Calibration Samples Are Needed

| Method | Recommended Samples | Sequence Length |
|--------|-------------------|-----------------|
| GPTQ (Hessian-based) | 128-256 | 512-2048 |
| AWQ (activation statistics) | 128-512 | 512-2048 |
| SmoothQuant | 128-512 | 512-2048 |
| FP8 static quantization | 512-1024 | 512-2048 |
| General PTQ | 200-1000 | varies |

**Key guidelines**:
- **128 samples** is the practical minimum for most methods
- **512 samples** is recommended for production use
- Returns diminish rapidly beyond 512-1024 samples
- Quantization time increases linearly with sample count
- **Diversity matters more than quantity**: Include varied text (prose, code, long context,
  domain-specific jargon, multilingual if relevant). Avoid calibrating only on "easy" text.

**Standard calibration datasets**:
- **C4 (Colossal Clean Crawled Corpus)**: Most commonly used, general web text
- **WikiText-2**: Clean Wikipedia text, good for reproducible benchmarks
- **Pile**: Diverse mix of sources (recommended for broad coverage)
- **Custom domain data**: If deploying for a specific domain, use representative data from that domain

### Implementation for Phase 4

For the Qwen2.5-7B target model on RTX 5080:

```python
# Pseudocode for calibration data preparation
def prepare_calibration_data(tokenizer, dataset="c4", n_samples=512, seq_len=2048):
    """
    Load calibration samples and tokenize them.
    Returns: list of input_ids tensors, each [1, seq_len]
    """
    samples = load_dataset(dataset, split="train", streaming=True)
    calibration_inputs = []
    for sample in itertools.islice(samples, n_samples):
        tokens = tokenizer(sample["text"], max_length=seq_len,
                          truncation=True, return_tensors="pt")
        if tokens.input_ids.shape[1] >= seq_len:
            calibration_inputs.append(tokens.input_ids[:, :seq_len])
    return calibration_inputs

# For each layer, compute calibration statistics
def calibrate_layer(layer, calibration_inputs, method="mse"):
    """
    Run calibration inputs through the layer, collect statistics.
    method: "minmax", "percentile", "mse"
    """
    all_activations = []
    for inp in calibration_inputs:
        with torch.no_grad():
            act = layer(inp)
            all_activations.append(act)

    acts = torch.cat(all_activations, dim=0)

    if method == "minmax":
        scale = acts.abs().max() / max_fp8_value
    elif method == "percentile":
        scale = torch.quantile(acts.abs(), 0.9999) / max_fp8_value
    elif method == "mse":
        scale = search_optimal_scale_mse(acts, max_fp8_value)

    return scale
```

### Key References

- [Optimizing LLMs for Performance and Accuracy with PTQ (NVIDIA Blog)](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)
- [Quantization Concept Guide (HuggingFace)](https://huggingface.co/docs/optimum/concept_guides/quantization)
- [Common PTQ Algorithms (APXML)](https://apxml.com/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/common-ptq-algorithms)
- [Regularized Calibration with Successive Rounding (2025)](https://arxiv.org/html/2602.05902v1)

---

## 9. Comparison Summary Table

### Method Comparison for Qwen2.5-7B on RTX 5080

| Method | Bits | Type | Wiki2 PPL | Model Size | Tok/s | Quantization Time | Implementation Complexity | RTX 5080 Native |
|--------|------|------|-----------|-----------|-------|-------------------|--------------------------|-----------------|
| FP16 (baseline) | 16 | - | ~5.5 | ~14 GB | ~130 | N/A | N/A | Yes (Tensor Core) |
| FP8 (E4M3) | 8 | W+A | ~5.5 | ~7 GB | ~175 | Minutes | Low | Yes (5th gen TC) |
| MXFP8 (per-block) | 8 | W+A | ~5.5 | ~7 GB | ~175 | Minutes | Low-Med | Yes (Blackwell native) |
| SmoothQuant W8A8 | 8 | W+A | ~5.6 | ~7 GB | ~200 | Minutes | Medium | Yes (INT8 TC) |
| NVFP4 | 4 | W | ~5.8-6.0 | ~4 GB | ~250+ | Minutes | Med-High | Yes (Blackwell native) |
| AWQ INT4 | 4 | W | ~6.8 | ~4 GB | ~740* | Minutes | Low-Med | Partial (dequant kernel) |
| GPTQ INT4 | 4 | W | ~6.9 | ~4 GB | ~710* | Hours | High | Partial (dequant kernel) |
| QuIP# 4-bit | 4 | W | ~6.3 | ~4 GB | ~400* | Hours | Very High | No (custom kernel) |
| QuIP# 2-bit | 2 | W | ~8.5 | ~2 GB | ~250* | Hours | Very High | No (custom kernel) |
| QTIP 2-bit | 2 | W | ~7.8 | ~2 GB | ~300* | Hours | Very High | No (custom kernel) |

*Tok/s numbers are approximate and hardware-dependent. AWQ/GPTQ numbers assume Marlin kernel backend.
Actual RTX 5080 numbers will vary -- benchmark everything.

### Decision Guide for Phase 4

```
Start here: What's your priority?
    |
    +-- Maximum quality, minimal effort?
    |       -> FP8 E4M3 per-tensor scaling
    |          (near-zero quality loss, native HW support, trivial to implement)
    |
    +-- Maximum performance, Blackwell-specific?
    |       -> NVFP4 with dual scaling
    |          (native HW support, 3.5x memory reduction, ~1% quality loss)
    |
    +-- Best quality at 4-bit?
    |       -> AWQ (fast quantization, good accuracy, Marlin kernel for speed)
    |       -> GPTQ with act-order (slightly worse but more mature ecosystem)
    |
    +-- Maximum compression (2-3 bit)?
    |       -> QuIP# or QTIP (best quality at extreme compression)
    |
    +-- Both weights AND activations quantized?
    |       -> SmoothQuant W8A8 (enables INT8 Tensor Core matmuls)
    |
    +-- Mixed-precision for best trade-off?
            -> FP8 attention + NVFP4 MLP
               (protects sensitive attention, aggressively compresses bulk parameters)
```

### Recommended Implementation Order for Phase 4

1. **INT8 weight-only (absmax)** -- Simplest possible quantization, good learning exercise
2. **FP8 E4M3** -- Native hardware support, near-zero quality loss, baseline for comparison
3. **NVFP4** -- Blackwell-exclusive, dual scaling system, significant engineering challenge
4. **AWQ** -- Per-channel scaling based on activation analysis, compare INT4 vs NVFP4
5. **SmoothQuant** -- Migration transformation, enables W8A8 for full Tensor Core INT8
6. **Mixed-precision** -- Combine best techniques per layer type
7. **(Stretch) GPTQ** -- Hessian-based, educational for understanding second-order methods
8. **(Stretch) QuIP#** -- Lattice codebooks, educational for understanding VQ and information theory

---

## Additional References

### Comprehensive Surveys
- [A Comprehensive Study on Quantization Techniques for LLMs (2024)](https://arxiv.org/html/2411.02530v1)
- [A Comprehensive Evaluation on Quantization Techniques for LLMs (2025)](https://arxiv.org/html/2507.17417v1)
- [Awesome LLM Quantization (GitHub)](https://github.com/pprp/Awesome-LLM-Quantization)

### Practical Guides
- [The Complete Guide to LLM Quantization with vLLM (JarvisLabs)](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
- [Accelerate Models with Quantization: Recipes for NVFP4, GPTQ, AWQ, SmoothQuant, AutoRound, and FP8](https://kaitchup.substack.com/p/quantizing-and-running-fast-models)
- [GPTQ vs AWQ vs EXL2 comparison (oobabooga)](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)

### Tools and Libraries
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [smoothquant](https://github.com/mit-han-lab/smoothquant)
- [quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)
- [NVIDIA TensorRT Model Optimizer](https://developer.nvidia.com/tensorrt)
- [LLM Compressor (vLLM)](https://docs.vllm.ai/projects/llm-compressor/)
- [NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/)
