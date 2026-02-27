# Inference Engineering Roadmap

A comprehensive, phase-by-phase guide to building an LLM inference engine from scratch.

**Hardware**: NVIDIA RTX 5080 (16GB GDDR7, 960 GB/s, 10,752 CUDA cores, Blackwell SM 10.x, native FP4/FP8)
**Target Model**: Qwen2.5-7B-Instruct (7.61B params, 28 layers, GQA 28Q/4KV, head dim 128)
**Fallback**: Llama-3.1-8B-Instruct

> Deep-dive reference files live in `docs/concepts/`. This roadmap links to them where relevant.

---

## Table of Contents

- [Phase 1 — Naked Forward Pass](#phase-1--naked-forward-pass)
- [Phase 2 — KV Cache + Memory Management](#phase-2--kv-cache--memory-management)
- [Phase 3 — Custom GPU Kernels](#phase-3--custom-gpu-kernels)
- [Phase 4 — Quantization](#phase-4--quantization)
- [Phase 5 — Serving: Continuous Batching + HTTP](#phase-5--serving-continuous-batching--http)
- [Phase 6 — Speculative Decoding](#phase-6--speculative-decoding)
- [Phase 7 — Structured Output / Constrained Decoding](#phase-7--structured-output--constrained-decoding)
- [Phase 8 — Advanced Attention Mechanisms](#phase-8--advanced-attention-mechanisms)
- [Phase 9 — Profiling + Benchmarking](#phase-9--profiling--benchmarking)
- [Phase 10 — Advanced Topics](#phase-10--advanced-topics)
- [Key References](#key-references)

---

## Phase 1 — Naked Forward Pass

**Goal**: Load Qwen2.5-7B and run inference using ONLY matrix operations. No `model.generate()`, no `AutoModelForCausalLM`. You touch every weight tensor yourself.

### What to Build

1. **Weight loader** — parse `model.safetensors.index.json`, load 4 safetensors shards, map tensor names to layer structure (28 layers, 12 tensors per layer + 3 global)
2. **Embedding lookup** — `embed_tokens.weight [152064, 3584]`, simple table index
3. **Transformer block** (repeat 28x):
   - RMSNorm (eps=1e-6): `x_norm = (x / sqrt(mean(x²) + eps)) * weight`
   - QKV projections: Q `[3584→3584]`, K `[3584→512]`, V `[3584→512]` — **all with bias**
   - RoPE: base θ=1,000,000, head_dim=128, 64 frequency pairs
   - GQA: 28 Q heads, 4 KV heads, group size 7 — expand KV via `repeat_interleave(7, dim=2)`
   - Causal attention: `softmax(QK^T / 11.3137) @ V`
   - Output projection: `o_proj [3584→3584]` — **no bias**
   - Residual connection
   - RMSNorm (post-attention)
   - SwiGLU FFN: `down_proj(SiLU(gate_proj(x)) * up_proj(x))` — 3 matrices, no bias
   - Residual connection
4. **Final norm + LM head** — RMSNorm → `lm_head [3584→152064]` (untied, no bias)
5. **Autoregressive decode loop** — prefill (parallel) → decode (one token at a time)
6. **Sampling** — greedy, temperature, top-k, top-p, repetition penalty
7. **Chat template** — ChatML format with `<|im_start|>`, `<|im_end|>` markers, two EOS tokens (151645, 151643)

### Key Architecture Details (Qwen2.5-7B)

```
Hidden dim:     3,584          Layers:        28
FFN dim:        18,944         Vocab:         152,064
Q heads:        28             KV heads:      4
Head dim:       128            GQA ratio:     7:1
RoPE theta:     1,000,000     RMSNorm eps:   1e-6
Attention bias: Q,K,V yes / O no
MLP bias:       None
Embeddings:     Not tied (separate embed_tokens and lm_head)
```

### Critical Implementation Gotchas

- **Qwen has QKV bias** — unlike Llama which has no bias at all. Missing this causes silent output mismatch.
- **SwiGLU uses 3 projections** — gate_proj, up_proj, down_proj. The 5.29x expansion ratio (18944/3584) compensates for the third matrix to maintain total parameter count.
- **RoPE applies to full head_dim** (128) — 64 rotation pairs, not partial.
- **Two EOS tokens** — generation stops on either `<|im_end|>` (151645) OR `<|endoftext|>` (151643).
- **No BOS token** — Qwen doesn't prepend BOS, unlike Llama.

### Tensor Shapes Cheat Sheet (Prefill: B=1, S=prompt_len)

```
After embedding:        [1, S, 3584]
After Q projection:     [1, S, 28, 128]    (reshape from [1, S, 3584])
After K projection:     [1, S, 4, 128]     (reshape from [1, S, 512])
After V projection:     [1, S, 4, 128]
After KV expansion:     [1, S, 28, 128]    (repeat_interleave by 7)
Attention scores:       [1, 28, S, S]      (QK^T / sqrt(128))
Attention output:       [1, 28, S, 128]    → reshape → [1, S, 3584]
After gate_proj:        [1, S, 18944]
After SiLU * up:        [1, S, 18944]
After down_proj:        [1, S, 3584]
Logits:                 [1, S, 152064]
```

### Memory Math

| Precision | Model Size | Fits 16GB? | KV headroom |
|-----------|-----------|------------|-------------|
| BF16      | 15.2 GB   | Barely — no room for KV cache | 0 |
| FP8       | 7.6 GB    | Yes, 8GB for KV + activations | ~32K context |
| FP4       | 3.8 GB    | Yes, 12GB headroom | ~128K context |

**Phase 1 recommendation**: Load weights in BF16, use FP8 for actual inference to leave room for KV cache. Or load in BF16 with short context (≤1K tokens) for correctness validation.

### Validation

- Compare greedy output **token-for-token** with HuggingFace `transformers`
- Use `torch.allclose(output, reference, atol=1e-4, rtol=1e-3)` for intermediate tensors
- Check RoPE frequencies against reference
- Verify all tensor shapes at every layer

### Benchmark Target

Measure tokens/sec vs HuggingFace baseline. **Your implementation will be slower** (no KV cache, no kernel fusion, no batching) — that's expected. The point is correctness.

### Concepts to Learn

- Transformer architecture: not the theory, the actual tensor shapes and operations
- Safetensors format and weight sharding
- Prefill (compute-bound, parallel) vs decode (memory-bound, sequential)
- BPE tokenization, special tokens, chat templates
- Why decode is memory-bandwidth-bound: AI ≈ 1 FLOP/byte for mat-vec at B=1

### References

- [Tiny LLM Week 1](https://skyzh.github.io/tiny-llm/) — matrix-only LLM, closest tutorial
- `docs/concepts/qwen2.5-7b-architecture.md` — full model spec, weight layout, tokenizer
- `docs/concepts/transformer_math_reference.md` — all formulas, FLOPs, memory math

---

## Phase 2 — KV Cache + Memory Management

**Goal**: Eliminate redundant computation in the decode phase. Then manage cache memory intelligently.

### What to Build

1. **Basic KV cache** — store K, V tensors from previous positions; during decode, only compute attention for the new token, reusing cached K/V
2. **Measure speedup** — tokens/sec before vs after caching
3. **Slot-based cache manager** — pre-allocate a pool of KV slots, assign/release per sequence
4. **Paged KV cache** — fixed-size blocks (e.g., 16 tokens/block), block table maps logical→physical, like OS virtual memory
5. **Cache eviction** — sliding window + sink tokens (StreamingLLM approach)

### The Memory Math

```
KV cache per token per layer:
  2 (K and V) × 4 (KV heads) × 128 (head dim) × 2 (bf16) = 2,048 bytes = 2 KB

KV cache per token (all 28 layers): 56 KB

Context     BF16 KV cache    FP8 KV cache
512         28 MB            14 MB
4,096       224 MB           112 MB
8,192       448 MB           224 MB
32,768      1.75 GB          896 MB
131,072     7.0 GB           3.5 GB
```

Note: Qwen2.5's 4 KV heads use **50% less** KV cache than Llama's 8 KV heads.

### PagedAttention

Invented by vLLM (SOSP 2023). Borrows from OS virtual memory:

```
Physical KV blocks:  [Block 0][Block 1][Block 2]...[Block N]  (pre-allocated pool)
Block table (per seq): logical_block → physical_block
Block size: typically 16 tokens

Allocation: grab free block when sequence grows past current block
Deallocation: return block to free list when sequence finishes
Copy-on-Write: shared prefixes share physical blocks until modified
```

Reduces KV cache waste from 60-80% to <4%.

### Sliding Window + Sink Tokens (StreamingLLM)

For sequences exceeding cache capacity:

```
KV cache layout = [sink_0, sink_1, ..., sink_{k-1}] + [recent_{t-w}, ..., recent_{t-1}]
                  |--- k sink tokens (keep forever) ---|--- w sliding window tokens ---|
```

- Keep first k=4 tokens ("attention sinks") that stabilize softmax
- Keep the most recent w tokens in a sliding window
- Position encoding: sinks keep original positions; window uses sequential positions
- Enables stable inference on **4M+ tokens** with constant memory

### Concepts to Learn

- Why decode is memory-bandwidth-bound (you read all weights but only compute for 1 token)
- Memory fragmentation: contiguous allocation wastes VRAM on short sequences
- PagedAttention: block table indirection, CoW sharing, prefix caching groundwork
- Prefill is compute-bound (large GEMM), decode is bandwidth-bound (mat-vec)
- The attention sink phenomenon (ICLR 2024)

### Benchmark Target

- Tokens/sec with and without KV cache (expect 10-50x speedup from caching)
- VRAM usage: peak and steady-state
- Profile memory allocation patterns

### References

- [PagedAttention paper (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [StreamingLLM: Attention Sinks (ICLR 2024)](https://arxiv.org/abs/2309.17453)
- `docs/concepts/attention_mechanisms_2024_2026.md` — §4 PagedAttention, §5 StreamingLLM

---

## Phase 3 — Custom GPU Kernels

**Goal**: Replace PyTorch ops with hand-written Triton kernels. Understand what the GPU actually does.

### What to Build

Each kernel: naive version → optimized version(s) → benchmark → correctness test.

| # | Kernel | Fuses | Why It Matters |
|---|--------|-------|---------------|
| 1 | **RMSNorm + residual** | norm + add + scale | Eliminates 1 read + 1 write of `[B,S,H]` |
| 2 | **RoPE** | rotation of Q/K pairs | Memory-bound, benefits from fusion with projections |
| 3 | **Fused QKV projection** | 3 matmuls → 1 | Eliminates 2 intermediate `[B,S,H]` writes |
| 4 | **Matrix-vector multiply** | The core decode op | Must max out bandwidth utilization |
| 5 | **SiLU + gate multiply** | `silu(gate) * up` | Eliminates 1 intermediate `[B,S,I]` write |
| 6 | **Softmax** | numerically stable, fused | Online softmax (Milakov & Gimelshein, 2018) |

### RTX 5080 Architecture Details

```
SMs:            84                    CUDA cores/SM:   128
Tensor Cores:   336 (4/SM, 5th gen)  TMEM:           256 KB/SM (new in Blackwell)
L1/SMEM:        228 KB/SM            L2 Cache:       64 MB
Registers:      256 KB/SM (65,536)   VRAM:           16 GB GDDR7, 960 GB/s
FP16 Tensor:    ~112 TFLOPS          FP8 Tensor:     ~900 TOPS
INT8 Tensor:    1,801 TOPS (sparse)  FP4 Tensor:     ~1,801 TOPS
```

### Roofline Analysis

```
Ridge point (FP16): 112 TFLOPS / 960 GB/s = ~117 FLOPs/byte
  → If kernel AI < 117: memory-bound
  → If kernel AI > 117: compute-bound

Operation                Phase      AI (bf16, B=1)    Bound
──────────────────────────────────────────────────────────────
QKV/FFN projection       Prefill    ~S FLOPs/byte     Compute if S ≥ 117
QKV/FFN projection       Decode     ~1 FLOPs/byte     Memory-bound
RMSNorm                  Both       ~0.75 FLOPs/byte  Memory-bound
RoPE                     Both       ~1.5 FLOPs/byte   Memory-bound
SiLU + gate              Both       ~0.375 FLOPs/byte Memory-bound
Attention (QK^T)         Prefill    ~S/2 FLOPs/byte   Compute if S ≥ 234

Theoretical max decode speed (B=1):
  BF16 weights: 960 GB/s / 15.2 GB = ~63 tok/s
  INT8 weights: 960 GB/s / 7.6 GB  = ~126 tok/s
  FP4 weights:  960 GB/s / 3.8 GB  = ~252 tok/s
```

### Triton 3.x Key Features

- **Block pointers** (`tl.make_block_ptr`): declarative tile API, handles bounds/coalescing
- **Persistent kernels**: launch one CTA per SM (84 CTAs), software tile scheduler for L2 cache reuse, up to 1.5x over non-persistent
- **Warp specialization** (Triton 3.2+): automatic producer/consumer warp groups — producer warps handle TMA loads, consumer warps do Tensor Core compute. 10-15% gains on attention and FP8 GEMM.
- **Autotuning**: `@triton.autotune` with `configs` for block sizes, num_warps, num_stages

### Memory Coalescing Rules

- Thread i should access element i in contiguous memory (128 bytes per warp transaction)
- For weight matrix `W[M,N]` stored row-major: threads access along N (columns within a row)
- Transposed access: load coalesced into shared memory, then read transposed from SMEM
- Use vectorized loads (`float4`, or Triton with block sizes that are multiples of 128 bytes)

### Concepts to Learn

- GPU execution model: warps (32 threads), blocks (multiple warps), grid (all blocks)
- Memory hierarchy: registers → TMEM → L1/SMEM → L2 → GDDR7
- Kernel fusion: fewer launches = fewer intermediate memory writes = faster
- Bandwidth utilization: what % of 960 GB/s are you achieving?
- Triton programming: `tl.load`, `tl.store`, `tl.dot`, program IDs, block pointers
- Roofline model: is your kernel compute-bound or memory-bound?
- **Megakernel approach** (Stanford, 2025): fuse entire forward pass into one persistent kernel. Achieved 78% bandwidth utilization (vs ~50% for vLLM/SGLang). Advanced stretch goal.

### Benchmark Target

For each kernel: report TFLOPS, GB/s, % of peak theoretical bandwidth.

### References

- [yalm: Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html)
- [llm.c dev/cuda kernels](https://github.com/karpathy/llm.c/blob/master/dev/cuda/README.md)
- [From 11% to 88% Peak Bandwidth: Triton Kernels](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [PyTorch Blog: Warp Specialization in Triton](https://pytorch.org/blog/warp-specialization/)
- [Look Ma, No Bubbles! (Stanford Megakernel)](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- `docs/concepts/transformer_math_reference.md` — §8 Roofline analysis
- [NVIDIA RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

---

## Phase 4 — Quantization

**Goal**: Shrink the model to fit more efficiently in 16GB, run faster, and leverage Blackwell-exclusive FP4.

### What to Build

| # | Technique | Bits | Type | Quality Loss | Speed | Complexity |
|---|-----------|------|------|-------------|-------|------------|
| 1 | **INT8 absmax** | 8 | Weight-only | ~0% | ~2x bandwidth | Low |
| 2 | **FP8 E4M3** | 8 | Weight+Activation | ~0% | ~2x, native TC | Low |
| 3 | **NVFP4** | 4 | Weight-only | ~1% | ~4x, native TC | Medium-High |
| 4 | **AWQ** | 4 | Weight-only | ~2-3% PPL | ~4x (Marlin) | Low-Medium |
| 5 | **SmoothQuant** | 8 | W8A8 | ~0.5% | INT8 TC matmuls | Medium |
| 6 | **Mixed-precision** | 4-8 | Hybrid | Optimized | Best trade-off | Medium-High |

### FP8 (E4M3) — Native Blackwell

```
E4M3: 4 exponent bits, 3 mantissa bits. Range: ±448.
Quantize:  Q(x) = clamp(round(x / scale), min_fp8, max_fp8)
Scale:     scale = amax(tensor) / max_fp8_value

Scaling strategies (increasing accuracy):
  Per-tensor:  1 FP32 scale for entire tensor
  Per-channel: 1 FP32 scale per output row
  Per-block (MXFP8): 1 E8M0 scale per 32 elements — Blackwell hardware-accelerated
```

Near-zero perplexity loss. ~33% tokens/sec improvement. This is the recommended baseline.

### NVFP4 — Blackwell-Exclusive

```
E2M1: 1 sign, 2 exponent, 1 mantissa = 16 representable values.
Values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

Dual-level scaling:
  Level 1: FP8 E4M3 scale per 16-element block (fine-grained)
  Level 2: FP32 scale per tensor (global normalization)

  x_reconstructed = x_fp4 × s_block_fp8 × s_tensor_fp32
```

3.5x memory reduction vs FP16. Only ~1% accuracy degradation. Software maturity is still evolving on consumer GPUs as of early 2026.

### AWQ — Activation-Aware Weight Quantization

```
Core insight: protect 1% of salient weight channels identified by ACTIVATION magnitude.
Transform:  Y = (X × diag(s⁻¹)) × (diag(s) × W)   [mathematically equivalent to Y = XW]
  s_j scales up important weight channels → less relative rounding error
  s_j⁻¹ scales down corresponding activations → compensates

Slightly better accuracy than GPTQ (6.84 vs 6.90 Wiki2 PPL), much faster to quantize.
```

### SmoothQuant — Enable W8A8 (Both INT8)

```
Problem: activations have outlier channels (100x larger than typical)
Solution: migrate difficulty from activations to weights

  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)    [α=0.5 default]
  X_smooth = X × diag(s)⁻¹     (outliers shrunk)
  W_smooth = diag(s) × W        (weights absorb difficulty)

Enables INT8×INT8 Tensor Core matmuls — 2x faster than FP16.
```

### Mixed-Precision Recipe for RTX 5080

```
Conservative (minimal quality loss):
  Embedding/LM Head:    FP16
  Attention (QKV+O):    FP8
  FFN/MLP:              INT4 (AWQ, group_size=128)
  LayerNorm:            FP16
  Memory: ~30% of FP16

Aggressive (Blackwell-optimized):
  Embedding/LM Head:    FP8
  Attention:            FP8
  FFN/MLP:              NVFP4
  LayerNorm:            FP16
  Memory: ~20% of FP16
```

### Calibration

- **128-512 samples**, 512-2048 token length
- Datasets: C4 (general web), WikiText-2 (benchmark), or domain-specific
- Methods: MinMax (simplest), Percentile 99.99% (robust), MSE-based (most accurate)
- Diversity matters more than quantity

### Benchmark Target

For each quantization scheme:
- Model size reduction (GB)
- Tokens/sec improvement
- Perplexity on WikiText-2
- VRAM usage (peak + steady-state)

Comprehensive comparison table across all formats on the RTX 5080.

### References

- `docs/concepts/quantization_techniques.md` — full research on all methods
- [NVIDIA: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [AWQ: MLSys 2024 Best Paper](https://arxiv.org/abs/2306.00978)
- [SmoothQuant (ICML 2023)](https://arxiv.org/abs/2211.10438)
- [GPTQ (ICLR 2023)](https://arxiv.org/abs/2210.17323)

---

## Phase 5 — Serving: Continuous Batching + HTTP

**Goal**: Turn the engine into a server handling concurrent requests efficiently.

### What to Build

1. **Request queue** — async request handling with FastAPI/Starlette
2. **Continuous batching scheduler** — iteration-level scheduling (from the Orca paper)
3. **Token streaming** — Server-Sent Events (SSE) for real-time token delivery
4. **Sequence management** — track active sequences, stop conditions, max_tokens
5. **Prefix caching** — detect shared prefixes, reuse their KV cache
6. **OpenAI-compatible API** — `/v1/chat/completions` endpoint

### Continuous Batching Algorithm

```python
while requests_pending or active_sequences:
    # 1. Remove finished sequences
    for seq in active_sequences:
        if seq.is_done():  # EOS or max_tokens
            release_kv_cache(seq)
            return_result(seq)
            active_sequences.remove(seq)

    # 2. Add new requests up to capacity
    while requests_pending and can_add_to_batch():
        req = requests_pending.pop()
        seq = prefill(req)  # or chunked prefill
        active_sequences.add(seq)

    # 3. Run one decode step for all active sequences
    next_tokens = decode_step(active_sequences)

    # 4. Update sequences
    for seq, token in zip(active_sequences, next_tokens):
        seq.append(token)
```

Achieves 2-3x throughput over static batching.

### Prefix Caching — Two Approaches

**Hash-based (vLLM style)**:
- Divide tokens into blocks of B tokens (e.g., 16)
- Hash each block: `hash(block_tokens, previous_block_hash)` (chained for position sensitivity)
- Global hash table maps hash → physical KV block
- O(1) lookup per block

**Radix tree (SGLang style)**:
- Token-level radix tree (compressed trie)
- Edges = token sequences, nodes = KV block references
- Can match prefixes at any token boundary (no block alignment needed)
- Better for complex multi-call workloads (agents, tree-of-thought)

### CUDA Graphs for Batch-1 Decode

Capture the decode loop as a CUDA Graph to eliminate per-kernel CPU launch overhead:
- ~100+ kernel launches per decode step, each costs 2-5μs CPU-side
- Without CUDA Graphs: CPU becomes the bottleneck at B=1
- With CUDA Graphs: **2.3x speedup** on batch-1 decode (LLaMA-7B benchmark)
- Capture graphs for padded batch sizes (1, 2, 4, 8, 16...) to handle dynamic shapes

### Scheduling & Preemption

| Policy | When to Use |
|--------|------------|
| **FCFS** | Default, simple, fair |
| **Priority-based** | Multi-tenant (paid vs free) |
| **SLO-aware** | Dynamic priority based on deadline proximity |
| **Preempt via recompute** | Higher-priority request arrives, discard KV and re-prefill later |

### Concepts to Learn

- Why naive batching wastes GPU (sequences finish at different times)
- Iteration-level scheduling: compose batch differently each step
- Chunked prefill: prevent long prefills from blocking decode latency
- Prefix caching: hash-based block matching, radix tree lookup
- Backpressure: what happens when requests arrive faster than you can serve

### Benchmark Target

Throughput (tok/s) at batch sizes 1, 4, 8, 16, 32.
Latency distribution: P50, P95, P99.
Compare against vLLM on same model/hardware.

### References

- [SimpleLLM](https://github.com/naklecha/simple-llm) — 950-line serving engine with continuous batching
- [HuggingFace: Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching)
- [vLLM V1 Blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [SGLang: RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)

---

## Phase 6 — Speculative Decoding

**Goal**: Use a small draft model to generate candidate tokens, verify in parallel with the target.

### What to Build

| # | Method | Training? | Extra Memory | Speedup | Complexity |
|---|--------|-----------|-------------|---------|------------|
| 1 | **Standard speculative** (Qwen2.5-0.5B draft) | No | ~1GB | 2-3x | Medium |
| 2 | **SWIFT self-speculative** | No | None | 1.3-1.6x | Low |
| 3 | **EAGLE with trained head** | Yes | ~50-100MB | 3-5x | High |
| 4 | **Lookahead decoding** (Jacobi) | No | None | 1.5-2.3x | Medium-High |

### The Core Algorithm (Standard Speculative Decoding)

```
Draft model q(x) generates K candidate tokens: x_1, ..., x_K
Target model p(x) verifies all K in ONE forward pass

For each token x_t in order:
  acceptance_probability = min(1, p(x_t) / q(x_t))
  if random() < acceptance_probability:
    ACCEPT → keep token, continue
  else:
    REJECT → discard x_t and all subsequent tokens
    resample from: p'(x) = normalize(max(0, p(x) - q(x)))
    break
```

**Lossless**: this exactly recovers the target model's distribution (mathematical proof in the paper).

Expected accepted tokens with draft length K:
```
E[accepted] = (1 - α^(K+1)) / (1 - α)    where α = 1 - TV(p, q)
```

### EAGLE — Feature-Level Drafting (Fastest)

Instead of a separate draft model, EAGLE trains a lightweight auto-regression head (~50M params) that predicts the next hidden state feature, then uses the frozen LM head to get token probabilities:

```
Draft: f_{t+1} = auto_reg_head(f_t, token_t)  →  LM_head(f_{t+1})  →  draft token
Tree verification: multiple candidates form a tree, verified in one forward pass
```

EAGLE-3 (NeurIPS 2025): 3.0-6.5x speedup, discovered scaling law for speculative decoding.

### Memory Budget (RTX 5080)

```
Qwen2.5-7B @ FP8:       7.6 GB
Qwen2.5-0.5B @ FP16:    1.0 GB
KV caches + activations: ~2-4 GB
                         ─────────
Total:                   ~11-13 GB  ✓ fits in 16 GB
```

### When Speculation Helps vs Hurts

- **Helps**: predictable text (code, JSON, structured data, formulaic language)
- **Hurts**: creative/diverse text where draft and target distributions diverge heavily
- Monitor acceptance rate: if α < 0.5, speculation may not help

### Concepts to Learn

- Why speculative decoding doesn't change the output distribution
- Rejection sampling and residual distributions
- When decode is bandwidth-bound, verification of K tokens ≈ cost of generating 1 token
- Tree-structured verification for multiple candidates per position
- Self-speculative decoding: using early exit from the same model

### Benchmark Target

Effective tokens/sec, acceptance rate, speedup factor vs standard decoding.

### References

- `docs/concepts/speculative-decoding-and-structured-generation.md` — full research
- [Leviathan et al. (2022)](https://arxiv.org/abs/2211.17192)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [Jay Mody: Speculative Sampling](https://jaykmody.com/blog/speculative-sampling/)
- [vLLM: Speculative Decoding Performance](https://blog.vllm.ai/2024/10/17/spec-decode.html)

---

## Phase 7 — Structured Output / Constrained Decoding

**Goal**: Force the model to produce valid JSON, SQL, or any grammar-defined format.

### What to Build

| # | Method | Expressiveness | Per-Token Overhead | Complexity |
|---|--------|---------------|-------------------|------------|
| 1 | **JSON parser state tracker** | JSON only | O(V) | Low |
| 2 | **Regex → FSM** (Outlines-style) | Regular languages | O(1) after precompute | Medium |
| 3 | **Jump-forward decoding** | Any grammar | Skips deterministic tokens | Low (add-on) |
| 4 | **PDA + CI/CD split** (XGrammar-style) | Full CFG | <40μs | High |

### How Constrained Decoding Works

At each generation step:
1. Determine what tokens are valid given the grammar state
2. Create a mask: valid tokens → keep, invalid → -∞
3. Apply mask to logits before softmax
4. Sample from masked distribution
5. Advance grammar state

### XGrammar — State of the Art (MLSys 2025)

The key innovation is splitting vocabulary tokens into two categories:

```
Context-Independent (~99%):  validity depends only on grammar position
  → Precompute bitmask per state in O(1) at runtime

Context-Dependent (~1%):     validity depends on the PDA stack (nesting depth)
  → Check dynamically at runtime

Result: <40μs per-token overhead, 100x faster than prior art
Default backend in both vLLM and SGLang as of 2025
```

### Jump-Forward Decoding

When only one valid token exists at the current grammar state, **skip the LLM entirely**:

```
Standard:     LLM → '{' → LLM → '"' → LLM → 'name' → LLM → '"' → ...
Jump-forward:       emit '{"'            → LLM → 'name' → emit '"' → ...

Can reduce LLM forward passes by 30-60% for JSON generation.
Constrained decoding becomes FASTER than unconstrained.
```

### Concepts to Learn

- FSM-based generation: states, transitions, token-to-transition mapping
- CFG-based generation: pushdown automata, stack-based parsing during generation
- The multi-byte token challenge: a single token like `": ["` spans multiple grammar transitions
- Why structured output matters: tool-calling, function-calling, agent frameworks

### References

- `docs/concepts/speculative-decoding-and-structured-generation.md` — Part 2
- [XGrammar Blog](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
- [Outlines GitHub](https://github.com/dottxt-ai/outlines)
- [LMSYS: Fast JSON Decoding with Compressed FSM](https://lmsys.org/blog/2024-02-05-compressed-fsm/)

---

## Phase 8 — Advanced Attention Mechanisms

**Goal**: Implement the attention variants that power production inference engines.

### What to Build

| # | Mechanism | Key Insight | Priority |
|---|-----------|------------|----------|
| 1 | **Tiled attention** with online softmax | Never materialize S×S attention matrix | Critical |
| 2 | **Flash-Decoding** (split-KV) | Parallelize decode across KV chunks | Critical |
| 3 | **Paged attention kernel** | Attention with non-contiguous KV blocks | Critical |
| 4 | **FlashAttention-style fused kernel** | Tiled, IO-aware, fits in SRAM | High |
| 5 | **FP8 attention** | Block quantization + incoherent processing | High |
| 6 | **Sliding window attention** | Limit attention to local window | Medium |
| 7 | **SpargeAttention** (stretch) | Training-free 4-7x sparse attention | Low |

### Online Softmax — The Core Primitive

Standard softmax requires 3 passes: find max, compute denominators, compute outputs.
Online softmax fuses max-finding with denominator accumulation in 2 passes:

```
Initialize: m = -∞, d = 0

For each tile of K:
  S_tile = Q × K_tile^T / √128
  m_new = max(m, rowmax(S_tile))
  correction = exp(m - m_new)
  d = d × correction + rowsum(exp(S_tile - m_new))
  O = O × correction + exp(S_tile - m_new) @ V_tile
  m = m_new

Final: output = O / d
```

This never materializes the full S×S attention matrix. Memory: O(S×D) instead of O(S²).

### Flash-Decoding (Split-KV) — Essential for Decode Phase

During decode (S_new=1), standard attention has insufficient parallelism. Flash-Decoding splits the KV cache across multiple thread blocks:

```
Phase 1 (parallel): each thread block processes a chunk of KV cache
  O_c, logsumexp_c = attention(Q, K_chunk_c, V_chunk_c)

Phase 2 (reduction): combine partial results
  O = Σ_c [exp(logsumexp_c - m) × O_c] / Σ_c [exp(logsumexp_c - m)]
```

Up to **8x speedup** for long-context decode at batch=1.

### FlashAttention Evolution

```
FA1 (2022): Tiled, IO-aware, online softmax                      ~50% peak
FA2 (2023): Better parallelism, work partitioning                 ~70% peak
FA3 (2024): Warp specialization, ping-pong, FP8 (Hopper)        ~75% peak
FA4 (2025): 5-stage pipeline, TMEM, tcgen05.mma (Blackwell)     >1 PFLOPS
```

FA4 is designed for your RTX 5080's architecture (SM 10.x).

### Newer Attention Variants (2025-2026)

- **MLA (DeepSeek)**: Compress K+V into low-rank latent. 93% KV cache reduction. "Absorb trick" avoids decompression at inference.
- **Differential Attention (Microsoft, ICLR 2025)**: Subtract two attention maps to cancel noise. Reduces hallucinations.
- **SpargeAttention (ICML 2025)**: Training-free 4-7x speedup via two-stage online block filtering.
- **NSA (DeepSeek, ACL 2025 Best Paper)**: Hardware-aligned trainable sparse attention.

### Benchmark Target

TFLOPS, memory usage, latency vs naive attention at sequence lengths: 512, 2K, 8K, 32K.

### References

- `docs/concepts/attention_mechanisms_2024_2026.md` — comprehensive survey
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Flash-Decoding blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [FlashInfer (MLSys 2025 Best Paper)](https://arxiv.org/abs/2501.01005)

---

## Phase 9 — Profiling + Benchmarking

**Goal**: Measure everything. Build a benchmarking harness used throughout all phases.

### What to Build

1. **Kernel-level profiling** — `torch.cuda.Event`, Nsight Systems, Nsight Compute
2. **End-to-end benchmark suite** — TTFT, ITL, TPS, throughput, VRAM usage
3. **Roofline analysis** — plot kernels on roofline chart
4. **Comparison framework** — benchmark vs vLLM, llama.cpp, TensorRT-LLM
5. **Regression tracking** — track performance across phases

### Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **TTFT** (Time to First Token) | Prefill latency | Lower is better |
| **ITL** (Inter-Token Latency) | Time between consecutive decode tokens | Lower is better |
| **TPS** (Tokens Per Second) | Per-request generation speed | Higher is better |
| **Throughput** | System-wide tok/s under load | Higher is better |
| **VRAM** | Peak and steady-state memory | Monitor for leaks |
| **GPU utilization** | SM occupancy, bandwidth utilization | % of theoretical |

### Benchmarking Protocol

- Median of 100 runs with 10 warmup runs
- Use `torch.cuda.Event` for GPU timing (not wall clock)
- Always include "RTX 5080, GDDR7" in results
- Save results as JSON in `benchmarks/results/`
- Every optimization must have before/after measurement

### References

- [NVIDIA: LLM Benchmarking Fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [BentoML: Key Metrics for LLM Inference](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)

---

## Phase 10 — Advanced Topics

Stretch goals once the core engine is solid:

1. **CUDA Graphs** — capture decode loop, eliminate kernel launch overhead (~2.3x at B=1)
2. **Megakernel** — fuse entire forward pass into one persistent kernel (Stanford, 2025)
3. **Mixture-of-Experts serving** — expert routing, load balancing
4. **Disaggregated prefill/decode** — separate GPU pools for each phase
5. **Tensor parallelism** — split across 2+ GPUs
6. **LoRA serving** — serve multiple LoRA adapters from one base model
7. **Vision-Language Model** — extend for multimodal inputs
8. **Linear attention hybrids** — Gated DeltaNet (ICLR 2025), used in Qwen3-Next

---

## Key References

### Books / Handbooks
- [Baseten: Inference Engineering](https://www.baseten.com/inference-engineering/) — full-stack, CUDA to Kubernetes
- [BentoML: LLM Inference Handbook](https://bentoml.com/llm/inference-optimization) — 12 optimization techniques

### Build-From-Scratch Projects
- [Tiny LLM](https://skyzh.github.io/tiny-llm/) — matrix-only LLM serving in a week
- [yalm (Andrew Chan)](https://andrewkchan.dev/posts/yalm.html) — C++/CUDA inference engine, excellent optimization narrative
- [SimpleLLM](https://github.com/naklecha/simple-llm) — 950-line serving engine with continuous batching
- [llm.c](https://github.com/karpathy/llm.c) — Karpathy's C/CUDA LLM, `dev/cuda/` for kernel patterns

### Technical Blogs
- [NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [NVIDIA: LLM Benchmarking Fundamentals](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)
- [Meta: Scaling LLM Inference (TP, CP, EP)](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [From 11% to 88% Peak Bandwidth: Triton Kernels](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)

### Key Papers
- FlashAttention 1-4: Tri Dao et al. — IO-aware exact attention
- PagedAttention: vLLM (SOSP 2023) — virtual memory for KV cache
- XGrammar (MLSys 2025) — efficient structured generation
- Speculative Decoding: Leviathan et al. (2022) — draft-then-verify
- AWQ (MLSys 2024 Best Paper) — activation-aware weight quantization
- EAGLE-3 (NeurIPS 2025) — feature-level speculative decoding scaling law
- SpargeAttention (ICML 2025) — training-free sparse attention
- FlashInfer (MLSys 2025 Best Paper) — composable attention engine

### Tools
- [vLLM](https://github.com/vllm-project/vllm) — study source, benchmark against
- [SGLang](https://github.com/sgl-project/sglang) — RadixAttention, structured generation
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — composable attention kernels
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) / [Nsight Compute](https://developer.nvidia.com/nsight-compute) — GPU profiling

### Concept Deep-Dives (in `docs/concepts/`)
- `qwen2.5-7b-architecture.md` — model spec, weights, tokenizer, memory math
- `transformer_math_reference.md` — formulas, FLOPs, roofline analysis
- `attention_mechanisms_2024_2026.md` — FA3/4, Flash-Decoding, PagedAttention, MLA
- `quantization_techniques.md` — FP8, FP4, GPTQ, AWQ, SmoothQuant, QuIP#
- `speculative-decoding-and-structured-generation.md` — all speculative methods, XGrammar, Outlines
