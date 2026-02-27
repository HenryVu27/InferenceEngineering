# Attention Mechanisms for LLM Inference (2024-2026)

Research survey compiled for the Inference Engineering project.
Hardware context: NVIDIA RTX 5080 (Blackwell, 16GB GDDR7, 960 GB/s, SM 10.x).

---

## Table of Contents

1. [FlashAttention-3](#1-flashattention-3)
2. [FlashAttention-4](#2-flashattention-4)
3. [Flash-Decoding / Flash-Decoding++](#3-flash-decoding--flash-decoding)
4. [PagedAttention (v1 and v2 kernels)](#4-pagedattention-v1-and-v2-kernels)
5. [Sliding Window + Sink Tokens (StreamingLLM)](#5-sliding-window--sink-tokens-streamingllm)
6. [Ring Attention / Sequence Parallelism](#6-ring-attention--sequence-parallelism)
7. [Multi-Head Latent Attention (MLA)](#7-multi-head-latent-attention-mla)
8. [New Attention Variants (2025-2026)](#8-new-attention-variants-2025-2026)
9. [Relevance Matrix for RTX 5080](#9-relevance-matrix-for-rtx-5080)

---

## 1. FlashAttention-3

**Paper**: Tri Dao, Jay Shah. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)

### Core Technical Insight

FlashAttention-2 was already IO-aware, tiling Q, K, V to fit in SRAM and using online softmax. But on Hopper (H100) GPUs, FA2 only reached ~35% of theoretical peak FLOPS because it could not exploit Hopper's new async execution model. FA3 closes this gap with three techniques:

1. **Warp specialization with producer-consumer asynchrony**
2. **Interleaved GEMM-softmax pipelining (ping-pong scheduling)**
3. **FP8 attention with block quantization and incoherent processing**

### Technique 1: Warp Specialization

Hopper introduces hardware support for asynchronous data movement via TMA (Tensor Memory Accelerator) and asynchronous matrix multiply via WGMMA (Warp-Group Matrix Multiply-Accumulate). FA3 exploits this by dividing warps within a CTA (Cooperative Thread Array) into distinct roles:

- **Producer warps**: Issue TMA loads to asynchronously move K, V tiles from HBM to shared memory. These warps never compute.
- **Consumer warps**: Execute WGMMA instructions on Tensor Cores for the GEMM operations (S = QK^T and O = PV). These warps never issue loads.

This separation allows the compiler to generate near-optimal instruction schedules for each role, rather than interleaving load and compute instructions within the same warp (which causes stalls).

### Technique 2: Ping-Pong Scheduling

Within the main loop, FA3 overlaps the softmax computation of iteration `j` with the second WGMMA of iteration `j+1`. Two warp groups alternate:

```
Warp Group 0:  [GEMM_0(S=QK^T)] [softmax + GEMM_1(O=PV)] [GEMM_0(S=QK^T)] ...
Warp Group 1:  ...idle...         [GEMM_0(S=QK^T)]          [softmax + GEMM_1(O=PV)] ...
```

While one warp group runs the (relatively slow) non-GEMM softmax operations, the other warp group runs the next GEMM on the Tensor Cores. This hides the softmax latency almost entirely.

### Technique 3: FP8 with Block Quantization and Incoherent Processing

Hopper has native FP8 Tensor Cores (E4M3 and E5M2 formats). Naive per-tensor FP8 quantization of Q, K, V loses significant accuracy due to outlier values. FA3 introduces two mitigations:

**Block quantization**: Q, K, V are partitioned into small blocks, and per-block scaling factors are computed. This localizes the impact of outliers to their block rather than the entire tensor.

**Incoherent processing**: Before quantization, Q and K are pre-multiplied by a random orthogonal matrix M (implemented as a diagonal random sign matrix followed by a Hadamard transform):

```
Q' = Q * M
K' = K * M
```

Since MM^T = I, the attention output is preserved: `softmax(Q'K'^T / sqrt(d)) = softmax(QMM^TK^T / sqrt(d)) = softmax(QK^T / sqrt(d))`. But the Hadamard transform disperses outlier values across all dimensions, making the distribution more uniform and friendlier to quantization.

The Hadamard transform runs in O(d log d) per head and is memory-bandwidth-bound, so it can be fused with preceding operations (like RoPE) for effectively zero overhead.

**Accuracy result**: FP8 FA3 with block quantization + incoherent processing achieves 2.6x lower numerical error than baseline FP8 attention (per-tensor quantization).

### Performance Numbers (H100)

| Metric | FA2 | FA3 (FP16/BF16) | FA3 (FP8) |
|--------|-----|------------------|-----------|
| TFLOPS | ~350 | ~740 | ~1,200 |
| % of peak | ~35% | ~75% | - |
| Speedup vs FA2 | 1.0x | 1.5-2.0x | ~3.0x |

### Key Formula (unchanged from FA1/FA2)

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

The algorithm computes this tile-by-tile using online softmax (Milakov & Gimelshein, 2018):

```
For each tile j of K, V:
    S_j = Q_tile * K_j^T                    # local attention scores
    m_new = max(m_old, rowmax(S_j))          # running max
    P_j = exp(S_j - m_new)                  # local probabilities
    l_new = exp(m_old - m_new) * l_old + rowsum(P_j)   # running sum
    O = (exp(m_old - m_new) * O) + P_j * V_j            # rescale and accumulate
```

### Implementation Complexity

**High**. Requires Hopper-specific PTX instructions (WGMMA, TMA, async barriers). The warp specialization and ping-pong scheduling are very architecture-specific. Not trivially portable to other architectures.

### RTX 5080 Relevance

**Partially relevant.** The RTX 5080 is Blackwell (SM 10.x), not Hopper (SM 9.0). FA3 was designed for Hopper. The async concepts (TMA, WGMMA) evolve further on Blackwell with 5th-gen Tensor Cores. FA3 code would need adaptation for SM 10.x; FA4 (below) is the Blackwell-native version. However, the FP8 block quantization and incoherent processing techniques are architecture-agnostic algorithmic improvements that are directly applicable.

---

## 2. FlashAttention-4

**Paper/Presentation**: Tri Dao. Preliminary results presented at Hot Chips 2025. No formal paper yet. [Blog: Modal reverse-engineering](https://modal.com/blog/reverse-engineer-flash-attention-4)

### Core Technical Insight

FA4 is the Blackwell-native successor to FA3. It is the first attention kernel to break the petaflop barrier (>1 PFLOPS). The key advance is a deeper pipeline exploiting Blackwell's new tensor memory hierarchy and 5th-gen Tensor Cores.

### Architecture Changes from FA3

FA3 used a 2-stage "ping-pong" pipeline (alternating between 2 warp groups). FA4 expands to a ~5-stage pipeline:

```
Stage 1: Load (TMA from HBM to shared memory)
Stage 2: Compute S = QK^T (GEMM via tcgen05.mma)
Stage 3: Softmax + normalize
Stage 4: Rescale + reduce (online softmax correction)
Stage 5: Compute O = PV + store (GEMM + TMA writeback)
```

The `tcgen05.mma` instruction is Blackwell's 5th-generation Tensor Core operation, replacing Hopper's WGMMA.

### Performance Numbers (B200/B100)

| Metric | FA3 (H100) | FA4 (B200) |
|--------|-----------|-----------|
| TFLOPS (BF16) | ~740 | >1,000 (1+ PFLOPS) |
| Speedup vs FA3 | 1.0x | ~2.0x |
| Speedup vs FA1 | ~7.5x | ~15x |
| vs cuDNN attention | - | ~1.2x faster |

### Current Limitations

- **BF16 only** (no FP8 support yet -- expected in future releases)
- **Forward pass only** (no backward pass yet, so inference-only for now)
- No public paper yet; implementation details reverse-engineered by Modal

### Implementation Complexity

**Very high**. Requires Blackwell-specific PTX (tcgen05.mma), multi-buffered pipeline design, programmer-managed tensor memory. Significantly more complex than FA3.

### RTX 5080 Relevance

**Directly relevant.** The RTX 5080 is SM 10.x (Blackwell). FA4 is designed for exactly this architecture. However, FA4 is currently tuned for data center GPUs (B200/B100). The RTX 5080 has less shared memory and fewer SMs than B200, so kernel tuning (tile sizes, pipeline depth) would differ. Once FA4 is released for consumer Blackwell, it will be the optimal attention kernel for the RTX 5080.

---

## 3. Flash-Decoding / Flash-Decoding++

### Flash-Decoding (2023)

**Source**: Stanford CRFM. Tri Dao, Daniel Haziza, et al. [Blog post](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (October 2023).

#### Core Technical Insight

FlashAttention parallelizes over the batch and head dimensions. During the **prefill** phase (many query tokens), this provides ample parallelism. But during the **decode** phase (single query token per sequence), there is only 1 query position per sequence -- so FA's parallelism over the sequence dimension of Q is wasted.

Flash-Decoding adds a new parallelism dimension: **split-KV**. Instead of one thread block processing the entire KV cache for a given (batch, head), the KV cache is split into chunks, and multiple thread blocks process different chunks in parallel.

#### Algorithm

```
Phase 1 (parallel across KV chunks):
    For each chunk c of the KV cache:
        Load Q (same for all chunks)
        Load K_c, V_c (this chunk's keys and values)
        Compute local attention: O_c = softmax(Q * K_c^T / sqrt(d)) * V_c
        Store: O_c, logsumexp_c (local softmax statistics)

Phase 2 (reduction):
    Combine all (O_c, logsumexp_c) using the log-sum-exp trick:
        m = max(logsumexp_c for all c)
        O = sum_c [ exp(logsumexp_c - m) * O_c ] / sum_c [ exp(logsumexp_c - m) ]
```

The reduction is a lightweight separate kernel. The key is that Phase 1 now has `num_chunks` parallel thread blocks per (batch, head), fully saturating the GPU even with batch_size=1.

#### Performance

- Up to **8x speedup** for long-context decode (e.g., 128K context, batch=1) on H100
- No speedup for short contexts (parallelism already sufficient)
- Negligible accuracy difference (exact attention, just reordered computation)

### Flash-Decoding++ (MLSys 2024)

**Paper**: Ke Hong et al. "FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics." MLSys 2024. [Paper](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html)

#### Core Technical Insight

Flash-Decoding++ addresses three remaining bottlenecks:

1. **Unified max value for softmax**: The split-KV approach requires a synchronization barrier to combine partial softmax results. FD++ introduces a technique that pre-estimates a unified maximum value, allowing partial softmax computations to proceed asynchronously without synchronization, eliminating ~20% overhead from the attention computation.

2. **Flat GEMM optimization**: During decode, the GEMM shapes are "flat" (1 x N x K), causing poor Tensor Core utilization. FD++ uses double-buffered flat GEMM with optimized tiling, achieving up to 52% speedup on these shapes.

3. **Heuristic dataflow**: Adaptively selects kernel configurations based on input shapes and hardware, rather than using a single static configuration.

#### Performance

- Up to **2.02x speedup** over Flash-Decoding for long-context decode
- Up to **4.86x speedup** over HuggingFace on NVIDIA GPUs
- Also works on AMD GPUs (up to **2.18x** over HuggingFace)

### Implementation Complexity

**Medium**. Flash-Decoding is straightforward: split the KV loop across thread blocks, add a reduction kernel. The reduction uses standard log-sum-exp combination. FD++ is more complex (async softmax, heuristic dispatch).

### RTX 5080 Relevance

**Highly relevant.** Flash-Decoding is essential for any single-GPU inference engine. During decode (the bottleneck phase), the RTX 5080 with its 10,752 CUDA cores needs sufficient parallelism. Split-KV is the standard technique used by vLLM, SGLang, and FlashInfer. You will implement this in Phase 8.

---

## 4. PagedAttention (v1 and v2 Kernels)

**Paper**: Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

### Core Technical Insight

Traditional KV caches allocate contiguous memory for the maximum possible sequence length. This wastes 60-80% of GPU memory due to:
- Pre-allocation for max_seq_len even when actual sequences are short
- Internal fragmentation from per-sequence contiguous blocks
- Inability to share memory across requests

PagedAttention borrows from OS virtual memory: the KV cache is divided into fixed-size **blocks** (pages), and a **block table** (page table) maps logical sequence positions to physical memory blocks. Blocks can be non-contiguous in physical memory.

### Algorithm

```
For each attention head h, sequence s:
    Q = query[s, h]                    # [1, d] for decode
    block_table = get_block_table(s)   # maps logical block idx -> physical block ptr

    For each logical block b in block_table:
        physical_block = block_table[b]
        K_b = kv_cache_k[physical_block]   # [block_size, d]
        V_b = kv_cache_v[physical_block]   # [block_size, d]
        score_b = Q * K_b^T / sqrt(d)
        accumulate with online softmax
```

### v1 vs v2 Kernels in vLLM

**v1 kernel**: One thread block handles one (head, sequence) pair. It iterates over all KV blocks sequentially. For short sequences this is fine, but for long sequences the single thread block becomes a bottleneck (same problem Flash-Decoding solves).

**v2 kernel**: Introduces **partitioning** along the KV sequence dimension (analogous to Flash-Decoding's split-KV). The grid becomes `(num_heads, num_seqs, max_num_partitions)`. Multiple thread blocks process different partitions of the KV cache for the same (head, sequence), then a **separate reduction kernel** combines partial results using the log-sum-exp trick.

```
v1 grid: (num_heads, num_seqs, 1)
v2 grid: (num_heads, num_seqs, num_partitions)
```

The v2 kernel automatically falls back to v1 behavior when `num_partitions = 1` (short sequences).

### Memory Efficiency

| System | KV Cache Waste |
|--------|---------------|
| FasterTransformer | 60-80% |
| Orca | ~50% |
| vLLM (PagedAttention) | <4% |

vLLM achieves 2-4x throughput improvement over existing systems at the same latency.

### Key Features

- **Copy-on-write (CoW)**: Shared prefixes (e.g., system prompts) share physical blocks across sequences. Only copied when modified.
- **Prefix caching**: Common prefixes (system prompts, few-shot examples) are hashed and reused across requests via a radix tree lookup.
- **Block size trade-off**: Larger blocks = more parallelism per block but more internal fragmentation. Typical: 16 tokens per block.

### Implementation Complexity

**Medium-high**. The core attention kernel with block table indirection is moderate. The memory manager (allocator, CoW, prefix caching) adds significant systems complexity.

### RTX 5080 Relevance

**Highly relevant.** With only 16GB VRAM, efficient memory management is critical. PagedAttention is the standard approach for any serving engine. You will implement a paged KV cache in Phase 2 and paged attention kernels in Phase 8.

---

## 5. Sliding Window + Sink Tokens (StreamingLLM)

**Paper**: Guangxuan Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

### Core Technical Insight

LLMs trained with a fixed context window (e.g., 4096 tokens) fail catastrophically when asked to process longer sequences. Two naive solutions exist:
- **Dense attention** on a sliding window: recompute all positions within the window (expensive)
- **Window attention** (discard old tokens): perplexity explodes because the model relies on attending to initial tokens

The key discovery is the **attention sink phenomenon**: even tokens that are semantically irrelevant (like the BOS token) receive disproportionately high attention scores from all subsequent tokens. These initial tokens serve as "sinks" that stabilize the softmax distribution. Without them, softmax outputs become degenerate.

### Algorithm: StreamingLLM

```
Maintain a KV cache with two regions:
    1. Sink tokens: first k tokens (typically k=4)
    2. Sliding window: the most recent w tokens

KV cache layout at position t (where t > k + w):
    [token_0, token_1, ..., token_{k-1}, token_{t-w}, token_{t-w+1}, ..., token_{t-1}]
    |<------ sink tokens (k) ------>|  |<----------- window (w) ------------->|

Position encoding fix:
    Sink tokens keep their original positions: 0, 1, ..., k-1
    Window tokens use positions: k, k+1, ..., k+w-1
    (The gap in position IDs doesn't matter because RoPE handles relative positions)
```

### Why Sinks Work (Intuition)

During training, the softmax in attention must sum to 1. When no token is truly "important" for a given query, the model learns to dump excess attention mass onto predictable, always-present tokens (initial tokens). Removing these tokens forces the softmax distribution to redistribute mass, causing instability.

### Performance

- Enables stable inference on **4M+ tokens** with constant memory
- In streaming settings: **22.2x speedup** over sliding window with recomputation
- Works with LLaMA-2, MPT, Falcon, Pythia out of the box (no retraining)
- Perplexity remains stable as sequence length increases (unlike window-only which degrades)
- Just 4 sink tokens are sufficient

### Models with Native Sliding Window

Several models now train with sliding window attention natively:
- **Mistral 7B**: Sliding window of 4096 tokens, every layer
- **Gemma**: Alternates between full attention and sliding window layers
- These models don't need the "sink token hack" because they were trained with the pattern

### Implementation Complexity

**Low**. This is purely a KV cache management strategy. No kernel changes needed. Just maintain a circular buffer with reserved slots for sink tokens.

### RTX 5080 Relevance

**Highly relevant.** On 16GB VRAM, you cannot store full attention KV caches for very long sequences. A 7B model at FP16 uses ~0.5 GB of KV cache per 1K tokens. At 32K context that is ~16 GB just for KV cache, consuming all VRAM. Sliding window + sinks lets you handle arbitrarily long sequences within a fixed memory budget. Implement in Phase 2 as a cache eviction strategy.

---

## 6. Ring Attention / Sequence Parallelism

**Paper**: Hao Liu, Matei Zaharia, Pieter Abbeel. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)

### Core Technical Insight

For very long sequences that exceed a single GPU's memory, the KV cache must be distributed across multiple devices. Ring Attention arranges N devices in a ring topology and overlaps inter-device communication with blockwise attention computation.

### Algorithm

```
Setup: N devices in a ring. Sequence of length L split into N blocks of L/N tokens each.
Device i holds: Q_i, K_i, V_i (its local block)

For round r = 0, 1, ..., N-1:
    In parallel:
        1. COMPUTE: Device i computes attention of Q_i against K_{(i+r) mod N}, V_{(i+r) mod N}
                    Uses online softmax to accumulate partial results
        2. COMMUNICATE: Device i sends its K, V block to device (i+1) mod N
                        Device i receives K, V block from device (i-1) mod N

After N rounds: each device has computed full attention for its Q block against all K, V blocks.
```

### Key Property: Communication Hiding

If the computation time for one block-attention exceeds the communication time for sending one KV block to the next device, then communication is **fully hidden**. The total time equals the computation time alone (no communication overhead).

This holds when `L/N` (block size) is large enough that the attention FLOPS dominate the bandwidth cost.

### Variants and Successors (2024-2025)

**Unified Sequence Parallelism (USP)** (2024, [arXiv:2405.07719](https://arxiv.org/abs/2405.07719)):
- Combines DeepSpeed-Ulysses (which splits along the head dimension) with Ring Attention (which splits along the sequence dimension) into a 2D parallelism scheme
- More robust to different model architectures and network topologies
- Achieved 86% MFU on 16 A800 GPUs for LLaMA3-8B at 208K context

**RingX** (SC 2024):
- Optimized for HPC interconnects (NVLink, InfiniBand)
- 3.4x speedup over standard Ring Attention on Frontier supercomputer
- 38% MFU for LLaMA3-8B at 1M context on 4,096 GPUs

**Context Parallelism** (NVIDIA, 2024):
- NVIDIA's production implementation of sequence-parallel attention
- Integrated into Megatron-LM and NeMo

### Implementation Complexity

**High** (multi-GPU). Requires distributed communication primitives (NCCL send/recv), careful overlap of compute and communication, and online softmax across distributed blocks.

### RTX 5080 Relevance

**Not directly relevant for single-GPU.** Ring Attention is a multi-device technique. With a single RTX 5080, you cannot use it. However, understanding the algorithm is valuable because: (a) it teaches blockwise attention with online softmax accumulation, which is the same primitive used in FlashAttention, and (b) if you ever add a second GPU, you could implement it.

---

## 7. Multi-Head Latent Attention (MLA)

**Paper**: DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)

Also used in: DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

### Core Technical Insight

Standard Multi-Head Attention (MHA) stores separate K and V tensors per head, requiring large KV caches:

```
KV cache per token (MHA) = 2 * n_heads * d_head * dtype_size
For a 7B model (32 heads, d=128, FP16): 2 * 32 * 128 * 2 = 16,384 bytes/token
```

Grouped-Query Attention (GQA) reduces this by sharing KV heads across query heads (e.g., 8 KV heads for 32 query heads), giving a 4x reduction.

MLA takes a radically different approach: instead of storing K and V per head, it compresses **both** into a single low-rank latent vector.

### Formulas

**Compression (at token position t)**:

```
c_t^{KV} = W^{DKV} * h_t        # h_t: [D] -> c_t^{KV}: [d_c]
                                  # W^{DKV} in R^{d_c x D}, d_c << D
```

Where `d_c` is the compressed latent dimension (512 in DeepSeek-V2, vs. original 4096).

**Decompression (during attention)**:

```
k_t^C = W^{UK} * c_t^{KV}       # [d_c] -> [n_h * d_h]  (keys, non-RoPE part)
v_t   = W^{UV} * c_t^{KV}       # [d_c] -> [n_h * d_h]  (values)
```

**Key insight -- the "absorb" trick**: During inference, W^{UK} can be absorbed into the query projection:

```
q_t * W^{UK} * c_t^{KV}  =  (q_t * W^{UK}) * c_t^{KV}  =  q'_t * c_t^{KV}
```

So at inference time, you never actually decompress the latent. You pre-multiply the query by the decompression matrix, then dot-product directly with the compressed latent. Only `c_t^{KV}` (d_c-dimensional) is stored in the KV cache.

**Decoupled RoPE**: RoPE requires position-dependent transformations on K, which breaks the compression. MLA decouples this by adding a small separate RoPE component:

```
k_t = [k_t^C ; k_t^R]           # concatenate compressed keys with RoPE keys
k_t^R = RoPE(W^{KR} * h_t)      # small separate projection for RoPE, dim d_R
```

Only `c_t^{KV}` (dim d_c) and `k_t^R` (dim d_R) are cached per token.

### KV Cache Savings

```
KV cache per token (MLA) = (d_c + d_R) * dtype_size
DeepSeek-V2: d_c=512, d_R=64 -> 576 * 2 bytes = 1,152 bytes/token

vs MHA: 16,384 bytes/token  -> 93.3% reduction
vs GQA (8 KV heads): 4,096 bytes/token -> 71.9% reduction
```

### TransMLA: Retrofitting MLA onto Any Model (2025)

**Paper**: Fuxiao Meng et al. "TransMLA: Multi-Head Latent Attention Is All You Need." NeurIPS 2025 Spotlight. [arXiv:2502.07864](https://arxiv.org/abs/2502.07864)

TransMLA converts existing GQA-based models (LLaMA, Qwen, Mixtral) to use MLA via:
1. Joint PCA across heads for finding optimal low-rank compression
2. Balanced Key-Value (BKV) procedure to equalize K and V subspace norms before PCA
3. FreqFold to improve compression by exploiting RoPE frequency similarity
4. Fine-tuning with only 6B tokens to recover performance

**Result**: 93% KV cache compression on LLaMA-2-7B, 10.6x inference speedup at 8K context, with quality on par with original model.

### Implementation Complexity

**Medium** (for inference with a pre-trained MLA model). The attention kernel is standard -- the main difference is that KV entries are smaller. The absorb trick is just a pre-computation of `q' = q * W^{UK}`. For converting existing models (TransMLA), complexity is higher due to the PCA/fine-tuning pipeline.

### RTX 5080 Relevance

**Highly relevant.** On 16GB VRAM, KV cache size is a primary bottleneck. If you serve DeepSeek-V3 (or a future MLA-based model), MLA's 93% KV reduction is transformative. For Qwen2.5-7B (your target, which uses GQA), you could explore TransMLA conversion. Understanding MLA is also valuable for Phase 8 because the "absorb trick" is a beautiful example of algebraic optimization in attention.

---

## 8. New Attention Variants (2025-2026)

### 8a. Native Sparse Attention (NSA)

**Paper**: DeepSeek-AI. "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention." ACL 2025 Best Paper. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)

**Core insight**: Most sparse attention methods are applied post-training as approximations. NSA is trained from scratch with sparsity, using a dynamic hierarchical strategy:

1. **Coarse-grained token compression**: Compress distant tokens into summary representations
2. **Fine-grained token selection**: Select the most important individual tokens within each block
3. **Sliding window**: Always attend to recent local tokens

The three components are combined dynamically per head per layer.

**Hardware alignment**: NSA designs its sparsity patterns to match GPU memory access patterns (aligned block sizes, coalesced access). This is what makes it fast in practice -- many sparse attention methods have theoretical speedups but poor actual GPU utilization.

**Performance**: Matches or exceeds full attention quality on general benchmarks while achieving substantial speedups on 64K+ sequences. Won ACL 2025 Best Paper.

**RTX 5080 relevance**: **Relevant** for serving DeepSeek models or future NSA-trained models. The kernels could be implemented in Triton for Phase 8 stretch goals.

### 8b. Differential Attention (Diff Transformer)

**Paper**: Tianzhu Ye et al. (Microsoft Research). "Differential Transformer." ICLR 2025. [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)

**Core insight**: Standard attention over-allocates attention to irrelevant context ("attention noise"). Differential attention computes two separate attention maps and subtracts them:

```
DiffAttn(Q, K, V) = (softmax(Q_1 * K_1^T / sqrt(d)) - lambda * softmax(Q_2 * K_2^T / sqrt(d))) * V
```

Where Q is split into Q_1, Q_2 (and similarly K into K_1, K_2), and lambda is a learnable scalar. The subtraction cancels common-mode noise (analogous to differential amplifiers in electrical engineering or noise-canceling headphones).

**Benefits**: Better long-context modeling, reduced hallucinations, improved key information retrieval, fewer activation outliers (which helps quantization).

**Implementation**: Since it uses standard softmax, it can leverage FlashAttention kernels. The extra computation is roughly 2x attention (two softmax maps), but each operates on half-dimension heads, so the total cost is similar.

**RTX 5080 relevance**: **Moderately relevant.** If future models adopt DiffAttn (which is likely given the ICLR 2025 publication and Microsoft backing), you would need to support it. The kernel changes are minimal.

### 8c. SpargeAttention

**Paper**: Tsinghua University. "SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference." ICML 2025. [arXiv:2502.18137](https://arxiv.org/abs/2502.18137)

**Core insight**: A universal, training-free sparse attention with a two-stage online filter:
1. **Stage 1**: Rapidly predict the attention map to skip unnecessary QK^T GEMM blocks
2. **Stage 2**: Online softmax-aware filter that further skips unnecessary PV GEMM blocks with zero extra overhead

**Performance**: 4-7x speedup on language, image, and video models without fine-tuning.

**RTX 5080 relevance**: **Highly relevant.** Training-free means it works with any existing model (including your Qwen2.5-7B). Could be implemented as a Phase 8 optimization.

### 8d. XAttention

**Paper**: Ruyi Xu, Guangxuan Xiao et al. (MIT Han Lab + NVIDIA). "XAttention: Block Sparse Attention with Antidiagonal Scoring." ICML 2025. [arXiv:2503.16428](https://arxiv.org/abs/2503.16428)

**Core insight**: For block-sparse attention, you need to efficiently determine which blocks are "important." XAttention discovers that summing **antidiagonal values** (lower-left to upper-right diagonal) of the attention matrix provides an excellent proxy for block importance, and this can be computed cheaply.

**Performance**: Up to 13.5x speedup on core attention, with near-lossless accuracy on long-context tasks.

**RTX 5080 relevance**: **Relevant** for long-context inference optimization in Phase 8.

### 8e. Star Attention

**Paper**: Shantanu Acharya, Fei Jia, Boris Ginsburg (NVIDIA). "Star Attention: Efficient LLM Inference over Long Sequences." ICML 2025. [arXiv:2411.17116](https://arxiv.org/abs/2411.17116)

**Core insight**: A two-phase block-sparse approximation for multi-host inference:
- **Phase 1 (context encoding)**: Shard input into blocks across hosts. Each host processes its block prefixed with an "anchor block" (first block of the sequence, similar to attention sinks).
- **Phase 2 (query/generation)**: Broadcast query to all hosts, each computes local attention, then aggregate softmax statistics at a designated host.

**Performance**: Up to 11x reduction in memory and inference time, preserving 97-100% accuracy.

**RTX 5080 relevance**: **Low for single-GPU.** This is a multi-host technique. However, the anchor block concept (attention sinks) reinforces StreamingLLM insights.

### 8f. FlashInfer: Composable Attention Engine

**Paper**: Zihao Ye et al. (University of Washington). "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." MLSys 2025 Best Paper. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

**Core insight**: Rather than one monolithic attention kernel, FlashInfer provides a composable framework:
- **Block-sparse format** for heterogeneous KV cache layouts (paged, ragged, shared-prefix)
- **JIT compilation** of custom attention variants (all LLaMA variants compile in <15s)
- **Load-balanced scheduling** across variable-length sequences, compatible with CUDA Graphs

**Performance**: 29-69% inter-token-latency reduction, 28-30% latency reduction for long-context inference.

Integrated into SGLang, vLLM, and MLC-Engine.

**RTX 5080 relevance**: **Highly relevant** as a reference implementation and potential dependency. Understanding FlashInfer's composable design is valuable for Phase 8.

### 8g. Linear Attention Hybrids (Gated DeltaNet)

**Paper**: Songlin Yang et al. (NVIDIA). "Gated Delta Networks: Improving Mamba2 with Delta Rule." ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

**Core insight**: Replace quadratic attention with a linear recurrence that combines:
- **Gated decay** (from Mamba2): adaptive memory erasure
- **Delta rule**: precise memory write/overwrite (vs. Mamba's additive-only updates)

Recent production models using this: Qwen3-Next and Kimi Linear use a 3:1 ratio (3 Gated DeltaNet layers per 1 full attention layer).

**Complexity**: O(L) instead of O(L^2) in sequence length. Constant KV cache size regardless of sequence length.

**RTX 5080 relevance**: **Relevant for future models.** If Qwen3-Next becomes your target model, you would need to implement Gated DeltaNet layers. For now, understand the concept.

---

## 9. Relevance Matrix for RTX 5080

Summary of which techniques to prioritize for your single-GPU RTX 5080 (Blackwell, 16GB) inference engine:

| Technique | Phase | Priority | Reason |
|-----------|-------|----------|--------|
| Flash-Decoding (split-KV) | 8 | **Critical** | Essential for decode performance on single GPU |
| PagedAttention | 2, 8 | **Critical** | Essential for memory efficiency on 16GB |
| Sliding Window + Sinks | 2 | **High** | Required for long-context on 16GB |
| FlashAttention-3 (concepts) | 8 | **High** | Online softmax, tiling, FP8 techniques |
| FlashAttention-4 | 8 | **High** | Native to your Blackwell arch (when available) |
| FP8 block quantization | 4, 8 | **High** | Your GPU has native FP8 Tensor Cores |
| SpargeAttention | 8+ | **Medium** | Training-free sparse, works on any model |
| MLA (DeepSeek) | 8+ | **Medium** | Relevant if serving DeepSeek or using TransMLA |
| Differential Attention | 8+ | **Medium** | Future models may use it |
| XAttention | 8+ | **Medium** | Good for long-context optimization |
| NSA (DeepSeek) | 8+ | **Low-Med** | Only for NSA-trained models |
| FlashInfer (as reference) | 8 | **Medium** | Study composable kernel design |
| Ring Attention | -- | **Low** | Multi-GPU only |
| Star Attention | -- | **Low** | Multi-host only |
| Gated DeltaNet | -- | **Low** | Future model architecture, not current target |

### Recommended Implementation Order for Phase 8

1. **Naive tiled attention** with online softmax (understand the core algorithm)
2. **Flash-Decoding** (split-KV for decode phase) -- immediate performance win
3. **Paged attention kernel** (integrate with Phase 2's paged KV cache)
4. **FlashAttention-style fused kernel** in Triton (tiled, IO-aware)
5. **FP8 attention** with block quantization (leverage Blackwell Tensor Cores)
6. **Sliding window attention** kernel variant
7. **(Stretch)** SpargeAttention or XAttention for long-context sparse attention

---

## Key References (Consolidated)

### FlashAttention Family
- Tri Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Tri Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Tri Dao, Jay Shah. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." NeurIPS 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Tri Dao. FlashAttention-4 (Blackwell). Hot Chips 2025. [Modal reverse-engineering blog](https://modal.com/blog/reverse-engineer-flash-attention-4)

### Decode-Phase Optimization
- Stanford CRFM. "Flash-Decoding for long-context inference." October 2023. [Blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- Ke Hong et al. "FlashDecoding++: Faster LLM Inference with Asynchronization, Flat GEMM Optimization, and Heuristics." MLSys 2024. [Paper](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html)

### Memory Management
- Woosuk Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- [vLLM PagedAttention kernel docs](https://docs.vllm.ai/en/stable/design/paged_attention/)

### Streaming / Long-Context
- Guangxuan Xiao et al. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Hao Liu et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context." ICLR 2024. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- Jiarui Fang et al. "A Unified Sequence Parallelism Approach for Long Context Generative AI." 2024. [arXiv:2405.07719](https://arxiv.org/abs/2405.07719)

### KV Cache Compression
- DeepSeek-AI. "DeepSeek-V2." 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI. "DeepSeek-V3 Technical Report." 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Fuxiao Meng et al. "TransMLA: Multi-Head Latent Attention Is All You Need." NeurIPS 2025 Spotlight. [arXiv:2502.07864](https://arxiv.org/abs/2502.07864)

### Sparse Attention (2025)
- DeepSeek-AI. "Native Sparse Attention." ACL 2025 Best Paper. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)
- "SpargeAttention." ICML 2025. [arXiv:2502.18137](https://arxiv.org/abs/2502.18137)
- Ruyi Xu et al. "XAttention: Block Sparse Attention with Antidiagonal Scoring." ICML 2025. [arXiv:2503.16428](https://arxiv.org/abs/2503.16428)

### New Architectures
- Tianzhu Ye et al. "Differential Transformer." ICLR 2025. [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)
- Songlin Yang et al. "Gated Delta Networks." ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Shantanu Acharya et al. "Star Attention." ICML 2025. [arXiv:2411.17116](https://arxiv.org/abs/2411.17116)

### Attention Engines / Frameworks
- Zihao Ye et al. "FlashInfer: Efficient and Customizable Attention Engine." MLSys 2025 Best Paper. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
