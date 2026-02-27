# Transformer Inference Mathematics Reference

A complete, implementable mathematical reference for building an LLM inference engine from scratch. All formulas are precise and include tensor shapes, FLOPs, and memory access patterns.

**Target model reference: Qwen2.5-7B-Instruct**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Hidden dimension | H | 3584 |
| Intermediate (FFN) dimension | I | 18944 |
| Number of attention heads | N | 28 |
| Number of KV heads | K | 4 |
| Head dimension | D | 128 (= H/N) |
| Number of layers | L | 28 |
| Vocabulary size | V | 152064 |
| Max position embeddings | - | 32768 |
| RMSNorm epsilon | eps | 1e-6 |
| RoPE base theta | theta | 1000000.0 |
| Dtype | - | bfloat16 |

**Notation conventions used throughout**:
- B = batch size
- S = sequence length (total, including past)
- S_new = number of new tokens (S_new = S during prefill, S_new = 1 during decode)
- H = hidden dimension
- N = number of query heads
- K = number of KV heads
- D = head dimension (= H / N)
- G = group size (= N / K = number of query heads per KV head)
- I = FFN intermediate dimension
- V = vocabulary size
- L = number of transformer layers

---

## 1. RMSNorm (Root Mean Square Layer Normalization)

### Formula

Given input `x` of shape `[B, S, H]` and learnable weight `w` of shape `[H]`:

```
RMS(x) = sqrt( (1/H) * sum(x_i^2, i=1..H) + eps )

RMSNorm(x) = (x / RMS(x)) * w
```

Expanded step by step:

```
1. Compute mean of squares:   ms = (1/H) * sum_{i=1}^{H} x_i^2      # scalar per (b,s) position
2. Compute RMS:               rms = sqrt(ms + eps)                     # add eps INSIDE sqrt for stability
3. Normalize:                 x_norm = x / rms                         # broadcast: [B,S,H] / [B,S,1]
4. Scale:                     output = x_norm * w                      # element-wise: [B,S,H] * [H]
```

**Critical detail**: The epsilon is added BEFORE the square root, not after. This prevents division by zero when the input is all zeros.

### Why not LayerNorm?

RMSNorm omits the mean-centering step of LayerNorm. LayerNorm computes `(x - mean) / std`, while RMSNorm computes `x / RMS(x)`. This saves one reduction operation (computing the mean) with negligible quality loss.

### Tensor shapes

| Tensor | Shape |
|--------|-------|
| Input x | [B, S, H] |
| Weight w | [H] |
| Mean of squares (ms) | [B, S, 1] |
| Output | [B, S, H] |

### FLOPs

Per token position (one [H]-vector):
- Square each element: H multiplications
- Sum: H-1 additions
- Divide by H: 1 division
- Add eps + sqrt: 2 ops
- Divide x by rms: H divisions
- Multiply by weight: H multiplications

**Total: ~3H FLOPs per position, or 3*B*S*H total.**

For Qwen2.5-7B: 3 * 3584 = 10,752 FLOPs per position.

### Memory access

- Read: x [B*S*H elements] + w [H elements] = (B*S*H + H) * sizeof(dtype) bytes
- Write: output [B*S*H elements] = B*S*H * sizeof(dtype) bytes
- Intermediate: ms [B*S] elements (can stay in registers if fused)

**Total bytes: (2*B*S*H + H) * sizeof(dtype)**

This operation is heavily memory-bound. Arithmetic intensity = 3H / ((2*H + H/(B*S)) * sizeof(dtype)) ~ 3 / (2 * sizeof(dtype)) ~ 0.75 FLOPs/byte for bf16. Far below the roofline ridge point.

### Fused RMSNorm + Residual

In practice, you fuse the residual connection with RMSNorm to save a memory round-trip:

```
# Instead of:
#   hidden = hidden + residual           # 1 read + 1 write of [B,S,H]
#   normed = rmsnorm(hidden)             # 1 read + 1 write of [B,S,H]

# Fused:
#   normed, hidden = fused_rmsnorm_residual(x, residual, w)  # 2 reads + 2 writes total
```

This eliminates one full read+write of a [B,S,H] tensor.

---

## 2. Rotary Position Embeddings (RoPE)

### Core Idea

RoPE encodes position by rotating pairs of dimensions in the query and key vectors. Two adjacent dimensions (2i, 2i+1) are treated as a 2D coordinate and rotated by an angle that depends on the token position.

### Frequency Computation

For dimension index `i` in range `[0, D/2)` where D is head_dim:

```
theta_i = base^(-2i / D)
```

For Qwen2.5-7B: base = 1,000,000, D = 128, so:

```
theta_0 = 1000000^(0/128) = 1.0
theta_1 = 1000000^(-2/128) = 1000000^(-1/64)
theta_2 = 1000000^(-4/128) = 1000000^(-2/64)
...
theta_63 = 1000000^(-126/128) = 1000000^(-63/64)
```

In code:

```python
# freqs has shape [D/2]
dim_indices = torch.arange(0, D, 2, dtype=torch.float32)  # [0, 2, 4, ..., D-2]
freqs = 1.0 / (base ** (dim_indices / D))                  # [D/2]
# freqs = [theta_0, theta_1, ..., theta_{D/2 - 1}]
```

### Angle Computation

For position `m`, the angle for the i-th dimension pair is:

```
angle_{m,i} = m * theta_i
```

Precompute for all positions:

```python
positions = torch.arange(max_seq_len, dtype=torch.float32)   # [max_seq_len]
angles = torch.outer(positions, freqs)                         # [max_seq_len, D/2]
# angles[m, i] = m * theta_i
```

### Rotation Application (Two equivalent methods)

**Method 1: Complex multiplication (recommended for simplicity)**

Treat each pair (x_{2i}, x_{2i+1}) as a complex number x_{2i} + j*x_{2i+1}:

```python
# x has shape [B, S, N, D] (query or key vectors after head splitting)
# Reshape to complex: [B, S, N, D/2, 2] -> view as complex [B, S, N, D/2]
x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

# Rotation factors as complex exponentials: [S, D/2]
cos_angles = torch.cos(angles)  # [S, D/2]
sin_angles = torch.sin(angles)  # [S, D/2]
rotation = torch.complex(cos_angles, sin_angles)  # [S, D/2] = e^{j * angle}

# Apply rotation: broadcast over [B, N] dimensions
# rotation shape: [1, S, 1, D/2]
x_rotated = x_complex * rotation.unsqueeze(0).unsqueeze(2)

# Convert back to real: [B, S, N, D/2] -> [B, S, N, D]
output = torch.view_as_real(x_rotated).reshape(*x.shape)
```

**Method 2: Sin/cos pairs (no complex numbers, used in most implementations)**

Split x into even and odd indices:

```
x_even = x[..., 0::2]   # [B, S, N, D/2]   (dims 0, 2, 4, ...)
x_odd  = x[..., 1::2]   # [B, S, N, D/2]   (dims 1, 3, 5, ...)
```

Apply 2D rotation:

```
output_even = x_even * cos(angles) - x_odd * sin(angles)
output_odd  = x_even * sin(angles) + x_odd * cos(angles)
```

Interleave back:

```
output[..., 0::2] = output_even
output[..., 1::2] = output_odd
```

This is equivalent to the rotation matrix per (2i, 2i+1) pair:

```
[ cos(m*theta_i)  -sin(m*theta_i) ] [ x_{2i}   ]
[ sin(m*theta_i)   cos(m*theta_i) ] [ x_{2i+1} ]
```

**Method 3: Llama-style (rotate_half)**

Some implementations (Llama, Qwen) use a "rotate half" approach:

```python
def rotate_half(x):
    """Split x into two halves and negate+swap."""
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    return torch.cat((-x2, x1), dim=-1)

# cos_cache, sin_cache have shape [S, D] (repeated to fill full D)
output = x * cos_cache + rotate_half(x) * sin_cache
```

Here `cos_cache` and `sin_cache` each have shape `[S, D]` where each theta is repeated:

```
cos_cache[m, :] = [cos(m*theta_0), cos(m*theta_1), ..., cos(m*theta_{D/2-1}),
                   cos(m*theta_0), cos(m*theta_1), ..., cos(m*theta_{D/2-1})]
```

This is mathematically identical to Method 2 but avoids interleaving.

### Why RoPE Works

The key property: after rotation, the dot product between query at position m and key at position n depends only on (m - n):

```
<RoPE(q, m), RoPE(k, n)> = <q, R(m-n) @ k>
```

This means attention scores naturally encode relative position without explicit position encodings.

### NTK-Aware Scaling for Extended Context

To extend context from trained length `L_train` to `L_target`, scale factor `s = L_target / L_train`.

**Linear scaling** (simple but degrades high-frequency info):

```
theta_i_scaled = theta_i / s
```

Equivalent to: `angles_scaled[m, i] = m * theta_i / s = (m/s) * theta_i`

**NTK-aware scaling** (preserves high-frequency dimensions):

Change the base instead of scaling positions:

```
base_scaled = base * s^(D / (D - 2))
theta_i_ntk = base_scaled^(-2i / D)
```

For Qwen2.5-7B extending 32K -> 128K (s=4):

```
base_scaled = 1,000,000 * 4^(128/126) = 1,000,000 * 4.0443 ~ 4,044,300
```

**YaRN scaling** (state of the art, blends linear and NTK):

```
alpha = (s * D) / (2 * pi * dim)    # per-dimension interpolation factor
gamma_i = 1 - ramp(alpha_i)          # 0 for low-freq (NTK), 1 for high-freq (linear)
theta_i_yarn = theta_i * (1 - gamma_i) + (theta_i / s) * gamma_i
```

Where `ramp(x)` smoothly transitions from 0 to 1 based on the frequency band.

### Tensor Shapes

| Tensor | Shape |
|--------|-------|
| Query/Key input | [B, S, N, D] or [B, S, K, D] |
| Frequencies | [D/2] |
| Angles | [S, D/2] |
| cos/sin cache | [S, D/2] or [S, D] |
| Output | same as input |

### FLOPs

Per query/key vector (one [D]-vector):
- 4 multiplications + 2 additions per dimension pair = 6 * (D/2) = 3D FLOPs

**Total per layer: 3D * (N + K) * B * S FLOPs** (applied to both Q and K)

For Qwen2.5-7B: 3 * 128 * (28 + 4) * B * S = 12,288 * B * S FLOPs per layer.

### Memory Access

- Read: Q or K tensor [B*S*N*D] + cos/sin caches [S*D]
- Write: rotated Q or K [B*S*N*D]

Memory-bound operation (arithmetic intensity ~ 3 FLOPs per element, 2 reads + 1 write per element).

---

## 3. Grouped-Query Attention (GQA)

### Overview

In GQA, N query heads are divided into K groups, each sharing one KV head. The group size is G = N/K.

For Qwen2.5-7B: N=28 query heads, K=4 KV heads, G=7 query heads per KV group.

### Step-by-step Computation

**Step 1: Linear projections**

```
Q = x @ W_q    # [B, S_new, H] @ [H, N*D] -> [B, S_new, N*D]
K = x @ W_k    # [B, S_new, H] @ [H, K*D] -> [B, S_new, K*D]
V = x @ W_v    # [B, S_new, H] @ [H, K*D] -> [B, S_new, K*D]
```

For Qwen2.5-7B:
- W_q: [3584, 3584] (28 heads * 128 dim = 3584)
- W_k: [3584, 512]  (4 heads * 128 dim = 512)
- W_v: [3584, 512]  (4 heads * 128 dim = 512)

**Step 2: Reshape into heads**

```
Q = Q.view(B, S_new, N, D)       # [B, S_new, 28, 128]
K = K.view(B, S_new, K, D)       # [B, S_new, 4, 128]
V = V.view(B, S_new, K, D)       # [B, S_new, 4, 128]
```

**Step 3: Apply RoPE to Q and K** (see Section 2)

```
Q = apply_rope(Q, positions)      # [B, S_new, N, D]
K = apply_rope(K, positions)      # [B, S_new, K, D]
```

**Step 4: KV cache update** (during decode)

```
# Append new K, V to cache
K_cache = concat(K_cache_prev, K, dim=1)   # [B, S, K, D]  (S = total seq len)
V_cache = concat(V_cache_prev, V, dim=1)   # [B, S, K, D]
```

**Step 5: Expand KV heads to match query heads**

Each KV head is repeated G times:

```
K_expanded = K_cache.repeat_interleave(G, dim=2)  # [B, S, K, D] -> [B, S, N, D]
V_expanded = V_cache.repeat_interleave(G, dim=2)  # [B, S, K, D] -> [B, S, N, D]
```

Alternatively (more memory-efficient, no copy): reshape Q to group KV heads:

```
# Instead of expanding K,V, reshape Q:
Q = Q.view(B, S_new, K, G, D)           # [B, S_new, K, G, D]
# Then attention is computed per KV-group
```

**Step 6: Transpose for batched matmul**

```
Q = Q.transpose(1, 2)          # [B, N, S_new, D]
K = K_expanded.transpose(1, 2) # [B, N, S, D]
V = V_expanded.transpose(1, 2) # [B, N, S, D]
```

**Step 7: Compute attention scores**

```
scores = Q @ K.transpose(-2, -1) / sqrt(D)
# [B, N, S_new, D] @ [B, N, D, S] -> [B, N, S_new, S]
# Division by sqrt(D) = sqrt(128) = 11.3137...
```

**Step 8: Apply causal mask**

```
# Causal mask: position i can only attend to positions <= i
# mask[i, j] = 0 if j <= i, else -inf
mask = torch.triu(torch.full((S_new, S), float('-inf')), diagonal=S - S_new + 1)
scores = scores + mask    # broadcast: [B, N, S_new, S] + [S_new, S]
```

During decode (S_new = 1), the mask is trivially all-zeros (the single new token can attend to all past tokens), so no mask is needed.

During prefill (S_new = S), the mask is a standard upper-triangular matrix of -inf.

**Step 9: Softmax**

```
attn_weights = softmax(scores, dim=-1)    # [B, N, S_new, S]
# Each row sums to 1
```

**Step 10: Weighted sum of values**

```
output = attn_weights @ V     # [B, N, S_new, S] @ [B, N, S, D] -> [B, N, S_new, D]
```

**Step 11: Reshape and project output**

```
output = output.transpose(1, 2).contiguous().view(B, S_new, N*D)  # [B, S_new, H]
output = output @ W_o                                               # [B, S_new, H] @ [H, H] -> [B, S_new, H]
```

### Causal Mask Construction Details

For prefill with sequence length S:

```python
# Method 1: Upper triangular
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))

# Method 2: Comparison (more flexible for non-contiguous positions)
positions = torch.arange(S)
mask = positions.unsqueeze(0) > positions.unsqueeze(1)  # [S, S] boolean
# mask[i,j] = True if j > i (future position)
```

For decode with KV cache (S_new=1, attending to S total positions): no mask needed (the single query can attend to all cached positions).

### FLOPs

Per layer, per batch element:

| Operation | FLOPs |
|-----------|-------|
| Q projection | 2 * S_new * H * (N*D) = 2 * S_new * H^2 |
| K projection | 2 * S_new * H * (K*D) |
| V projection | 2 * S_new * H * (K*D) |
| QK^T | 2 * N * S_new * S * D |
| Softmax | ~5 * N * S_new * S (exp, sum, div per element) |
| Attn @ V | 2 * N * S_new * S * D |
| Output projection | 2 * S_new * H * H |

For Qwen2.5-7B (per token during decode, S_new=1):
- Q proj: 2 * 3584 * 3584 = 25,690,112
- K proj: 2 * 3584 * 512 = 3,670,016
- V proj: 2 * 3584 * 512 = 3,670,016
- QK^T: 2 * 28 * 1 * S * 128 = 7,168 * S
- Attn@V: 2 * 28 * 1 * S * 128 = 7,168 * S
- O proj: 2 * 3584 * 3584 = 25,690,112

**Total attention: ~58.7M + 14,336*S FLOPs per token per layer.**

### Memory Access (Decode Phase, S_new=1)

This is the critical bottleneck. For each token decoded:

| Data | Size (bytes, bf16) |
|------|--------------------|
| W_q | N*D*H * 2 = 3584*3584*2 = 25.7 MB |
| W_k | K*D*H * 2 = 512*3584*2 = 3.67 MB |
| W_v | K*D*H * 2 = 512*3584*2 = 3.67 MB |
| W_o | H*H * 2 = 3584*3584*2 = 25.7 MB |
| K cache | B*S*K*D * 2 |
| V cache | B*S*K*D * 2 |
| **Total weights** | **~58.7 MB per layer** |

For 28 layers: 28 * 58.7 MB = ~1.64 GB of weight reads per token.

At 960 GB/s bandwidth: minimum 1.64 / 960 = 1.71 ms per token = ~585 tokens/sec (B=1, weight-read limited, ignoring KV cache reads).

---

## 4. SwiGLU / SiLU-GLU Feed-Forward Network

### Formula

```
FFN(x) = W_down @ (SiLU(W_gate @ x) * (W_up @ x))
```

Where:

```
SiLU(z) = z * sigmoid(z) = z * (1 / (1 + exp(-z)))
```

Sometimes written as Swish_1(z) (Swish with beta=1).

### Step-by-step

```
1. gate = x @ W_gate.T     # [B, S, H] @ [H, I] -> [B, S, I]      (linear, no bias)
2. up   = x @ W_up.T       # [B, S, H] @ [H, I] -> [B, S, I]      (linear, no bias)
3. gate = SiLU(gate)        # [B, S, I] -> [B, S, I]                (element-wise)
4. hidden = gate * up       # [B, S, I] * [B, S, I] -> [B, S, I]   (element-wise multiply)
5. output = hidden @ W_down.T  # [B, S, I] @ [I, H] -> [B, S, H]  (linear, no bias)
```

For Qwen2.5-7B: H=3584, I=18944
- W_gate: [18944, 3584]
- W_up:   [18944, 3584]
- W_down: [3584, 18944]

### SiLU Implementation Detail

```python
def silu(x):
    return x * torch.sigmoid(x)

# Numerically: for large negative x, sigmoid(x) -> 0, so SiLU(x) -> 0
# For large positive x, sigmoid(x) -> 1, so SiLU(x) -> x
# SiLU(0) = 0
# SiLU has a small negative region: minimum at x ~ -1.278, SiLU(-1.278) ~ -0.278
```

### Why SwiGLU over standard FFN?

Standard FFN: `FFN(x) = W2 @ ReLU(W1 @ x)` with shapes [H, 4H] and [4H, H].

SwiGLU uses 3 weight matrices instead of 2, but each matrix is smaller (I ~ 8H/3 instead of 4H), so total parameter count is: 3 * H * (8H/3) = 8H^2 vs 2 * H * 4H = 8H^2. Same parameter count, better performance.

For Qwen2.5-7B: 3 * 3584 * 18944 = 203,685,888 parameters per FFN layer. 18944/3584 = 5.286 expansion ratio (not exactly 8/3 due to rounding to nice numbers).

### FLOPs

Per token position:

| Operation | FLOPs |
|-----------|-------|
| Gate projection | 2 * H * I |
| Up projection | 2 * H * I |
| SiLU | ~4 * I (sigmoid: exp + add + div, then multiply) |
| Element-wise multiply | I |
| Down projection | 2 * I * H |

**Total: 6*H*I + 5*I FLOPs per position.**

For Qwen2.5-7B: 6 * 3584 * 18944 + 5 * 18944 = 407,371,776 + 94,720 ~ 407.5M FLOPs per position.

### Memory Access (Decode Phase)

| Data | Size (bytes, bf16) |
|------|--------------------|
| W_gate | H * I * 2 = 135.8 MB |
| W_up | H * I * 2 = 135.8 MB |
| W_down | I * H * 2 = 135.8 MB |
| **Total** | **~407.4 MB per layer** |

For 28 layers of FFN: 28 * 407.4 MB = ~11.4 GB of weight reads per token.

### Kernel Fusion Opportunity

Steps 1-4 can be fused: compute gate and up projections, apply SiLU, and multiply, writing only the final [B, S, I] intermediate. This eliminates writing/reading two [B, S, I] intermediates.

---

## 5. Autoregressive Decoding

### The Full Forward Pass (One Transformer Layer)

```
Input: x [B, S_new, H], KV_cache_prev

1. residual = x
2. x = RMSNorm(x, w_input_norm)                     # Pre-norm
3. Q, K, V = linear_projections(x)                   # QKV projections
4. Q, K = apply_rope(Q, K, positions)                # Positional encoding
5. K_cache, V_cache = update_cache(K, V, cache)      # Append to cache
6. attn_output = grouped_query_attention(Q, K_cache, V_cache)
7. attn_output = attn_output @ W_o                   # Output projection
8. x = residual + attn_output                        # Residual connection
9. residual = x
10. x = RMSNorm(x, w_post_norm)                      # Pre-norm for FFN
11. ffn_output = SwiGLU_FFN(x)                        # Feed-forward
12. x = residual + ffn_output                         # Residual connection

Output: x [B, S_new, H], updated KV_cache
```

### Full Model Forward Pass

```
1. token_ids: [B, S_new]                                       # Input token IDs
2. x = embedding_table[token_ids]                               # [B, S_new, H] — lookup, no matmul
3. for layer_i in range(L):                                     # L = 28 for Qwen2.5-7B
       x, kv_cache[layer_i] = transformer_layer(x, kv_cache[layer_i])
4. x = RMSNorm(x, w_final_norm)                                # Final layer norm
5. logits = x @ W_lm_head.T                                    # [B, S_new, H] @ [H, V] -> [B, S_new, V]
6. next_token_logits = logits[:, -1, :]                         # [B, V] — only last position matters
```

### Prefill Phase

Process all input tokens in parallel:

```
Input: prompt_token_ids [B, S]       # Full prompt

1. x = embed(prompt_token_ids)       # [B, S, H]
2. positions = [0, 1, 2, ..., S-1]
3. Forward pass with S_new = S
4. Causal mask applied (upper triangular -inf)
5. KV cache initialized: K_cache [B, S, K, D], V_cache [B, S, K, D]
6. Return logits[:, -1, :]           # Logits for next token prediction

Compute characteristics: matrix-matrix multiplications (compute-bound)
- Projections: [B*S, H] @ [H, D*N] — large GEMM
- QK^T: [B*N, S, D] @ [B*N, D, S] — large batched GEMM
```

### Decode Phase

Process one token at a time, reusing cached KV:

```
Input: new_token_id [B, 1], kv_cache

1. x = embed(new_token_id)           # [B, 1, H]
2. position = current_position       # scalar
3. Forward pass with S_new = 1
4. No causal mask needed (single query attends to all past)
5. KV cache appended: K_cache grows to [B, S+1, K, D]
6. Return logits[:, 0, :]            # [B, V]
7. Sample next token
8. If not stop_condition: repeat with new token

Compute characteristics: matrix-vector multiplications (memory-bound)
- Projections: [1, H] @ [H, D*N] — each weight is read for just 1 token
- QK^T: [B*N, 1, D] @ [B*N, D, S] — matrix-vector product
```

### The Complete Decode Loop Algorithm

```python
def generate(prompt_tokens, max_new_tokens, temperature, top_k, top_p):
    # PREFILL: process entire prompt at once
    positions = range(len(prompt_tokens))
    logits, kv_cache = forward(prompt_tokens, positions, kv_cache=None)

    generated = []
    next_token_logits = logits[:, -1, :]       # [B, V]

    for step in range(max_new_tokens):
        # SAMPLE
        next_token = sample(next_token_logits, temperature, top_k, top_p)
        generated.append(next_token)

        # CHECK STOP
        if next_token == eos_token_id:
            break

        # DECODE: forward pass with single token
        position = len(prompt_tokens) + step
        next_token_logits, kv_cache = forward(
            next_token.unsqueeze(1),            # [B, 1]
            positions=[position],
            kv_cache=kv_cache
        )
        next_token_logits = next_token_logits[:, 0, :]  # [B, V]

    return generated
```

### Memory Budget (Qwen2.5-7B, bf16)

| Component | Size |
|-----------|------|
| Embedding table | V * H * 2 = 152064 * 3584 * 2 = ~1.04 GB |
| Per-layer weights (attn) | (H*N*D + 2*H*K*D + H*H) * 2 = ~58.7 MB |
| Per-layer weights (FFN) | 3 * H * I * 2 = ~407.4 MB |
| Per-layer total | ~466 MB |
| All 28 layers | ~13.1 GB |
| Final norm + LM head | H * 2 + V * H * 2 = ~1.04 GB |
| **Total model weights** | **~15.2 GB (bf16)** |
| KV cache per layer per token | 2 * K * D * 2 = 2 * 4 * 128 * 2 = 2,048 bytes |
| KV cache for S=2048, 28 layers | 28 * 2048 * 2048 = ~114 MB |

---

## 6. Sampling Algorithms

All sampling operates on `logits` of shape `[B, V]` — the raw output of the LM head before any probability conversion.

### 6.1 Temperature Scaling

```
scaled_logits = logits / T
```

- T = 1.0: no change
- T < 1.0: sharper distribution (more confident, less random)
- T > 1.0: flatter distribution (more random, more creative)
- T -> 0: approaches greedy (argmax)
- T -> inf: approaches uniform random

**Apply temperature FIRST, before other sampling methods.** Temperature is applied to logits, not to probabilities.

### 6.2 Top-k Sampling

Keep only the k tokens with highest logits, set rest to -infinity:

```python
def top_k_sample(logits, k):
    # logits: [B, V]
    values, indices = torch.topk(logits, k, dim=-1)   # [B, k]
    # Create mask: set everything below the k-th value to -inf
    min_value = values[:, -1].unsqueeze(-1)             # [B, 1]  (k-th largest)
    logits[logits < min_value] = float('-inf')
    probs = softmax(logits, dim=-1)                     # [B, V]
    return torch.multinomial(probs, num_samples=1)      # [B, 1]
```

**FLOPs**: O(V * log(k)) for partial sort, O(V) for mask + softmax.

### 6.3 Top-p (Nucleus) Sampling

Keep the smallest set of tokens whose cumulative probability >= p:

```python
def top_p_sample(logits, p):
    # logits: [B, V]
    probs = softmax(logits, dim=-1)                     # [B, V]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # [B, V]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, V]

    # Find cutoff: first position where cumulative prob exceeds p
    # Remove tokens with cumulative probability above the threshold
    # Keep the first token that crosses the threshold (shift right by 1)
    sorted_mask = cumulative_probs - sorted_probs >= p  # [B, V] boolean
    # sorted_mask[i] = True means: even without token i, cumsum already >= p

    sorted_probs[sorted_mask] = 0.0                     # Zero out excess tokens
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # Renormalize

    # Sample from filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # [B, 1]
    # Map back to original vocabulary indices
    next_token = sorted_indices.gather(-1, sampled_sorted_idx)  # [B, 1]
    return next_token
```

**Key detail**: The subtraction `cumulative_probs - sorted_probs` ensures we keep the token that first crosses the threshold. Without this, we might exclude the most probable token.

**FLOPs**: O(V log V) for sort, O(V) for cumsum + mask + renormalize.

### 6.4 Min-p Sampling

Dynamic threshold based on the top token's probability:

```python
def min_p_sample(logits, min_p, temperature=1.0):
    # logits: [B, V]
    # Apply temperature first
    logits = logits / temperature
    probs = softmax(logits, dim=-1)                     # [B, V]

    # Find the maximum probability token
    p_max = probs.max(dim=-1, keepdim=True).values      # [B, 1]

    # Compute dynamic threshold
    p_threshold = p_max * min_p                          # [B, 1]

    # Mask tokens below threshold
    mask = probs < p_threshold                           # [B, V]
    probs[mask] = 0.0

    # Renormalize and sample
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)
```

**Key properties**:
- When model is confident (high p_max), threshold is high -> fewer candidates -> coherent
- When model is uncertain (low p_max), threshold is low -> more candidates -> diverse
- Typical values: min_p = 0.05 to 0.1
- min_p = 0.1 means: only consider tokens with at least 10% of the top token's probability

**FLOPs**: O(V) for softmax + max + threshold + mask.

### 6.5 Repetition Penalty

From the CTRL paper (Keskar et al., 2019). Applied to logits BEFORE temperature/sampling:

```python
def apply_repetition_penalty(logits, generated_token_ids, penalty):
    # logits: [B, V]
    # generated_token_ids: set or tensor of previously generated token IDs
    # penalty: float > 1.0 to discourage repetition

    for token_id in generated_token_ids:
        if logits[0, token_id] > 0:
            logits[0, token_id] = logits[0, token_id] / penalty
        else:
            logits[0, token_id] = logits[0, token_id] * penalty

    return logits
```

**Why the if/else?** Both operations push the logit toward zero (reducing its relative probability after softmax). Division shrinks positive logits; multiplication makes negative logits more negative.

Vectorized form:

```python
# Gather logits for generated tokens
score = torch.gather(logits, 1, generated_ids)   # [B, num_generated]
# Apply penalty
score = torch.where(score > 0, score / penalty, score * penalty)
# Scatter back
logits.scatter_(1, generated_ids, score)
```

### 6.6 Frequency and Presence Penalty (OpenAI-style)

Alternative to repetition penalty, used by OpenAI API:

```
logits[token_id] -= frequency_penalty * count[token_id] + presence_penalty * (1 if count[token_id] > 0 else 0)
```

- `frequency_penalty`: penalizes proportional to how many times the token appeared
- `presence_penalty`: flat penalty if the token appeared at all

### 6.7 Complete Sampling Pipeline

The order matters:

```python
def sample(logits, config):
    # 1. Repetition penalty (modify raw logits)
    logits = apply_repetition_penalty(logits, past_tokens, config.repetition_penalty)

    # 2. Temperature scaling
    logits = logits / config.temperature

    # 3. Top-k filtering (optional)
    if config.top_k > 0:
        logits = top_k_filter(logits, config.top_k)

    # 4. Top-p filtering (optional)
    if config.top_p < 1.0:
        logits = top_p_filter(logits, config.top_p)

    # 5. Min-p filtering (optional, alternative to top-k/top-p)
    if config.min_p > 0.0:
        logits = min_p_filter(logits, config.min_p)

    # 6. Convert to probabilities
    probs = softmax(logits, dim=-1)

    # 7. Sample
    if config.temperature == 0 or config.greedy:
        return torch.argmax(logits, dim=-1)
    else:
        return torch.multinomial(probs, num_samples=1)
```

---

## 7. Softmax and Numerical Stability

### Standard Softmax

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

**Problem**: `exp(x)` overflows for large x. `exp(89)` overflows float32. `exp(710)` overflows float64.

### Safe Softmax (subtract max)

```
m = max(x)
softmax(x)_i = exp(x_i - m) / sum_j(exp(x_j - m))
```

This is mathematically identical (the max cancels in numerator and denominator) but ensures all exponents are <= 0, preventing overflow.

### Three-Pass Standard Algorithm

```
Pass 1: m = max_j(x_j)                       # Find maximum
Pass 2: d = sum_j(exp(x_j - m))              # Compute denominator
Pass 3: softmax(x)_i = exp(x_i - m) / d      # Compute each output
```

Each pass reads the full input vector from memory. For attention with sequence length S, this means 3 reads of S elements.

### Two-Pass Online Softmax (Milakov & Gimelshein, 2018)

Fuse Pass 1 and Pass 2 into a single pass using running statistics:

**Pass 1 (fused max + denominator):**

Process elements one at a time (or in tiles):

```
Initialize: m_0 = -inf, d_0 = 0

For j = 1 to S:
    m_j = max(m_{j-1}, x_j)                              # Update running max
    d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j) # Rescale and accumulate
```

**The key insight**: When the max increases (m_j > m_{j-1}), all previously accumulated terms must be rescaled by `exp(m_{j-1} - m_j)`. This multiplicative correction is exact.

**Pass 2 (compute outputs):**

```
For i = 1 to S:
    softmax(x)_i = exp(x_i - m_S) / d_S
```

**Total: 2 passes instead of 3.** This saves one full read of the input vector.

### Tile-Based Online Softmax (for GPU implementation)

Process in tiles of size T (matching GPU shared memory):

```
Initialize: m = -inf, d = 0

For each tile [x_{t*T}, ..., x_{(t+1)*T - 1}]:
    m_tile = max(x in tile)
    m_new = max(m, m_tile)
    d = d * exp(m - m_new) + sum_j_in_tile(exp(x_j - m_new))
    m = m_new
```

### One-Pass Online Softmax (for FlashAttention)

FlashAttention extends this to fuse softmax with the attention value multiplication V:

```
Initialize: m = -inf, d = 0, o = 0   (o is the output accumulator, a vector of size D)

For each KV tile:
    S_tile = Q @ K_tile^T / sqrt(D)             # [S_new, tile_size] local scores
    m_tile = rowmax(S_tile)                       # [S_new]
    m_new = max(m, m_tile)                        # [S_new]

    # Rescale previous accumulator
    correction = exp(m - m_new)                   # [S_new]
    d = d * correction + rowsum(exp(S_tile - m_new))  # [S_new]
    o = o * correction.unsqueeze(-1) + exp(S_tile - m_new) @ V_tile  # [S_new, D]

    m = m_new

# Final normalization
output = o / d.unsqueeze(-1)                      # [S_new, D]
```

**This never materializes the full S_new x S attention matrix.** Memory usage: O(S_new * D) instead of O(S_new * S).

### Why This Matters for Attention

Standard attention writes an S x S intermediate matrix (the attention scores). For S = 32768:
- S x S * sizeof(bf16) = 32768^2 * 2 = ~2 GB per head per batch element

Online softmax in FlashAttention eliminates this entirely.

---

## 8. Memory Bandwidth and Roofline Analysis

### The Roofline Model

A computational kernel is characterized by its **arithmetic intensity** (AI):

```
Arithmetic Intensity = FLOPs / Bytes accessed
                     = (floating point operations) / (bytes read from + written to memory)
Unit: FLOPs/byte
```

The achievable performance is:

```
Performance = min(Peak_Compute, Peak_Bandwidth * AI)
```

Where:
- Peak_Compute = peak FLOPS of the device
- Peak_Bandwidth = peak memory bandwidth in bytes/sec
- The intersection point is the **ridge point**: AI_ridge = Peak_Compute / Peak_Bandwidth

For RTX 5080:
- Peak bf16 Tensor Core compute: ~228 TFLOPS (estimated, based on Blackwell architecture)
- Peak memory bandwidth: 960 GB/s
- Ridge point: 228,000 / 960 = **237.5 FLOPs/byte**

**If AI < 237.5 FLOPs/byte: memory-bound (bandwidth limited)**
**If AI > 237.5 FLOPs/byte: compute-bound (FLOPS limited)**

### Arithmetic Intensity of Matrix Multiplication

For C = A @ B where A is [M, K] and B is [K, N]:

```
FLOPs = 2 * M * N * K       (multiply-add for each output element)

Bytes = (M*K + K*N + M*N) * sizeof(dtype)     (read A, read B, write C)

AI = 2*M*N*K / ((M*K + K*N + M*N) * sizeof(dtype))
```

**Special cases:**

Matrix-matrix (large M, N, K):
```
AI ~ 2*M*N*K / (M*K + K*N) * 1/sizeof(dtype)
For square matrices (M=N=K): AI = 2K / (3 * sizeof(dtype))
For bf16, K=3584: AI = 7168 / 6 = 1195 FLOPs/byte >> 237.5 -> COMPUTE-BOUND
```

Matrix-vector (M=1 during decode):
```
AI = 2*K*N / ((K + K*N + N) * sizeof(dtype))
   ~ 2*K*N / (K*N * sizeof(dtype))    [K*N dominates]
   = 2 / sizeof(dtype)
For bf16: AI = 2/2 = 1 FLOPs/byte << 237.5 -> MEMORY-BOUND
```

**This is the fundamental reason decode is memory-bound**: with batch size 1, you read every weight but only do 2 FLOPs per weight element.

### Increasing Arithmetic Intensity During Decode

**Batching** is the primary tool:

For batch size B (B sequences decoded simultaneously):
```
A = [B, K], B_mat = [K, N]    # B input vectors share the same weight matrix

FLOPs = 2 * B * K * N
Bytes = (B*K + K*N + B*N) * sizeof(dtype)

AI = 2*B*K*N / ((B*K + K*N + B*N) * sizeof(dtype))
   ~ 2*B / sizeof(dtype)      [when K*N >> B*K, B*N]
```

For bf16: AI ~ B FLOPs/byte. To reach compute-bound: need B >= 238 simultaneous sequences.

### Arithmetic Intensity of Each Transformer Operation

| Operation | Type | AI (bf16, B=1) | Bound |
|-----------|------|-----------------|-------|
| **Prefill** | | | |
| QKV projection (S tokens) | [S,H]@[H,ND] | ~S FLOPs/byte | Compute if S >= 238 |
| QK^T per head | [S,D]@[D,S] | ~S/2 FLOPs/byte | Compute if S >= 475 |
| Attn@V per head | [S,S]@[S,D] | ~S/2 FLOPs/byte | Compute if S >= 475 |
| FFN projections | [S,H]@[H,I] | ~S FLOPs/byte | Compute if S >= 238 |
| RMSNorm | element-wise | ~0.75 FLOPs/byte | Always memory-bound |
| RoPE | element-wise | ~1.5 FLOPs/byte | Always memory-bound |
| **Decode** | | | |
| QKV projection | [1,H]@[H,ND] | ~1 FLOPs/byte | Memory-bound |
| QK^T per head | [1,D]@[D,S] | ~1 FLOPs/byte | Memory-bound |
| Attn@V per head | [1,S]@[S,D] | ~1 FLOPs/byte | Memory-bound |
| FFN projections | [1,H]@[H,I] | ~1 FLOPs/byte | Memory-bound |
| RMSNorm | element-wise | ~0.75 FLOPs/byte | Memory-bound |

### Theoretical Maximum Decode Speed

For a memory-bound decode step (B=1, bf16):

```
Total weight bytes per token = (sum of all weight matrices) * 2 bytes
For Qwen2.5-7B (bf16): ~15.2 GB

Theoretical max tokens/sec = Bandwidth / Model_Size
                           = 960 GB/s / 15.2 GB
                           = ~63 tokens/sec
```

With INT8 quantization (half the weight bytes):
```
= 960 / 7.6 = ~126 tokens/sec
```

With FP4 quantization (quarter the weight bytes):
```
= 960 / 3.8 = ~252 tokens/sec
```

These are hard upper bounds for B=1 decode. Real performance will be lower due to KV cache reads, activation memory, kernel launch overhead, and imperfect bandwidth utilization.

### KV Cache Memory Math

```
Per token, per layer:
    K: [K_heads * D] = 4 * 128 = 512 elements
    V: [K_heads * D] = 4 * 128 = 512 elements
    Total: 1024 elements * sizeof(dtype)
    BF16: 1024 * 2 = 2,048 bytes = 2 KB

Per token, all 28 layers:
    28 * 2,048 = 57,344 bytes = 56 KB

For sequence length S:
    56 KB * S

For S = 32768 (max positions):
    56 * 32768 = 1,835,008 KB = ~1.75 GB per sequence
```

With 16 GB VRAM and 15.2 GB model weights (bf16): only ~0.8 GB left for KV cache = ~14 tokens of context. This is why quantization is essential for the 7B model on 16 GB.

With INT8 weights (7.6 GB): 8.4 GB for KV cache = ~150K tokens of context (bf16 cache) or ~300K tokens (int8 cache).

### Roofline Plot Construction

```
X-axis: Arithmetic Intensity (FLOPs/byte), log scale
Y-axis: Achievable Performance (FLOPS), log scale

Two lines define the roofline:
1. Memory-bound region: Performance = Bandwidth * AI    (sloped line)
2. Compute-bound region: Performance = Peak_Compute     (horizontal line)

Ridge point: AI where the two lines meet
    AI_ridge = Peak_Compute / Bandwidth

Plot each kernel as a point:
    x = measured_AI = measured_FLOPs / measured_bytes
    y = measured_performance = measured_FLOPs / measured_time

Distance from roofline indicates optimization opportunity:
    - Far below sloped line: poor bandwidth utilization (fix: coalesced access, fewer accesses)
    - Far below horizontal line: poor compute utilization (fix: better tiling, higher occupancy)
```

---

## Appendix A: Complete Per-Token FLOPs for Qwen2.5-7B

### Prefill (per token, amortized over S tokens)

| Operation | FLOPs per token | Notes |
|-----------|----------------|-------|
| Embedding lookup | 0 | Table lookup, no compute |
| RMSNorm (input) | 3 * 3584 = 10,752 | Per layer |
| Q projection | 2 * 3584 * 3584 = 25,690,112 | Per layer |
| K projection | 2 * 3584 * 512 = 3,670,016 | Per layer |
| V projection | 2 * 3584 * 512 = 3,670,016 | Per layer |
| RoPE (Q) | 3 * 128 * 28 = 10,752 | Per layer |
| RoPE (K) | 3 * 128 * 4 = 1,536 | Per layer |
| QK^T | 2 * 28 * S * 128 = 7,168*S | Per layer, depends on S |
| Softmax | ~5 * 28 * S = 140*S | Per layer |
| Attn@V | 2 * 28 * S * 128 = 7,168*S | Per layer |
| Output projection | 2 * 3584 * 3584 = 25,690,112 | Per layer |
| RMSNorm (post-attn) | 10,752 | Per layer |
| Gate projection | 2 * 3584 * 18944 = 135,790,592 | Per layer |
| Up projection | 2 * 3584 * 18944 = 135,790,592 | Per layer |
| SiLU + multiply | 5 * 18944 = 94,720 | Per layer |
| Down projection | 2 * 18944 * 3584 = 135,790,592 | Per layer |
| **Per-layer total** | **~466.2M + 14,476*S** | |
| **All 28 layers** | **~13.05B + 405,328*S** | |
| Final RMSNorm | 10,752 | |
| LM head | 2 * 3584 * 152064 = 1,089,994,752 | ~1.09B |
| **Grand total** | **~14.14B + 405,328*S** | |

For S=1 (decode): ~14.14 billion FLOPs per token
For S=2048 (prefill): ~14.14B + 0.83B = ~14.97B FLOPs per token

### Total FLOPs for a Prefill of S=2048

~14.97B * 2048 = ~30.7 TFLOPS total

At 228 TFLOPS peak: ~30.7 / 228 = ~0.135 seconds = 135 ms (lower bound)

### Total Time for Decode (B=1)

Memory-bound: ~15.2 GB of weights read per token.
At 960 GB/s: 15.2 / 960 = ~15.8 ms per token = ~63 tokens/sec.

---

## Appendix B: Qwen2.5-7B Weight Tensor Names and Shapes

For implementation, here are the expected safetensor key names:

```
model.embed_tokens.weight                          [152064, 3584]

model.layers.{i}.input_layernorm.weight            [3584]
model.layers.{i}.self_attn.q_proj.weight           [3584, 3584]
model.layers.{i}.self_attn.q_proj.bias             [3584]
model.layers.{i}.self_attn.k_proj.weight           [512, 3584]
model.layers.{i}.self_attn.k_proj.bias             [512]
model.layers.{i}.self_attn.v_proj.weight           [512, 3584]
model.layers.{i}.self_attn.v_proj.bias             [512]
model.layers.{i}.self_attn.o_proj.weight           [3584, 3584]
model.layers.{i}.post_attention_layernorm.weight   [3584]
model.layers.{i}.mlp.gate_proj.weight              [18944, 3584]
model.layers.{i}.mlp.up_proj.weight                [18944, 3584]
model.layers.{i}.mlp.down_proj.weight              [3584, 18944]

model.norm.weight                                  [3584]
lm_head.weight                                     [152064, 3584]
```

Where `i` ranges from 0 to 27 (28 layers).

**Note**: Qwen2.5 uses biases on QKV projections (unlike Llama which has no biases). The output projection and FFN projections have no bias.

---

## Appendix C: Numerical Constants

| Constant | Value | Context |
|----------|-------|---------|
| sqrt(128) | 11.31370849898476 | Attention scaling factor |
| 1/sqrt(128) | 0.08838834764831845 | Multiply instead of divide |
| ln(2) | 0.6931471805599453 | For converting log bases |
| bf16 max | 3.3895e+38 | Overflow threshold |
| bf16 min positive | 9.1835e-41 | Underflow threshold |
| bf16 epsilon | 0.0078125 (2^-7) | Smallest representable diff near 1.0 |
| fp32 epsilon | 1.1920929e-7 (2^-23) | For numerical comparisons |

---

## Appendix D: Index of Notation

| Symbol | Meaning | Qwen2.5-7B Value |
|--------|---------|-------------------|
| B | Batch size | Variable |
| S | Total sequence length (including cache) | Up to 32768 |
| S_new | New tokens this forward pass | S (prefill) or 1 (decode) |
| H | Hidden dimension | 3584 |
| N | Number of query attention heads | 28 |
| K | Number of key/value attention heads | 4 |
| G | Group size (queries per KV head) = N/K | 7 |
| D | Head dimension = H/N | 128 |
| I | FFN intermediate dimension | 18944 |
| V | Vocabulary size | 152064 |
| L | Number of transformer layers | 28 |
| eps | RMSNorm epsilon | 1e-6 |
| theta | RoPE base | 1,000,000 |
