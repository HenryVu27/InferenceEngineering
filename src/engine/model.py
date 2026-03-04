"""Qwen2.5-7B forward pass — from raw weight tensors, no nn.Module.

Architecture (Qwen2.5-7B-Instruct):
  Hidden dim:     3,584          Layers:        28
  FFN dim:        18,944         Vocab:         152,064
  Q heads:        28             KV heads:      4
  Head dim:       128            GQA ratio:     7:1
  RoPE theta:     1,000,000     RMSNorm eps:   1e-6
  Attention bias: Q,K,V yes / O no
  MLP bias:       None
  Embeddings:     Not tied (separate embed_tokens and lm_head)

Weight tensor naming convention (per layer i):
  model.layers.{i}.input_layernorm.weight              [3584]
  model.layers.{i}.self_attn.q_proj.weight              [3584, 3584]
  model.layers.{i}.self_attn.q_proj.bias                [3584]
  model.layers.{i}.self_attn.k_proj.weight              [512, 3584]
  model.layers.{i}.self_attn.k_proj.bias                [512]
  model.layers.{i}.self_attn.v_proj.weight              [512, 3584]
  model.layers.{i}.self_attn.v_proj.bias                [512]
  model.layers.{i}.self_attn.o_proj.weight              [3584, 3584]
  model.layers.{i}.post_attention_layernorm.weight      [3584]
  model.layers.{i}.mlp.gate_proj.weight                 [18944, 3584]
  model.layers.{i}.mlp.up_proj.weight                   [18944, 3584]
  model.layers.{i}.mlp.down_proj.weight                 [3584, 18944]

Global weights:
  model.embed_tokens.weight                             [152064, 3584]
  model.norm.weight                                     [3584]
  lm_head.weight                                        [152064, 3584]
"""

from pathlib import Path

import torch

# ─── Model config ───────────────────────────────────────────────────────────
HIDDEN_DIM = 3584
NUM_LAYERS = 28
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
FFN_DIM = 18944
VOCAB_SIZE = 152064
ROPE_THETA = 1_000_000.0
RMSNORM_EPS = 1e-6
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 7


# ─── Building block operations ─────────────────────────────────────────────
# These are stepping stones. Implement these first to build intuition,
# then tackle the full model operations below.

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Subtracting max(x) before exp() prevents overflow with large values.
    This is mathematically equivalent to the naive version but won't produce
    NaN/Inf when inputs are large (e.g., attention scores can reach ~100).

    Args:
        x: Input tensor of any shape.
        dim: Dimension to apply softmax over.

    Returns:
        Probability distribution along dim (sums to 1).
    """
    # TODO: Implement numerically stable softmax.
    #   1. max_val = x.max(dim=dim, keepdim=True).values
    #   2. exp_x = torch.exp(x - max_val)      # subtract max for stability
    #   3. return exp_x / exp_x.sum(dim=dim, keepdim=True)
    #
    # Why subtract max? Without it, exp(1000) = Inf. With it, exp(1000 - 1002) = exp(-2) = 0.13.
    # The subtraction doesn't change the result because it cancels in the numerator/denominator.
    #
    # After implementing, compare: softmax(x, dim=-1) vs torch.softmax(x, dim=-1)
    raise NotImplementedError


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).

    Also called "swish". Used inside SwiGLU as the gating activation.
    - silu(0) = 0 * 0.5 = 0
    - silu(x) ≈ x for large positive x (sigmoid → 1)
    - silu(x) ≈ 0 for large negative x (sigmoid → 0)
    - Smooth, non-monotonic — slightly negative for x ≈ -1.28

    Args:
        x: Input tensor of any shape.

    Returns:
        Activated tensor, same shape.
    """
    # TODO: Implement SiLU.
    #   sigmoid = 1 / (1 + torch.exp(-x))    — or use torch.sigmoid(x)
    #   return x * sigmoid
    #
    # After implementing, compare: silu(x) vs torch.nn.functional.silu(x)
    raise NotImplementedError


def simple_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention (same head count for Q, K, V).

    This is the basic attention mechanism WITHOUT GQA. Q, K, V all have the
    same number of heads. Implement this first to understand the core algorithm,
    then move to attention() which adds GQA head expansion.

    scores = softmax((Q @ K^T) / sqrt(head_dim) + mask)
    output = scores @ V

    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_heads, head_dim]  (same num_heads as q)
        v: [batch, seq_len, num_heads, head_dim]  (same num_heads as q)
        mask: Optional [1, 1, seq_len, seq_len]. 0 = attend, -inf = block.
              Or boolean: True = attend, False = block.

    Returns:
        [batch, seq_len, num_heads, head_dim]
    """
    # TODO: Implement basic scaled dot-product attention.
    #   1. Transpose to [B, heads, S, D] for batched matmul
    #      q = q.transpose(1, 2)  (same for k, v)
    #
    #   2. Compute attention scores: Q @ K^T / sqrt(head_dim)
    #      scale = 1.0 / (head_dim ** 0.5)
    #      scores = (q @ k.transpose(-2, -1)) * scale   → [B, H, S, S]
    #
    #   3. Apply mask (if provided)
    #      If boolean mask: scores = scores.masked_fill(mask == False, float('-inf'))
    #      If additive mask: scores = scores + mask
    #
    #   4. Softmax over the last dim (key positions)
    #      scores = torch.softmax(scores, dim=-1)      — or use your softmax()
    #
    #   5. Weighted sum: scores @ V
    #      output = scores @ v                          → [B, H, S, D]
    #
    #   6. Transpose back: [B, H, S, D] → [B, S, H, D]
    #      output = output.transpose(1, 2)
    #
    # Once this works, attention() below just adds KV head expansion before step 1.
    raise NotImplementedError


# ─── Weight loading ─────────────────────────────────────────────────────────

def load_weights(model_dir: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Load all weights from safetensors shards into a flat dict.

    Qwen2.5-7B has 4 safetensors shards. The index file
    model.safetensors.index.json maps tensor names → shard filenames.

    Args:
        model_dir: Path to model directory with safetensors files.
        device: Target device ("cuda" or "cpu").
        dtype: Target dtype (torch.bfloat16 for Phase 1).

    Returns:
        Dict mapping tensor name → tensor on device with dtype.
        Example: {"model.embed_tokens.weight": tensor([152064, 3584])}
    """
    # TODO: 1. Read model.safetensors.index.json to get the weight_map
    #       2. Load each unique shard with safetensors.torch.load_file()
    #       3. Collect all tensors into one dict, cast to dtype, move to device
    #       4. Validate: should have 323 tensors total
    # Hint: from safetensors.torch import load_file
    raise NotImplementedError("Load safetensors shards")


# ─── Individual operations ──────────────────────────────────────────────────

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMSNORM_EPS) -> torch.Tensor:
    """RMSNorm: x_norm = (x / sqrt(mean(x²) + eps)) * weight

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: Learnable scale [hidden_dim]
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor, same shape as x.
    """
    # TODO: Implement RMSNorm.
    #   1. Save input dtype: input_dtype = x.dtype
    #   2. Upcast to float32: x = x.float()
    #      IMPORTANT: bfloat16 has only 8 bits of mantissa. Computing mean(x²)
    #      in bfloat16 with large values loses precision and causes divergence
    #      from HuggingFace. Always compute in float32.
    #   3. Compute variance: variance = x.pow(2).mean(dim=-1, keepdim=True)
    #   4. Normalize: x = x * torch.rsqrt(variance + eps)
    #   5. Scale and cast back: return (x * weight).to(input_dtype)
    raise NotImplementedError


def rotary_embedding(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to Q and K.

    RoPE encodes position by rotating pairs of dimensions. For head_dim=128,
    there are 64 frequency pairs. Each pair is rotated by angle θ*pos.

    Frequencies: θ_i = 1 / (theta_base^(2i/head_dim)) for i in [0, 64)

    IMPORTANT — RoPE variants:
      - Traditional (interleaved): pairs are (dim0, dim1), (dim2, dim3), ...
        Uses x[..., 0::2] and x[..., 1::2].
      - Split-half (HuggingFace): pairs are (dim0, dim64), (dim1, dim65), ...
        Uses x[..., :half] and x[..., half:].

    Qwen2.5 (and Llama, Gemma) use the SPLIT-HALF variant in HuggingFace.
    The split-half approach uses a rotate_half() helper:
      rotate_half(x) = [-x[..., half:], x[..., :half]]

    Then: x_rotated = x * cos + rotate_half(x) * sin

    Args:
        q: Query tensor  [batch, seq_len, num_q_heads, head_dim]
        k: Key tensor    [batch, seq_len, num_kv_heads, head_dim]
        positions: Position indices [batch, seq_len]

    Returns:
        (q_rotated, k_rotated) — same shapes and dtype as inputs.
    """
    # TODO: Implement RoPE (split-half variant to match HuggingFace).
    #
    #   1. Compute inverse frequencies:
    #      inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    #      → shape [64] — one frequency per dimension pair
    #
    #   2. Compute angles:
    #      angles = positions.unsqueeze(-1).float() * inv_freq   → [B, S, 64]
    #      These are the rotation angles for each position and frequency pair.
    #
    #   3. Build cos/sin for full head_dim (duplicate for split-half):
    #      emb = torch.cat([angles, angles], dim=-1)             → [B, S, 128]
    #      cos = torch.cos(emb).unsqueeze(2)                     → [B, S, 1, 128]
    #      sin = torch.sin(emb).unsqueeze(2)                     → [B, S, 1, 128]
    #      The unsqueeze(2) broadcasts across heads.
    #
    #   4. Define rotate_half helper:
    #      def rotate_half(x):
    #          x1 = x[..., : x.shape[-1] // 2]    # first 64 dims
    #          x2 = x[..., x.shape[-1] // 2 :]    # last 64 dims
    #          return torch.cat([-x2, x1], dim=-1)
    #
    #   5. Apply rotation:
    #      q_rot = q.float() * cos + rotate_half(q.float()) * sin
    #      k_rot = k.float() * cos + rotate_half(k.float()) * sin
    #
    #   6. Cast back to input dtype:
    #      return q_rot.to(q.dtype), k_rot.to(k.dtype)
    #
    # Why split-half instead of interleaved?
    # Both are mathematically equivalent — they define different dimension pairings.
    # HuggingFace Qwen2 uses split-half (rotate_half), so we match it for correctness.
    # See: docs/phase1-guide.md "RoPE Variants" section for both implementations.
    raise NotImplementedError


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped-Query Attention (GQA) with causal mask.

    Qwen2.5-7B uses 28 Q heads and 4 KV heads (ratio 7:1).
    KV heads must be expanded to match Q heads before computing attention.

    Args:
        q: [batch, seq_len, num_q_heads, head_dim]   (28 heads)
        k: [batch, seq_len, num_kv_heads, head_dim]   (4 heads)
        v: [batch, seq_len, num_kv_heads, head_dim]   (4 heads)
        mask: Optional causal mask [1, 1, seq_len, seq_len] or None.

    Returns:
        Attention output [batch, seq_len, num_q_heads, head_dim]
    """
    # TODO: Implement GQA.
    #
    #   Option A — repeat_interleave (simple, recommended first):
    #   1. Expand KV heads: k = k.repeat_interleave(GQA_RATIO, dim=2) → [B, S, 28, 128]
    #      Same for v. This copies each KV head 7 times to match Q head count.
    #      Memory cost: allocates 7x the KV memory.
    #
    #   Option B — broadcast (memory-efficient alternative):
    #   1. Reshape Q to group heads: q = q.view(B, S, NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)
    #   2. Reshape K/V to add broadcast dim: k = k.unsqueeze(3)  → [B, S, 4, 1, 128]
    #   3. Attention scores broadcast: [B, 4, 7, S, D] @ [B, 4, 1, D, S] → [B, 4, 7, S, S]
    #   4. No extra memory allocated — PyTorch broadcasts without copying.
    #   Try this after Option A works, to learn the optimization.
    #
    #   After KV expansion (or broadcast setup):
    #   2. Transpose to [B, heads, S, head_dim] for batched matmul
    #   3. Compute scores: Q @ K^T / sqrt(HEAD_DIM)
    #      scale = 1.0 / (HEAD_DIM ** 0.5)   — for head_dim=128: scale ≈ 0.0884
    #   4. Apply causal mask: scores.masked_fill(mask == False, float('-inf'))
    #      Scores for future positions become -inf → softmax gives them 0 weight.
    #   5. Softmax over last dim (in float32 for stability):
    #      scores = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    #   6. Compute output: scores @ V    → [B, heads, S, head_dim]
    #   7. Transpose back to [B, S, heads, head_dim]
    raise NotImplementedError


def swiglu_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU feed-forward network.

    SwiGLU(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
    SiLU(x) = x * sigmoid(x)

    Three projections, no bias. FFN dim = 18,944 (5.29x expansion).

    Args:
        x: Input [batch, seq_len, hidden_dim]
        gate_weight: [ffn_dim, hidden_dim]  (18944, 3584)
        up_weight:   [ffn_dim, hidden_dim]  (18944, 3584)
        down_weight: [hidden_dim, ffn_dim]  (3584, 18944)

    Returns:
        Output [batch, seq_len, hidden_dim]
    """
    # TODO: Implement SwiGLU.
    #   1. gate = x @ gate_weight.T        → [B, S, ffn_dim]
    #   2. gate = silu(gate)               → use YOUR silu() function (not torch's)
    #   3. up = x @ up_weight.T            → [B, S, ffn_dim]
    #   4. fused = gate * up               → element-wise gating
    #   5. output = fused @ down_weight.T  → [B, S, hidden_dim]
    #
    # Why SwiGLU instead of standard FFN (just up → activation → down)?
    # Standard FFN: down(relu(up(x)))           — 2 projections
    # SwiGLU:       down(silu(gate(x)) * up(x)) — 3 projections but better quality
    # The gate and up projections create a gating mechanism: the gate "decides"
    # how much of each up-projected feature to keep. This is why FFN dim is 5.29x
    # instead of the usual 4x — the third projection costs extra parameters.
    # Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    raise NotImplementedError


# ─── Causal mask ────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, key_len: int | None = None, device: str = "cuda") -> torch.Tensor:
    """Create causal attention mask (upper-triangular = blocked).

    Supports rectangular masks for KV-cache decode: query_len != key_len.
    When key_len > seq_len, the mask allows each query to see further back
    into the KV cache history.

    Args:
        seq_len: Query sequence length (L).
        key_len: Key sequence length (S). Defaults to seq_len if None.
        device: Target device.

    Returns:
        Boolean mask [1, 1, seq_len, key_len]. True = attend, False = block.
        mask[i][j] = True if j <= i + (key_len - seq_len)
    """
    # TODO: Create causal mask supporting both square (L=S) and rectangular (L<S).
    #
    #   For square (standard prefill):
    #     torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    #
    #   For rectangular (KV cache decode, L < S):
    #     offset = key_len - seq_len
    #     mask = torch.tril(torch.ones(seq_len, key_len, dtype=torch.bool, device=device), diagonal=offset)
    #     The diagonal parameter shifts the triangle right by `offset` positions.
    #
    #   Then unsqueeze to [1, 1, L, S] for broadcasting across batch and heads.
    raise NotImplementedError


# ─── Transformer block ──────────────────────────────────────────────────────

def transformer_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer_idx: int,
    positions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single transformer layer: attention + FFN with residual connections.

    Flow:
        residual = x
        x = rmsnorm(x, input_layernorm)
        x = attention(q_proj(x), k_proj(x), v_proj(x)) → o_proj
        x = residual + x
        residual = x
        x = rmsnorm(x, post_attention_layernorm)
        x = swiglu_ffn(x)
        x = residual + x

    Args:
        x: Input hidden states [batch, seq_len, hidden_dim]
        weights: Full weight dict (this function indexes into it by layer_idx).
        layer_idx: Which layer (0-27).
        positions: Position indices [batch, seq_len].
        mask: Causal mask [1, 1, seq_len, seq_len].

    Returns:
        Output hidden states [batch, seq_len, hidden_dim]
    """
    prefix = f"model.layers.{layer_idx}"

    # TODO: Implement the full transformer block.
    #
    # 1. Pre-attention norm
    #    residual = x
    #    x = rmsnorm(x, weights[f"{prefix}.input_layernorm.weight"])
    #
    # 2. QKV projections (NOTE: Q, K, V have bias! O does not.)
    #    q = x @ weights[f"{prefix}.self_attn.q_proj.weight"].T + weights[f"{prefix}.self_attn.q_proj.bias"]
    #    k = x @ weights[f"{prefix}.self_attn.k_proj.weight"].T + weights[f"{prefix}.self_attn.k_proj.bias"]
    #    v = x @ weights[f"{prefix}.self_attn.v_proj.weight"].T + weights[f"{prefix}.self_attn.v_proj.bias"]
    #
    # 3. Reshape for multi-head attention
    #    q: [B, S, 3584] → [B, S, 28, 128]
    #    k: [B, S, 512]  → [B, S, 4, 128]
    #    v: [B, S, 512]  → [B, S, 4, 128]
    #
    # 4. Apply RoPE to q and k
    #    q, k = rotary_embedding(q, k, positions)
    #
    # 5. Compute attention
    #    attn_out = attention(q, k, v, mask)  → [B, S, 28, 128]
    #
    # 6. Reshape back and output projection (no bias!)
    #    attn_out: [B, S, 28, 128] → [B, S, 3584]
    #    attn_out = attn_out @ weights[f"{prefix}.self_attn.o_proj.weight"].T
    #
    # 7. Residual connection
    #    x = residual + attn_out
    #
    # 8. Post-attention norm + FFN
    #    residual = x
    #    x = rmsnorm(x, weights[f"{prefix}.post_attention_layernorm.weight"])
    #    x = swiglu_ffn(x, gate, up, down weights)
    #
    # 9. Residual connection
    #    x = residual + x
    #
    # return x
    raise NotImplementedError


# ─── Full forward pass ──────────────────────────────────────────────────────

def forward(
    token_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full forward pass: tokens → logits.

    Flow:
        1. Embedding lookup
        2. 28x transformer blocks
        3. Final RMSNorm
        4. LM head projection → logits

    Args:
        token_ids: Input token IDs [batch, seq_len]
        weights: Full weight dict from load_weights().
        positions: Optional position indices [batch, seq_len].
                   If None, uses 0..seq_len-1.

    Returns:
        Logits [batch, seq_len, vocab_size] (152064)
    """
    batch, seq_len = token_ids.shape
    device = token_ids.device

    # TODO: Implement the full forward pass.
    #
    # 1. Embedding lookup
    #    x = weights["model.embed_tokens.weight"][token_ids]  → [B, S, 3584]
    #    This is just a table lookup: token_id 42 → row 42 of the embedding matrix.
    #
    # 2. Build positions if not provided
    #    if positions is None:
    #        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    #
    # 3. Build causal mask (only needed when seq_len > 1)
    #    Optimization: for single-token decode, masking is unnecessary since there's
    #    only one position. But for Phase 1 (no KV cache), we always process full
    #    sequences, so always create the mask.
    #    mask = make_causal_mask(seq_len, device=device)
    #
    # 4. Run through all 28 layers
    #    for i in range(NUM_LAYERS):
    #        x = transformer_block(x, weights, i, positions, mask)
    #
    # 5. Final norm
    #    x = rmsnorm(x, weights["model.norm.weight"])
    #
    # 6. LM head (no bias, embeddings NOT tied in Qwen2.5-7B)
    #    logits = x @ weights["lm_head.weight"].T  → [B, S, 152064]
    #    Note: some smaller Qwen2 models tie embed_tokens and lm_head weights.
    #    Qwen2.5-7B does NOT — it has a separate lm_head.weight.
    #
    # return logits
    raise NotImplementedError


# ─── Autoregressive generation ──────────────────────────────────────────────

def generate(
    prompt_tokens: list[int],
    weights: dict[str, torch.Tensor],
    max_new_tokens: int = 100,
    sample_fn=None,
    device: str = "cuda",
) -> list[int]:
    """Generate tokens autoregressively (no KV cache — Phase 1).

    This is the naive version: every new token recomputes the entire sequence.
    Phase 2 adds KV cache to avoid this redundant computation.

    Args:
        prompt_tokens: List of prompt token IDs.
        weights: Full weight dict.
        max_new_tokens: Maximum tokens to generate.
        sample_fn: Sampling function (token_logits → token_id). Defaults to greedy.
        device: Target device.

    Returns:
        Full sequence: prompt_tokens + generated tokens.
    """
    from .sampler import greedy  # avoid circular import

    if sample_fn is None:
        sample_fn = greedy

    # TODO: Implement naive autoregressive generation.
    #
    # This has two logical phases (even though Phase 1 doesn't optimize them separately):
    #   PREFILL: Process the full prompt to get the first next-token prediction.
    #   DECODE:  Generate one token at a time, appending to the sequence.
    #
    # In Phase 1 (no KV cache), both phases look identical — we always re-process
    # the entire sequence. In Phase 2, prefill processes the prompt in one shot,
    # and decode only processes the new token (using cached K/V from previous steps).
    #
    # tokens = prompt_tokens.copy()
    # for _ in range(max_new_tokens):
    #     input_ids = torch.tensor([tokens], device=device)     # [1, current_len]
    #     logits = forward(input_ids, weights)                   # [1, current_len, vocab]
    #     next_logits = logits[0, -1, :]                         # [vocab] — last position only
    #
    #     # Numerical stability: subtract logsumexp before sampling.
    #     # logsumexp = log(sum(exp(logits))). Subtracting it converts logits to
    #     # log-probabilities. This prevents overflow in softmax during sampling.
    #     # next_logits = next_logits - torch.logsumexp(next_logits, dim=-1, keepdim=True)
    #
    #     next_token = sample_fn(next_logits)
    #
    #     # Qwen2.5 has TWO end-of-sequence tokens:
    #     #   151645 = <|im_end|>     (end of assistant turn in ChatML)
    #     #   151643 = <|endoftext|>  (end of text)
    #     if next_token in (151645, 151643):
    #         break
    #     tokens.append(next_token)
    # return tokens
    #
    # NOTE: This recomputes ALL previous tokens every step. That's O(n²) in FLOPs
    # and painfully slow. For a 100-token generation, the last step processes all
    # 100+ tokens just to predict token 101. Phase 2 fixes this with KV cache.
    raise NotImplementedError
