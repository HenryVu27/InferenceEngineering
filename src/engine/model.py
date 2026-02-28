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
    #   1. Compute variance: mean(x², dim=-1, keepdim=True)
    #   2. Compute rsqrt: 1 / sqrt(variance + eps)
    #   3. Normalize: x * rsqrt
    #   4. Scale: normalized * weight
    raise NotImplementedError


def rotary_embedding(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding (RoPE) to Q and K.

    RoPE encodes position by rotating pairs of dimensions. For head_dim=128,
    there are 64 frequency pairs. Each pair (x0, x1) is rotated by angle θ*pos.

    Frequencies: θ_i = 1 / (theta_base^(2i/head_dim)) for i in [0, 64)

    Args:
        q: Query tensor  [batch, seq_len, num_q_heads, head_dim]
        k: Key tensor    [batch, seq_len, num_kv_heads, head_dim]
        positions: Position indices [batch, seq_len]

    Returns:
        (q_rotated, k_rotated) — same shapes as inputs.
    """
    # TODO: Implement RoPE.
    #   1. Compute frequency pairs: freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    #   2. Compute angles: angles = positions.unsqueeze(-1) * freqs  → [batch, seq_len, 64]
    #   3. Build cos/sin: [batch, seq_len, 1, head_dim] (repeat each freq for the pair)
    #   4. Apply rotation: split q/k into even/odd, rotate using cos/sin
    #      q_rot = q_even * cos - q_odd * sin
    #      q_rot_odd = q_even * sin + q_odd * cos
    #      q_out = interleave(q_rot, q_rot_odd)
    #
    # Note: RoPE applies to full head_dim (128), 64 rotation pairs. Not partial.
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
    #   1. Expand KV heads: repeat_interleave(k, GQA_RATIO, dim=2) → [B, S, 28, 128]
    #      Same for v.
    #   2. Transpose to [B, heads, S, head_dim] for matmul
    #   3. Compute scores: Q @ K^T / sqrt(HEAD_DIM)   (scale = 1/11.3137 ≈ 1/sqrt(128))
    #   4. Apply causal mask: scores.masked_fill(mask == 0, float('-inf'))
    #   5. Softmax over last dim
    #   6. Compute output: scores @ V
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
    #   1. gate = x @ gate_weight.T        → [B, S, 18944]
    #   2. gate = SiLU(gate)               → x * sigmoid(x), or use torch.nn.functional.silu
    #   3. up = x @ up_weight.T            → [B, S, 18944]
    #   4. fused = gate * up               → element-wise multiply
    #   5. output = fused @ down_weight.T  → [B, S, 3584]
    raise NotImplementedError


# ─── Causal mask ────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """Create causal attention mask (upper-triangular = -inf).

    Args:
        seq_len: Sequence length.
        device: Target device.

    Returns:
        Boolean mask [1, 1, seq_len, seq_len]. True = attend, False = mask.
    """
    # TODO: Create lower-triangular boolean mask.
    # mask[i][j] = True if j <= i (can attend to current and past positions)
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
    #
    # 2. Build positions if not provided
    #    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    #
    # 3. Build causal mask
    #    mask = make_causal_mask(seq_len, device=device)
    #
    # 4. Run through all 28 layers
    #    for i in range(NUM_LAYERS):
    #        x = transformer_block(x, weights, i, positions, mask)
    #
    # 5. Final norm
    #    x = rmsnorm(x, weights["model.norm.weight"])
    #
    # 6. LM head (no bias, embeddings NOT tied)
    #    logits = x @ weights["lm_head.weight"].T  → [B, S, 152064]
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
    # tokens = prompt_tokens.copy()
    # for _ in range(max_new_tokens):
    #     input_ids = torch.tensor([tokens], device=device)     # [1, current_len]
    #     logits = forward(input_ids, weights)                   # [1, current_len, vocab]
    #     next_logits = logits[0, -1, :]                         # [vocab] — last position only
    #     next_token = sample_fn(next_logits)
    #     if next_token in EOS_IDS:
    #         break
    #     tokens.append(next_token)
    # return tokens
    #
    # NOTE: This recomputes ALL previous tokens every step. That's O(n²) and slow.
    # That's intentional for Phase 1 — Phase 2 fixes it with KV cache.
    raise NotImplementedError
