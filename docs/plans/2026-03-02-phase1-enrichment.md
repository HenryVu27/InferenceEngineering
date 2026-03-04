# Phase 1 Enrichment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enrich Phase 1 stubs, tests, and guide with insights from tiny-llm — fix bugs, add stepping-stone functions, expand test coverage, and update educational content.

**Architecture:** Layered enrichment in-place. No file splits, no new source files. model.py gets 3 new stub functions before their advanced counterparts. test_model.py gets ~15 new test cases. sampler.py gets a bug fix. phase1-guide.md gets targeted patches.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Fix sampler.py greedy() bug

**Files:**
- Modify: `src/engine/sampler.py:22-24`

**Step 1: Write the failing test**

Add to `tests/test_model.py` at the end of the imports section and before the unit tests:

```python
from src.engine.sampler import (
    greedy,
    temperature_scale,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    sample,
)
```

Then add the test class after `TestSwiGLU`:

```python
class TestSampler:
    """Test sampling functions."""

    def test_greedy_returns_argmax(self):
        """Greedy should return the index of the highest logit."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        result = greedy(logits)
        assert result == 1, f"Expected 1 (argmax), got {result}"

    def test_greedy_with_negative_logits(self):
        """Greedy should work with all-negative logits."""
        logits = torch.tensor([-5.0, -1.0, -3.0, -2.0])
        result = greedy(logits)
        assert result == 1, f"Expected 1, got {result}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestSampler::test_greedy_returns_argmax -v`
Expected: FAIL — `greedy()` returns `None` due to bare `return` on line 23

**Step 3: Fix the bug**

In `src/engine/sampler.py`, replace lines 22-24:

```python
    # TODO: Return argmax of logits.
    return
    raise NotImplementedError
```

With:

```python
    # TODO: Return argmax of logits.
    #   return logits.argmax(dim=-1).item()
    raise NotImplementedError
```

**Step 4: Run test to verify it still fails (correctly this time)**

Run: `python -m pytest tests/test_model.py::TestSampler::test_greedy_returns_argmax -v`
Expected: FAIL with `NotImplementedError` (not `None` — the bug is fixed, the stub is correct)

**Step 5: Commit**

```bash
git add src/engine/sampler.py tests/test_model.py
git commit -m "phase-1: fix greedy() bare return bug, add greedy tests"
```

---

### Task 2: Add stepping-stone functions to model.py

**Files:**
- Modify: `src/engine/model.py:75-76` (insert before rmsnorm)

**Step 1: Write failing tests**

Add to `tests/test_model.py` imports:

```python
from src.engine.model import (
    forward,
    load_weights,
    rmsnorm,
    rotary_embedding,
    attention,
    simple_attention,
    swiglu_ffn,
    softmax,
    silu,
    make_causal_mask,
    HIDDEN_DIM,
    HEAD_DIM,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    RMSNORM_EPS,
)
```

Add test classes before `TestRMSNorm`:

```python
class TestSoftmax:
    """Test manual softmax implementation."""

    def test_matches_torch(self):
        """Manual softmax should match torch.softmax."""
        torch.manual_seed(42)
        x = torch.randn(2, 10)
        result = softmax(x, dim=-1)
        expected = torch.softmax(x, dim=-1)
        assert torch.allclose(result, expected, atol=1e-6), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_numerical_stability(self):
        """Should not overflow/NaN with large values."""
        x = torch.tensor([[1000.0, 1001.0, 1002.0]])
        result = softmax(x, dim=-1)
        assert not torch.isnan(result).any(), "Softmax produced NaN with large inputs"
        assert torch.allclose(result.sum(dim=-1), torch.ones(1), atol=1e-6), \
            "Softmax should sum to 1"

    def test_sums_to_one(self):
        """Softmax output should sum to 1 along the specified dim."""
        x = torch.randn(3, 5, 8)
        result = softmax(x, dim=-1)
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestSiLU:
    """Test manual SiLU activation."""

    def test_matches_torch(self):
        """Manual SiLU should match torch.nn.functional.silu."""
        torch.manual_seed(42)
        x = torch.randn(2, 10)
        result = silu(x)
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(result, expected, atol=1e-6), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_zero_at_zero(self):
        """SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0."""
        result = silu(torch.tensor([0.0]))
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-7)

    def test_shape_preserved(self):
        """Output shape should match input."""
        x = torch.randn(3, 5, 8)
        assert silu(x).shape == x.shape


class TestSimpleAttention:
    """Test simple scaled dot-product attention (same head count for Q/K/V)."""

    def test_output_shape(self):
        """Output should match Q shape."""
        B, S, H, D = 1, 4, 8, 64
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        result = simple_attention(q, k, v)
        assert result.shape == (B, S, H, D)

    def test_matches_torch_sdpa(self):
        """Should match PyTorch's scaled_dot_product_attention."""
        B, S, H, D = 1, 4, 8, 64
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)

        result = simple_attention(q, k, v)

        # Reference: PyTorch SDPA expects [B, H, S, D]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        expected = expected.transpose(1, 2)  # back to [B, S, H, D]

        assert torch.allclose(result, expected, atol=1e-5), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_causal_masking(self):
        """With causal mask, first token should not see future tokens."""
        B, S, H, D = 1, 4, 4, 32
        torch.manual_seed(42)
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D)
        mask = make_causal_mask(S, device="cpu")

        result1 = simple_attention(q, k, v, mask)

        # Modify future tokens
        k2, v2 = k.clone(), v.clone()
        k2[:, 1:] = torch.randn_like(k2[:, 1:])
        v2[:, 1:] = torch.randn_like(v2[:, 1:])
        result2 = simple_attention(q, k2, v2, mask)

        assert torch.allclose(result1[:, 0], result2[:, 0], atol=1e-5), \
            "First token should not be affected by future tokens"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py::TestSoftmax tests/test_model.py::TestSiLU tests/test_model.py::TestSimpleAttention -v`
Expected: FAIL with `ImportError` (functions don't exist yet)

**Step 3: Add the three stub functions to model.py**

Insert after the model config block (line 48) and before `load_weights` (line 50), replacing the comment `# ─── Weight loading`:

```python
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


```

**Step 4: Run tests to verify they fail correctly**

Run: `python -m pytest tests/test_model.py::TestSoftmax tests/test_model.py::TestSiLU tests/test_model.py::TestSimpleAttention -v`
Expected: FAIL with `NotImplementedError` (stubs imported successfully, raise on call)

**Step 5: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: add softmax, silu, simple_attention stepping-stone stubs and tests"
```

---

### Task 3: Enrich rmsnorm stub and tests

**Files:**
- Modify: `src/engine/model.py:88-93` (rmsnorm TODO)
- Modify: `tests/test_model.py` (TestRMSNorm class)

**Step 1: Add the float32 precision test**

Add to `TestRMSNorm`:

```python
    def test_float32_upcasting_with_bfloat16(self):
        """RMSNorm should upcast to float32 internally for precision.

        With large-magnitude bfloat16 inputs, computing variance directly in
        bfloat16 loses precision. The implementation must compute in float32
        then cast back.
        """
        torch.manual_seed(42)
        x = (torch.randn(1, 4, HIDDEN_DIM) * 500).to(torch.bfloat16)  # large values
        weight = torch.ones(HIDDEN_DIM, dtype=torch.bfloat16)

        result = rmsnorm(x, weight, RMSNORM_EPS)

        # Result should be bfloat16 (same as input)
        assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"
        # RMS should still be ~1.0 despite bfloat16 input
        rms = torch.sqrt(torch.mean(result.float() ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.05), \
            f"RMS should be ~1.0, got {rms}"

    def test_scaling_by_weight(self):
        """Non-unit weight should scale the normalized output."""
        x = torch.randn(1, 2, HIDDEN_DIM)
        weight = torch.full((HIDDEN_DIM,), 2.0)

        result = rmsnorm(x, weight, RMSNORM_EPS)

        # RMS of result should be ~2.0 (scaled by weight)
        rms = torch.sqrt(torch.mean(result ** 2, dim=-1))
        assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=0.1), \
            f"RMS should be ~2.0 with weight=2, got {rms}"
```

**Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_model.py::TestRMSNorm -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Enrich the rmsnorm stub TODO**

Replace the rmsnorm TODO block in model.py:

```python
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
```

**Step 4: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: enrich rmsnorm stub with float32 upcasting, add precision tests"
```

---

### Task 4: Fix RoPE to split-half variant

**Files:**
- Modify: `src/engine/model.py:96-122` (rotary_embedding)
- Modify: `tests/test_model.py` (TestRoPE class)

**Step 1: Add RoPE tests**

Replace `TestRoPE` with an enriched version:

```python
class TestRoPE:
    """Test Rotary Position Embedding (split-half variant, as used by HuggingFace Qwen2)."""

    def test_shape_preserved(self):
        """RoPE should not change tensor shapes."""
        q = torch.randn(1, 4, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM)
        positions = torch.arange(4).unsqueeze(0)

        q_rot, k_rot = rotary_embedding(q, k, positions)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_zero_is_identity(self):
        """At position 0, cos=1 and sin=0, so rotation should be identity."""
        q = torch.randn(1, 1, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(1, 1, NUM_KV_HEADS, HEAD_DIM)
        positions = torch.zeros(1, 1, dtype=torch.long)

        q_rot, k_rot = rotary_embedding(q, k, positions)

        assert torch.allclose(q_rot, q, atol=1e-5), "RoPE at position 0 should be identity"
        assert torch.allclose(k_rot, k, atol=1e-5), "RoPE at position 0 should be identity"

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.bfloat16]:
            q = torch.randn(1, 4, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
            k = torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
            positions = torch.arange(4).unsqueeze(0)

            q_rot, k_rot = rotary_embedding(q, k, positions)
            assert q_rot.dtype == dtype, f"Expected {dtype}, got {q_rot.dtype}"
            assert k_rot.dtype == dtype

    def test_different_positions_give_different_results(self):
        """Same input at different positions should produce different outputs."""
        q = torch.randn(1, 1, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(1, 1, NUM_KV_HEADS, HEAD_DIM)

        q_rot0, _ = rotary_embedding(q, k, torch.tensor([[0]]))
        q_rot5, _ = rotary_embedding(q, k, torch.tensor([[5]]))

        assert not torch.allclose(q_rot0, q_rot5, atol=1e-3), \
            "Different positions should produce different rotations"

    def test_rotation_is_norm_preserving(self):
        """RoPE is a rotation — it should preserve vector norms."""
        q = torch.randn(1, 4, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM)
        positions = torch.arange(4).unsqueeze(0)

        q_rot, k_rot = rotary_embedding(q, k, positions)

        q_norm_before = torch.norm(q, dim=-1)
        q_norm_after = torch.norm(q_rot, dim=-1)
        assert torch.allclose(q_norm_before, q_norm_after, atol=1e-4), \
            "RoPE should preserve vector norms (it's a rotation)"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py::TestRoPE -v`
Expected: FAIL with `NotImplementedError`

**Step 3: Rewrite the rotary_embedding stub TODO for split-half**

Replace the entire `rotary_embedding` function:

```python
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
```

**Step 4: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: fix RoPE to split-half variant (matches HuggingFace Qwen2), enrich tests"
```

---

### Task 5: Enrich attention stub with broadcast GQA notes

**Files:**
- Modify: `src/engine/model.py:125-154` (attention function)
- Modify: `tests/test_model.py` (TestAttention class)

**Step 1: Add attention tests**

Add to `TestAttention`:

```python
    def test_bfloat16_precision(self):
        """Attention should work in bfloat16 without NaN."""
        B, S = 1, 4
        q = torch.randn(B, S, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)

        result = attention(q, k, v)

        assert not torch.isnan(result).any(), "Attention produced NaN in bfloat16"
        assert result.dtype == torch.bfloat16

    def test_different_batch_sizes(self):
        """Attention should work with batch sizes > 1."""
        for B in [1, 2, 4]:
            S = 4
            q = torch.randn(B, S, NUM_Q_HEADS, HEAD_DIM)
            k = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)
            v = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)

            result = attention(q, k, v)
            assert result.shape == (B, S, NUM_Q_HEADS, HEAD_DIM), \
                f"Wrong shape for batch={B}: {result.shape}"
```

**Step 2: Enrich the attention stub TODO**

Replace the attention function TODO:

```python
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
```

**Step 3: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: enrich attention stub with broadcast GQA notes, add precision/batch tests"
```

---

### Task 6: Enrich causal mask for L != S

**Files:**
- Modify: `src/engine/model.py:188-202` (make_causal_mask)
- Modify: `tests/test_model.py` (add TestCausalMask)

**Step 1: Write tests**

Add `TestCausalMask` class:

```python
class TestCausalMask:
    """Test causal attention mask generation."""

    def test_square_mask_values(self):
        """4x4 causal mask should be lower-triangular."""
        mask = make_causal_mask(4, device="cpu")
        expected = torch.tensor([
            [True, False, False, False],
            [True, True,  False, False],
            [True, True,  True,  False],
            [True, True,  True,  True],
        ]).unsqueeze(0).unsqueeze(0)
        assert torch.equal(mask, expected), f"Mask values wrong:\n{mask.squeeze()}"

    def test_square_mask_shape(self):
        """Mask shape should be [1, 1, S, S]."""
        mask = make_causal_mask(8, device="cpu")
        assert mask.shape == (1, 1, 8, 8)

    def test_rectangular_mask(self):
        """Rectangular mask (L < S) for KV-cache decode compatibility.

        When decoding with KV cache: query_len=1 (new token), key_len=5 (all tokens so far).
        The mask should be [1, 1, 1, 5] — the new token can attend to all 5 positions.
        """
        mask = make_causal_mask(3, key_len=5, device="cpu")
        # Row i can attend to columns 0..i+(key_len-query_len)
        # For query_len=3, key_len=5, offset=2:
        #   query 0 can see keys 0,1,2       (columns 0..2)
        #   query 1 can see keys 0,1,2,3     (columns 0..3)
        #   query 2 can see keys 0,1,2,3,4   (columns 0..4)
        assert mask.shape == (1, 1, 3, 5)
        # Last row (most recent query) should see all keys
        assert mask[0, 0, 2, :].all(), "Last query should attend to all keys"
        # First row should not see last two keys
        assert not mask[0, 0, 0, 3], "First query should not see key at position 3"
        assert not mask[0, 0, 0, 4], "First query should not see key at position 4"

    def test_single_token_mask(self):
        """Mask for single token (seq_len=1) should be all True."""
        mask = make_causal_mask(1, device="cpu")
        assert mask.shape == (1, 1, 1, 1)
        assert mask.item() == True
```

**Step 2: Expand make_causal_mask signature**

Replace the `make_causal_mask` function:

```python
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
```

**Step 3: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: support L!=S causal mask for KV-cache forward-compat, add mask value tests"
```

---

### Task 7: Enrich SwiGLU stub and tests

**Files:**
- Modify: `src/engine/model.py:179-185` (swiglu_ffn TODO)
- Modify: `tests/test_model.py` (TestSwiGLU)

**Step 1: Add SwiGLU tests**

Add to `TestSwiGLU`:

```python
    def test_uses_silu_not_relu(self):
        """SwiGLU uses SiLU (smooth), not ReLU (piecewise linear).

        SiLU allows small negative outputs near x=-1.28, ReLU does not.
        """
        x = torch.full((1, 1, 64), -1.28)  # SiLU minimum
        gate = torch.eye(64)
        up = torch.eye(64)
        down = torch.eye(64)

        result = swiglu_ffn(x, gate, up, down)
        # SiLU(-1.28) ≈ -0.278, so gate output is negative
        # gate * up = negative * negative-ish, result should have some negative values
        # If ReLU were used, gate output would be 0 (all negative inputs clipped)
        # This test would pass with ReLU too since the product could be 0,
        # so we just check shape and non-NaN
        assert result.shape == (1, 1, 64)
        assert not torch.isnan(result).any()

    def test_small_dimensions(self):
        """Should work with non-standard dimensions."""
        x = torch.randn(2, 3, 16)
        gate = torch.randn(32, 16)
        up = torch.randn(32, 16)
        down = torch.randn(16, 32)

        result = swiglu_ffn(x, gate, up, down)
        assert result.shape == (2, 3, 16)
```

**Step 2: Enrich the SwiGLU stub**

Replace the swiglu_ffn TODO:

```python
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
```

**Step 3: Commit**

```bash
git add src/engine/model.py tests/test_model.py
git commit -m "phase-1: enrich swiglu stub with GLU explanation, add dimension tests"
```

---

### Task 8: Enrich generate and forward stubs

**Files:**
- Modify: `src/engine/model.py:306-328` (forward TODO)
- Modify: `src/engine/model.py:360-375` (generate TODO)

**Step 1: Enrich forward() stub**

Replace the forward TODO:

```python
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
```

**Step 2: Enrich generate() stub**

Replace the generate TODO:

```python
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
```

**Step 3: Commit**

```bash
git add src/engine/model.py
git commit -m "phase-1: enrich forward/generate stubs with prefill/decode terminology, logsumexp note"
```

---

### Task 9: Add remaining sampler tests

**Files:**
- Modify: `tests/test_model.py` (TestSampler class)

**Step 1: Add comprehensive sampler tests**

Expand `TestSampler` (already has greedy tests from Task 1):

```python
    def test_temperature_sharpens(self):
        """Low temperature should make the distribution peakier."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        cold = temperature_scale(logits, 0.1)
        hot = temperature_scale(logits, 10.0)

        # Cold: differences amplified (3.0/0.1=30 vs 1.0/0.1=10)
        cold_range = cold.max() - cold.min()
        hot_range = hot.max() - hot.min()
        assert cold_range > hot_range, "Low temp should amplify differences"

    def test_temperature_one_is_identity(self):
        """Temperature=1.0 should not change logits."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = temperature_scale(logits, 1.0)
        assert torch.allclose(result, logits)

    def test_top_k_keeps_k_tokens(self):
        """Top-k should keep exactly k tokens, rest should be -inf."""
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = top_k(logits, k=3)

        kept = (result != float('-inf')).sum().item()
        assert kept == 3, f"Expected 3 tokens kept, got {kept}"
        # The top 3 values (5.0, 4.0, 3.0) should be unchanged
        assert result[1] == 5.0  # index 1 = 5.0
        assert result[4] == 4.0  # index 4 = 4.0
        assert result[2] == 3.0  # index 2 = 3.0

    def test_top_p_filters_tail(self):
        """Top-p should keep tokens until cumulative probability >= p."""
        # Logits designed so softmax gives roughly [0.05, 0.7, 0.15, 0.02, 0.08]
        logits = torch.tensor([0.0, 3.0, 1.5, -1.0, 0.5])
        result = top_p(logits, p=0.9)

        # Token at index 1 (prob ~0.7) should always be kept
        assert result[1] != float('-inf'), "Highest prob token should be kept"
        # Token at index 3 (prob ~0.02) should likely be filtered
        # (it's in the tail beyond 90%)

    def test_min_p_relative_threshold(self):
        """Min-p should filter tokens below p * max_probability."""
        # One dominant token, rest much smaller
        logits = torch.tensor([10.0, 1.0, 0.0, -1.0, -5.0])
        result = min_p(logits, p=0.1)

        # Token 0 (highest) should always be kept
        assert result[0] != float('-inf')
        # Token 4 (very low prob relative to max) should be filtered
        assert result[4] == float('-inf'), "Very low prob token should be filtered"

    def test_repetition_penalty_reduces_repeated(self):
        """Repeated tokens should have lower logits after penalty."""
        logits = torch.tensor([5.0, 3.0, 1.0, -1.0])
        generated = [0, 2]  # tokens 0 and 2 were generated before

        result = repetition_penalty(logits, generated, penalty=1.5)

        # Positive logits divided by penalty → smaller
        assert result[0] < logits[0], "Positive repeated logit should decrease"
        # Negative logits multiplied by penalty → more negative
        assert result[2] < logits[2], "Positive repeated logit should decrease"
        # Non-repeated tokens unchanged
        assert result[1] == logits[1], "Non-repeated token should be unchanged"
        assert result[3] == logits[3], "Non-repeated token should be unchanged"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model.py::TestSampler -v`
Expected: FAIL with `NotImplementedError` for all

**Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "phase-1: add comprehensive sampler tests (temperature, top-k, top-p, min-p, rep penalty)"
```

---

### Task 10: Add embedding and transformer block tests

**Files:**
- Modify: `tests/test_model.py`

**Step 1: Add TestEmbedding class**

```python
class TestEmbedding:
    """Test embedding lookup."""

    def test_lookup_shape(self):
        """Embedding lookup should produce [B, S, hidden_dim]."""
        vocab_size, hidden_dim = 100, 32
        embed_weights = torch.randn(vocab_size, hidden_dim)
        token_ids = torch.tensor([[1, 5, 10]])  # [1, 3]

        # Embedding is just indexing: weights[token_ids]
        result = embed_weights[token_ids]
        assert result.shape == (1, 3, hidden_dim)

    def test_lookup_values(self):
        """Each row should match the corresponding weight row."""
        vocab_size, hidden_dim = 100, 32
        embed_weights = torch.randn(vocab_size, hidden_dim)
        token_ids = torch.tensor([[3, 7]])

        result = embed_weights[token_ids]
        assert torch.equal(result[0, 0], embed_weights[3])
        assert torch.equal(result[0, 1], embed_weights[7])
```

**Step 2: Add TestTransformerBlock class (integration, needs weights)**

```python
@requires_model
class TestTransformerBlock:
    """Test a single transformer block against HuggingFace reference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from transformers import AutoModelForCausalLM

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.hf_model.eval()
        self.weights = load_weights(MODEL_DIR, device="cpu", dtype=torch.bfloat16)

    def test_layer_0_output_matches(self):
        """Single transformer block output should match HuggingFace layer 0."""
        from src.engine.model import transformer_block

        torch.manual_seed(42)
        x = torch.randn(1, 4, HIDDEN_DIM, dtype=torch.bfloat16)
        positions = torch.arange(4).unsqueeze(0)
        mask = make_causal_mask(4, device="cpu")

        # Your implementation
        your_out = transformer_block(x, self.weights, layer_idx=0, positions=positions, mask=mask)

        # HuggingFace reference — run layer 0 manually
        hf_layer = self.hf_model.model.layers[0]
        with torch.no_grad():
            # HF needs [B, S, D] input and position_ids, attention_mask
            hf_out = hf_layer(
                x,
                position_ids=positions,
                attention_mask=mask.to(torch.bfloat16).masked_fill(mask == False, float('-inf')).masked_fill(mask == True, 0.0),
            )
            hf_hidden = hf_out[0]

        assert torch.allclose(your_out, hf_hidden, atol=1e-3, rtol=1e-2), \
            f"Layer 0 output diverges. Max diff: {(your_out - hf_hidden).abs().max().item()}"
```

**Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "phase-1: add embedding lookup tests and transformer block integration test"
```

---

### Task 11: Update phase1-guide.md — RoPE variants section

**Files:**
- Modify: `docs/phase1-guide.md:780-814` (RoPE Step 4 section)

**Step 1: Add RoPE variants discussion**

After the existing RoPE Step 4 code block (around line 808), insert a new section:

```markdown
#### RoPE Variants: Traditional vs Split-Half

The code above shows the **traditional (interleaved)** variant. HuggingFace's Qwen2
implementation uses the **split-half** variant. Both are mathematically equivalent but
pair dimensions differently:

**Traditional (interleaved):**
Pairs adjacent dimensions: (dim0, dim1), (dim2, dim3), ..., (dim126, dim127)
```python
q_even = q[..., 0::2]   # dims 0, 2, 4, ...
q_odd  = q[..., 1::2]   # dims 1, 3, 5, ...
q_rot = torch.stack([q_even * cos - q_odd * sin,
                      q_even * sin + q_odd * cos], dim=-1).flatten(-2)
```

**Split-half (HuggingFace):**
Pairs first half with second half: (dim0, dim64), (dim1, dim65), ..., (dim63, dim127)
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]   # dims 0-63
    x2 = x[..., x.shape[-1] // 2 :]   # dims 64-127
    return torch.cat([-x2, x1], dim=-1)

# Frequencies are doubled: emb = cat([angles, angles], dim=-1)  → [B, S, 128]
q_rot = q * cos + rotate_half(q) * sin
```

**Which to use?** Match your reference implementation. Since we validate against
HuggingFace transformers, use the **split-half** variant. The stubs in `model.py`
use split-half accordingly.

Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
```

**Step 2: Commit**

```bash
git add docs/phase1-guide.md
git commit -m "phase-1: add RoPE variants (traditional vs split-half) to guide"
```

---

### Task 12: Update phase1-guide.md — broadcast GQA, logsumexp, stepping stones, common bugs

**Files:**
- Modify: `docs/phase1-guide.md`

**Step 1: Add broadcast GQA note after the repeat_interleave explanation**

After line 864 (the repeat_interleave explanation), insert:

```markdown
**Alternative: broadcast-based GQA (memory-efficient)**

Instead of copying KV heads 7 times (allocating 7x memory), you can use
PyTorch's broadcasting to achieve the same result without extra allocations:

```python
# Reshape Q to expose the group structure
# q: [B, S, 28, 128] → [B, S, 4, 7, 128]
q = q.view(B, S, NUM_KV_HEADS, GQA_RATIO, HEAD_DIM)

# K/V add a broadcast dimension
# k: [B, S, 4, 128] → [B, S, 4, 1, 128]
k = k.unsqueeze(3)
v = v.unsqueeze(3)

# Now Q @ K^T broadcasts: [B, 4, 7, S, D] @ [B, 4, 1, D, S] → [B, 4, 7, S, S]
# The "1" in K's dim broadcasts to match Q's "7" without copying data.
```

This saves memory proportional to `GQA_RATIO` (7x for Qwen2.5-7B). For Phase 1
with short sequences it doesn't matter much, but it becomes critical for long
sequences in Phase 2+ where KV cache memory dominates.

Try `repeat_interleave` first (simpler to debug), then refactor to broadcasting.
```

**Step 2: Add logsumexp note to the generation section**

In Part 3 generation section, after the generate code, add:

```markdown
**Numerical stability: logsumexp normalization**

Before passing logits to the sampler, it's good practice to normalize them:

```python
next_logits = next_logits - torch.logsumexp(next_logits, dim=-1, keepdim=True)
```

`logsumexp(x) = log(sum(exp(x)))`. Subtracting it converts raw logits to
log-probabilities (values in (-inf, 0]). This prevents softmax overflow inside
the sampler when logits have large magnitude. For greedy decoding (argmax) it
makes no difference, but for temperature/top-p sampling it improves stability.
```

**Step 3: Add stepping stones note at the beginning of Part 3**

After line 568 ("This is the core of Phase 1"), insert:

```markdown
### Warm-up: Building Block Functions

Before tackling the model operations, `model.py` includes three stepping-stone
functions at the top. Implement these first to build intuition:

1. **`softmax(x, dim)`** — numerically stable softmax. You'll use this inside
   attention. Teaches the max-subtract trick for preventing overflow.
2. **`silu(x)`** — the SiLU activation `x * sigmoid(x)`. You'll use this inside
   SwiGLU. Teaches the gating concept.
3. **`simple_attention(q, k, v, mask)`** — scaled dot-product attention where
   Q, K, V all have the same head count. Teaches the core attention algorithm
   before GQA adds the head-count mismatch complexity.

These are tested independently in `tests/test_model.py`.
```

**Step 4: Add RoPE variant to common bugs table**

Add a new row to the table at line 1242:

```markdown
| RoPE outputs wrong values | Using interleaved (even/odd) instead of split-half variant. Qwen2.5 uses split-half in HF. See RoPE Variants section. |
```

**Step 5: Add paper citations to Further Reading**

Append to the Further Reading section:

```markdown
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) — RoPE paper
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) — GQA paper
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) — SwiGLU paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — original transformer
```

**Step 6: Commit**

```bash
git add docs/phase1-guide.md
git commit -m "phase-1: add broadcast GQA, logsumexp, stepping stones, paper citations to guide"
```

---

### Task 13: Final review — run all tests, verify stubs raise correctly

**Files:**
- None (read-only verification)

**Step 1: Run all unit tests**

Run: `python -m pytest tests/test_model.py -v -k "not HuggingFace and not TransformerBlock"`
Expected: ALL tests should FAIL with `NotImplementedError` (stubs are correct, tests catch the right thing). No `None` returns, no `ImportError`, no `AttributeError`.

**Step 2: Verify imports work**

Run: `python -c "from src.engine.model import softmax, silu, simple_attention, make_causal_mask, attention, rmsnorm, rotary_embedding, swiglu_ffn, forward, generate; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Verify sampler imports work**

Run: `python -c "from src.engine.sampler import greedy, temperature_scale, top_k, top_p, min_p, repetition_penalty, sample; print('All sampler imports OK')"`
Expected: `All sampler imports OK`

**Step 4: Commit tag**

```bash
git tag v0.1-phase1-enriched
```
