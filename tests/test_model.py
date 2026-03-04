"""Test forward pass correctness against HuggingFace transformers.

Run: pytest tests/test_model.py -v

These tests load both your implementation and HuggingFace's, feed the same
input, and compare outputs. This is the ground truth for Phase 1.

Requirements:
  - Qwen2.5-7B-Instruct weights downloaded locally
  - HuggingFace transformers + accelerate installed
  - Enough VRAM for both models (run sequentially, not simultaneously)

Set MODEL_DIR env var to your local weights path:
  export MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
"""

import os

import pytest
import torch

# Will be importable once you implement the modules
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
from src.engine.tokenizer import QwenTokenizer
from src.engine.sampler import (
    greedy,
    temperature_scale,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    sample,
)

MODEL_DIR = os.environ.get("MODEL_DIR", "")

# Tolerances
FP16_ATOL = 1e-4
FP16_RTOL = 1e-3


def requires_model(func):
    """Skip test if model weights aren't available."""
    return pytest.mark.skipif(
        not MODEL_DIR or not os.path.isdir(MODEL_DIR),
        reason="MODEL_DIR not set or invalid — download Qwen2.5-7B-Instruct first",
    )(func)


# ─── Unit tests (no model weights needed) ───────────────────────────────────

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


class TestRMSNorm:
    """Test RMSNorm against a known reference."""

    def test_basic(self):
        """RMSNorm with ones weight should just normalize."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, HIDDEN_DIM)
        weight = torch.ones(HIDDEN_DIM)

        result = rmsnorm(x, weight, RMSNORM_EPS)

        # RMS of output should be approximately 1.0
        rms = torch.sqrt(torch.mean(result ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), \
            f"RMS should be ~1.0 after normalization, got {rms}"

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(2, 8, HIDDEN_DIM)
        weight = torch.randn(HIDDEN_DIM)
        result = rmsnorm(x, weight, RMSNORM_EPS)
        assert result.shape == x.shape

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


class TestAttention:
    """Test GQA attention."""

    def test_output_shape(self):
        """Attention output should have Q head count, not KV head count."""
        B, S = 1, 4
        q = torch.randn(B, S, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)

        result = attention(q, k, v)

        assert result.shape == (B, S, NUM_Q_HEADS, HEAD_DIM)

    def test_causal_mask(self):
        """First token's attention output should not depend on future tokens."""
        B, S = 1, 4
        q = torch.randn(B, S, NUM_Q_HEADS, HEAD_DIM)
        k = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)
        v = torch.randn(B, S, NUM_KV_HEADS, HEAD_DIM)

        from src.engine.model import make_causal_mask
        mask = make_causal_mask(S, device="cpu")

        result_masked = attention(q, k, v, mask)

        # Change future tokens in k/v — first position output should be unchanged
        k2 = k.clone()
        v2 = v.clone()
        k2[:, 1:, :, :] = torch.randn_like(k2[:, 1:, :, :])
        v2[:, 1:, :, :] = torch.randn_like(v2[:, 1:, :, :])

        result_modified = attention(q, k2, v2, mask)

        assert torch.allclose(result_masked[:, 0], result_modified[:, 0], atol=1e-5), \
            "First token should not be affected by future tokens when mask is applied"


class TestSwiGLU:
    """Test SwiGLU FFN."""

    def test_output_shape(self):
        """FFN should preserve hidden_dim."""
        x = torch.randn(1, 4, HIDDEN_DIM)
        gate = torch.randn(18944, HIDDEN_DIM)
        up = torch.randn(18944, HIDDEN_DIM)
        down = torch.randn(HIDDEN_DIM, 18944)

        result = swiglu_ffn(x, gate, up, down)

        assert result.shape == x.shape


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


# ─── Integration tests (need model weights) ─────────────────────────────────

@requires_model
class TestAgainstHuggingFace:
    """Compare your implementation against HuggingFace transformers output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load HuggingFace model for reference."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # keep on CPU to save VRAM for your implementation
        )
        self.hf_model.eval()

        self.weights = load_weights(MODEL_DIR, device="cpu", dtype=torch.bfloat16)
        self.tokenizer = QwenTokenizer(MODEL_DIR)

    def test_tokenizer_matches(self):
        """Tokenizer should produce same token IDs as HuggingFace."""
        text = "Hello, how are you?"
        your_ids = self.tokenizer.encode(text)
        hf_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        assert your_ids == hf_ids, f"Tokenizer mismatch:\n  yours: {your_ids}\n  HF:    {hf_ids}"

    def test_forward_logits_match(self):
        """Forward pass logits should match HuggingFace within tolerance."""
        text = "The capital of France is"
        hf_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor([hf_ids])

        # HuggingFace reference
        with torch.no_grad():
            hf_out = self.hf_model(input_tensor)
            hf_logits = hf_out.logits  # [1, seq_len, vocab]

        # Your implementation
        your_logits = forward(input_tensor, self.weights)

        assert torch.allclose(your_logits, hf_logits, atol=FP16_ATOL, rtol=FP16_RTOL), \
            f"Logits diverge! Max diff: {(your_logits - hf_logits).abs().max().item()}"

    def test_greedy_generation_matches(self):
        """Greedy decoding should produce identical tokens to HuggingFace."""
        prompt = "The meaning of life is"
        hf_ids = self.hf_tokenizer.encode(prompt, add_special_tokens=False)

        # HuggingFace greedy
        input_tensor = torch.tensor([hf_ids])
        with torch.no_grad():
            hf_out = self.hf_model.generate(
                input_tensor,
                max_new_tokens=20,
                do_sample=False,
            )
        hf_tokens = hf_out[0].tolist()

        # Your greedy
        from src.engine.model import generate
        your_tokens = generate(hf_ids, self.weights, max_new_tokens=20, device="cpu")

        assert your_tokens == hf_tokens, \
            f"Generation mismatch:\n  yours: {your_tokens}\n  HF:    {hf_tokens}"
