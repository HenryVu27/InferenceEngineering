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
    swiglu_ffn,
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


class TestRoPE:
    """Test Rotary Position Embedding."""

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
