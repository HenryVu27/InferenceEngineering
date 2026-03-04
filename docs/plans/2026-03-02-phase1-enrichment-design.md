# Phase 1 Enrichment Design

**Date**: 2026-03-02
**Approach**: Layered enrichment (in-place, no file splits)
**Inspiration**: tiny-llm reference solutions and book chapters

## Motivation

Cross-referencing the Phase 1 stubs against skyzh's tiny-llm project revealed bugs, test gaps,
and pedagogical opportunities. This design enriches Phase 1 without changing its structure.

## Bug Fixes

1. **sampler.py greedy()** — bare `return` on line 23 before `raise NotImplementedError` causes
   the function to return `None` instead of raising. Remove it.
2. **model.py RoPE variant** — stubs describe interleaved (even/odd) RoPE, but Qwen2.5 uses
   split-half in HuggingFace. Fix the TODO to describe the correct variant and document both.

## New Stepping-Stone Functions (model.py)

Three warm-up functions placed before their advanced counterparts:

1. **`softmax(x, dim)`** — manual implementation with max-subtract for numerical stability.
   Teaches the foundation before it's used inside attention.
2. **`silu(x)`** — manual `x * sigmoid(x)` instead of `torch.nn.functional.silu`.
   Teaches the activation function before SwiGLU uses it.
3. **`simple_attention(q, k, v, mask)`** — scaled dot-product attention where Q/K/V have
   identical head counts. Stepping stone before GQA, which adds head-count mismatch.

## Enriched Stub TODOs

| Function | Enrichment |
|---|---|
| `rmsnorm()` | Add float32 upcasting note — compute in float32, cast back to input dtype |
| `rotary_embedding()` | Document traditional (interleaved) vs non-traditional (split-half), note Qwen2.5 uses split-half |
| `attention()` | Add broadcast GQA alternative to repeat_interleave, note memory trade-off |
| `make_causal_mask()` | Support L != S (query_len != key_len) for KV cache forward-compat |
| `generate()` | Add prefill/decode terminology, logsumexp normalization note |
| `forward()` | Note causal mask only needed when seq_len > 1 |

## Test Expansion (~15 new test cases)

### Building block tests (no weights needed)

- `TestSoftmax` — vs `torch.softmax`, numerical stability with large values
- `TestSiLU` — vs `torch.nn.functional.silu`
- `TestCausalMask` — verify values (not just shape), square and rectangular (L != S)
- `TestEmbedding` — lookup shape and values from small random weight matrix

### Enriched existing tests

- `TestRMSNorm` — add bfloat16 large-magnitude test ([-1000, 1000]) to catch missing float32 upcast
- `TestRoPE` — parametrize traditional vs split-half, add position-offset test
- `TestAttention` — add simple_attention test, multi-precision (float32 + bfloat16), varied batch dims
- `TestSwiGLU` — reference comparison test, multiple dimension configs

### New integration tests

- `TestTransformerBlock` — single block vs HuggingFace reference (requires weights)
- `TestSampler` — deterministic tests per function: greedy, temperature, top_k, top_p, min_p, repetition_penalty

## Guide Updates (phase1-guide.md)

Targeted patches:

- **RoPE section**: traditional vs non-traditional discussion, side-by-side code, flag the gotcha
- **Attention section**: broadcast GQA explanation after repeat_interleave approach
- **Generation section**: prefill/decode terminology, logsumexp normalization
- **New "Stepping Stones" subsection**: note that softmax/silu/simple_attention are warm-ups
- **Paper citations**: RoPE, GQA, SwiGLU, GLU Variants papers where relevant
- **Common bugs table**: add RoPE variant mismatch entry

## What Does NOT Change

- File structure (no new files, no splits)
- Pure-function style, weight-dict approach
- Stubs remain stubs (enriched TODOs, not implementations)
- run.py, tokenizer.py untouched
