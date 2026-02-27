# Inference Engineering

Build an LLM inference engine from scratch. Learning project — every phase builds on the last.

**Hardware**: RTX 5080 (16GB GDDR7, 960 GB/s, Blackwell, native FP4/FP8)
**Model**: Qwen2.5-7B-Instruct (7.61B params, 28 layers, GQA 28Q/4KV, head dim 128)
**Roadmap**: `docs/roadmap.md` — detailed phase-by-phase guide with research and references
**Concept deep-dives**: `docs/concepts/*.md`

## Project Structure

```
src/engine/             # Core inference engine
  model.py              # Phase 1: Forward pass
  tokenizer.py          # Phase 1: Tokenizer
  sampler.py            # Phase 1: Sampling
  kv_cache.py           # Phase 2: KV cache
  memory.py             # Phase 2: Memory manager
  kernels/              # Phase 3: Triton/CUDA kernels
  quantization/         # Phase 4: INT8, FP8, FP4
  serving/              # Phase 5: HTTP + batching
  speculative.py        # Phase 6: Speculative decoding
  structured.py         # Phase 7: Constrained decoding
  attention/            # Phase 8: FlashAttention, PagedAttention
tests/                  # Correctness tests (vs HuggingFace reference)
benchmarks/             # Performance benchmarks
notebooks/              # Exploration and visualization
docs/                   # Roadmap, concepts, benchmark results
```

## Progress

- [ ] Phase 1 — Naked Forward Pass
- [ ] Phase 2 — KV Cache + Memory Management
- [ ] Phase 3 — Custom GPU Kernels (Triton)
- [ ] Phase 4 — Quantization (INT8, FP8, FP4)
- [ ] Phase 5 — Serving: Continuous Batching + HTTP
- [ ] Phase 6 — Speculative Decoding
- [ ] Phase 7 — Structured Output / Constrained Decoding
- [ ] Phase 8 — Advanced Attention Mechanisms
- [ ] Phase 9 — Profiling + Benchmarking (ongoing)
- [ ] Phase 10 — Advanced Topics (stretch)

## Coding Conventions

### Python
- Python 3.11+ with type hints
- `torch` for tensor ops — no `torch.nn.Module` for core engine
- `ruff` format (line length 100), `pytest` for tests
- Docstrings with tensor shapes: `# [batch, seq_len, hidden_dim]`

### Kernels (Triton)
- Each file: naive → optimized version(s) + PyTorch reference + benchmark + test
- Name convention: `kernel_name_v1`, `kernel_name_v2`
- Document block sizes, num_warps, and rationale
- `@triton.autotune` for autotuning

### Benchmarking
- Every optimization: before/after measurement
- `torch.cuda.Event` for GPU timing (not wall clock)
- Median of 100 runs with warmup
- Include GPU model in results, save as JSON in `benchmarks/results/`

### Testing
- Correctness vs HuggingFace `transformers` reference
- `torch.allclose(output, ref, atol=1e-4, rtol=1e-3)` for FP16
- Wider tolerance for quantized (measure perplexity, not exact match)

### Git
- One commit per logical change
- Messages: `phase-N: description`
- Tags: `v0.1-phase1`, `v0.2-phase2`, etc.
- Never commit model weights

## Notes for Claude

1. **Validate against reference** — test against HuggingFace transformers output
2. **Benchmark everything** — never claim "should be faster" without measuring
3. **Explain the why** — this is a learning project
4. **Show tensor shapes** — annotate dimensions in comments
5. **Progressive complexity** — simplest correct version first, then optimize
6. **Use the RTX 5080** — leverage Blackwell FP4/FP8/Tensor Cores
7. **No magic** — avoid abstractions that hide operations
8. **Real numbers** — concrete benchmarks, memory calculations, roofline analysis
