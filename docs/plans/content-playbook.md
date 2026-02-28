# Content Playbook: Building an Inference Engine in Public

## Positioning

**One-liner**: Building an LLM inference engine from scratch on consumer hardware (RTX 5080).

**Why this works**:
- Almost nobody documents this publicly — most content is "how to use vLLM" or API wrappers
- RTX 5080 is new — people are hungry for real FP4/FP8 benchmarks on Blackwell
- From-scratch builds attract engineers, hiring managers, and GPU vendors alike
- Each phase produces concrete artifacts (benchmarks, kernels, flame graphs) that are inherently shareable

**Audience**: ML engineers, CUDA/GPU enthusiasts, inference infra people, hiring managers at AI companies. Secondary: students and self-learners following along.

**Voice**: Honest, technical, learning-oriented. Not "I'm an expert teaching you" — more "I'm figuring this out and here's what I found." Show the failures and debugging, not just polished results.

---

## Platform Strategy

| Platform | Purpose | Cadence | Content Type |
|----------|---------|---------|--------------|
| **X/Twitter** | Discovery + community | Multiple per phase | Short updates, screenshots, TILs, benchmark charts, threads |
| **Blog** | Depth + SEO + portfolio | 1 post per phase | Long-form deep dives, the canonical reference |
| **LinkedIn** | Career visibility | 1 post per phase | Condensed version of blog post, career-framed |
| **Reddit** | Distribution | Per blog post | r/LocalLLaMA, r/MachineLearning, r/CUDA |
| **Hacker News** | Distribution | Best posts only | Submit the posts with strongest hook |
| **GitHub** | Credibility | Always current | Clean repo, good README, tagged releases per phase |

### X/Twitter Playbook

This is where your brother's TinyTPU approach applies most directly.

**What to post**:
- "Starting Phase N" announcements with a one-sentence goal
- Screenshots of terminal output (first token generated, benchmark results, errors)
- Before/after benchmark charts (simple bar charts, not overproduced)
- "TIL" moments — the surprising gotchas (Qwen's QKV bias, RoPE frequency edge cases)
- Short code snippets that show something interesting (a clean Triton kernel, a tricky tensor reshape)
- Questions to the community ("anyone know why X happens on Blackwell?")

**What NOT to post**:
- Polished "10 things I learned" threads (feels like content marketing, not building)
- Vague motivational takes ("day 47 of my AI journey")
- Reposting others' content without your own angle

**Engagement tactics**:
- Tag relevant people when you have genuine results (Tri Dao for attention work, Tim Dettmers for quantization, the vLLM team for serving)
- Reply to others' inference/GPU threads with your own data points
- Share concrete numbers — "63 tok/s BF16 → 126 tok/s INT8 on RTX 5080" gets way more engagement than "I optimized my inference engine"

### Blog Playbook

**Title format**: "Building an LLM Inference Engine from Scratch — Part N: [Phase Name]"

**Structure for each post**:
1. What I'm building this phase (1-2 paragraphs)
2. The core concept explained from first principles (the learning part)
3. Implementation walkthrough with code snippets
4. What went wrong and how I debugged it (the build-in-public part)
5. Benchmarks with methodology (the credibility part)
6. What's next (hooks the reader for the series)

**Blog improvements**:
- Add dates to posts — build-in-public needs a visible timeline
- Lead with the inference engine series, not generic AI engineering content
- Add a project landing page: "Follow along as I build an LLM inference engine from scratch"

### LinkedIn Playbook

Reframe blog content for a career audience:
- "I implemented X and learned Y about how LLM inference actually works"
- Focus on the skills demonstrated, not just the technical details
- Keep it shorter than the blog — 3-5 key takeaways with a link to the full post

---

## Phase-by-Phase Content Map

### Phase 1 — Naked Forward Pass

**The hook**: "I generated my first token without using any ML framework's model API"

**Tweet-worthy moments**:
- Loading safetensor weights manually and seeing tensor shapes for the first time
- First successful token — screenshot the terminal output
- Comparison: your output vs HuggingFace output (matching to 1e-4!)
- Diagram: the full forward pass data flow you drew to understand it
- The QKV bias gotcha if you hit it (Qwen-specific, people will relate)

**Blog post angle**: "What actually happens when an LLM generates a token" — walk through every matrix multiply, every normalization, every activation. The post most tutorials skip.

**Visual artifacts**: Forward pass diagram, tensor shape annotations, accuracy comparison table

### Phase 2 — KV Cache + Memory Management

**The hook**: "Why your LLM re-computes the same thing every token (and how to stop it)"

**Tweet-worthy moments**:
- Before/after tok/s with KV cache (this will be dramatic)
- Memory usage visualization — how KV cache grows with sequence length
- The moment you realize why long contexts are expensive (show the math)

**Blog post angle**: The KV cache is the single most impactful optimization and most people don't understand why. Explain it with concrete memory numbers for Qwen on your specific GPU.

**Visual artifacts**: Memory growth chart, tok/s comparison, KV cache layout diagram

### Phase 3 — Custom GPU Kernels (Triton)

**The hook**: "I wrote my first GPU kernel and it was 10x slower than PyTorch. Here's how I fixed it."

**Tweet-worthy moments**:
- First Triton kernel working (even if slow)
- Naive vs optimized kernel benchmark comparison
- Roofline analysis chart showing where your kernels land
- "My fused RMSNorm+Linear kernel" — the first real speedup

**Blog post angle**: GPU kernel development for someone who's never written one. Start with "why write kernels at all", show the roofline model, walk through naive → optimized.

**Visual artifacts**: Roofline chart, kernel benchmark table, occupancy analysis

**High engagement potential**: Triton content is underserved. Most Triton tutorials are trivial vector-add examples. Real inference kernels are rare.

### Phase 4 — Quantization (INT8, FP8, FP4)

**The hook**: "FP4 on RTX 5080: 4x less memory, and it actually works"

**Tweet-worthy moments**:
- Side-by-side output quality: BF16 vs FP8 vs FP4 (same prompt, compare responses)
- Benchmark chart: tok/s across precisions on RTX 5080
- NVFP4 dual-level scaling explained in one diagram
- Perplexity table showing quality vs compression tradeoff

**Blog post angle**: This is your highest-value content. RTX 5080 FP4 benchmarks barely exist publicly. You'd be one of the first to publish real from-scratch FP4 inference results on Blackwell consumer hardware.

**Visual artifacts**: Precision comparison table, tok/s bar chart, quality vs compression scatter plot

**Distribution note**: This post has the best shot at HN/Reddit traction. Lead with the RTX 5080 angle.

### Phase 5 — Serving: Continuous Batching + HTTP

**The hook**: "I built an inference server that handles multiple users without wasting GPU cycles"

**Tweet-worthy moments**:
- First HTTP request returning a streamed response
- Throughput chart: naive batching vs continuous batching
- "My server handles N concurrent requests on a single RTX 5080"

**Blog post angle**: How inference serving actually works under the hood. Most people use vLLM/TGI as black boxes. Show what continuous batching really does.

**Visual artifacts**: Request timeline diagram, throughput comparison, architecture diagram

### Phase 6 — Speculative Decoding

**The hook**: "2-3x faster inference by guessing tokens (and being right most of the time)"

**Tweet-worthy moments**:
- Acceptance rate charts for different draft models
- Side-by-side: speculative vs standard decoding speed
- "Qwen2.5-0.5B drafting for Qwen2.5-7B — N% acceptance rate"
- The math of why speculative decoding is lossless (people find this surprising)

**Blog post angle**: Speculative decoding is one of those ideas that sounds like it shouldn't work. Great for a "here's the theory, here's my implementation, here's the proof it's lossless" structure.

**Visual artifacts**: Acceptance rate by position chart, speedup comparison, draft/verify timeline

### Phase 7 — Structured Output / Constrained Decoding

**The hook**: "Forcing an LLM to output valid JSON without hoping really hard"

**Tweet-worthy moments**:
- Demo: same prompt, unconstrained vs JSON-constrained output
- Per-token overhead of grammar enforcement (< 40 microseconds with XGrammar)
- "My engine now guarantees valid JSON/XML/SQL output"

**Blog post angle**: How constrained decoding works (FSMs, PDAs, token masking). Practical and increasingly relevant as structured output becomes standard.

**Visual artifacts**: FSM state diagram, overhead benchmark, before/after output comparison

### Phase 8 — Advanced Attention

**The hook**: "I implemented FlashAttention from scratch and finally understand why it's fast"

**Tweet-worthy moments**:
- Memory usage: standard attention vs your implementation
- Speed comparison across sequence lengths
- "The online softmax trick that makes it all work" — one-diagram explanation
- PagedAttention memory waste comparison

**Blog post angle**: FlashAttention is famously hard to understand from the paper alone. A from-scratch implementation walkthrough with diagrams would be extremely valuable.

**Visual artifacts**: Memory comparison chart, tiling diagram, sequence length scaling plot

### Phase 9 — Profiling + Benchmarking

**The hook**: "The full picture: where every microsecond goes in LLM inference"

**Tweet-worthy moments**:
- Full profiling flame graph
- Breakdown pie chart: what percentage of time is attention vs FFN vs sampling
- End-to-end benchmark: Phase 1 performance vs Phase 9 performance (the full journey)

**Blog post angle**: Capstone post. Show the complete optimization journey with numbers at every phase. This is your portfolio piece.

**Visual artifacts**: Before/after comparison across all phases, flame graph, component breakdown

---

## Engagement Flywheel

```
Build a phase
    → Tweet progress (discovery)
        → Write blog post (depth)
            → Share on Reddit/HN/LinkedIn (distribution)
                → People follow for next phase (retention)
                    → Build next phase (repeat)
```

The key insight from your brother's TinyTPU experience: the build itself IS the content. You don't need to create content separately from the engineering work. Every debugging session, every benchmark, every "aha" moment is a post.

---

## Quick Wins to Start Now

1. **Update your blog** — add a project page for the inference engine series, add dates to posts
2. **First tweet** — post about the project with your hardware and goals: "Building an LLM inference engine from scratch on RTX 5080. Starting with a naked forward pass for Qwen2.5-7B. Following along as I go from zero to serving."
3. **Pin the announcement** — make it easy for new followers to find the thread
4. **Start Phase 1** — the content comes from the building, not the other way around

---

## What Good Looks Like (Examples to Study)

- **Andrej Karpathy** — llm.c / minbpe: from-scratch builds with great explanations
- **George Hotz** — tinygrad: build-in-public with strong opinions and real benchmarks
- **Your brother's TinyTPU** — niche + building + engaging the community directly
- **Simon Willison** — prolific blog + twitter, documents everything he builds

Common pattern: they all build real things and show their work. The content is a byproduct of the building, not the other way around.
