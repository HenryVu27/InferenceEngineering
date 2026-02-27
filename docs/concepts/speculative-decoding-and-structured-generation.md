# Speculative Decoding & Structured/Constrained Generation: Research Summary

> Research compiled February 2026. Covers developments from 2022-2026.

---

## Part 1: Speculative Decoding

### 1.1 Original Speculative Decoding (Leviathan et al., 2022)

**Papers:**
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" ([arXiv:2302.01318](https://arxiv.org/pdf/2302.01318))

**Core Mechanism:**

Speculative decoding exploits a fundamental asymmetry: verifying K tokens in parallel (a single forward pass of the target model on K tokens) is nearly as fast as generating 1 token, because the decode phase is memory-bandwidth-bound, not compute-bound. The algorithm:

1. **Draft phase**: A small, fast draft model `q(x)` autoregressively generates K candidate tokens: `x_1, x_2, ..., x_K`.
2. **Verify phase**: The large target model `p(x)` processes all K tokens in a single forward pass, producing probabilities `p(x_i | x_{0:n}, x_{1:i-1})` for each position.
3. **Accept/Reject**: For each drafted token `x_t`, in order:
   - Compute acceptance probability: **`a_t = min(1, p(x_t) / q(x_t))`**
   - Draw `r ~ Uniform(0, 1)`
   - If `r < a_t`: **accept** the token and proceed to `x_{t+1}`
   - If `r >= a_t`: **reject** the token, discard all subsequent tokens, and **resample** from the adjusted residual distribution

**Residual Distribution on Rejection:**

When a token is rejected, the replacement token is sampled from:

```
p'(x) = normalize(max(0, p(x) - q(x)))
```

This is the key to losslessness. The combined probability of accepting a draft token plus resampling on rejection exactly recovers the target distribution:

```
P(output = x) = q(x) * min(1, p(x)/q(x))  +  [rejection prob] * p'(x)  =  p(x)
```

**Expected Accepted Tokens:**

If we define the token-level acceptance rate `alpha` as:

```
alpha = E[min(1, p(x)/q(x))]  =  1 - (1/2) * sum_x |p(x) - q(x)|
```

(which equals `1 - TV(p, q)` where TV is total variation distance)

Then the expected number of tokens generated per speculation round with draft length K follows a truncated geometric distribution. In the idealized case (K -> infinity):

```
E[accepted tokens] = 1 / (1 - alpha)
```

For finite K, it is:

```
E[accepted tokens] = (1 - alpha^(K+1)) / (1 - alpha)
```

The "+1" accounts for the bonus token sampled on rejection or at the end.

**Performance Characteristics:**
- Speedup: 2-3x typical at batch size 1, highly dependent on draft-target alignment
- Overhead: one extra forward pass of the draft model per speculation round
- No quality degradation: mathematically guaranteed to produce the exact target distribution
- Works best on "predictable" text (code, structured data, formulaic language)
- Works worst on creative/diverse text where draft and target distributions diverge

**Implementation Complexity for From-Scratch Engine:** **Medium**
- Need two models loaded simultaneously (memory overhead)
- Need the rejection sampling loop with correct residual distribution
- Need to handle KV cache management for both models
- Need to handle partial acceptance (discard suffix, keep accepted prefix)
- Core algorithm is ~200 lines of Python; the complexity is in efficient KV cache management

---

### 1.2 Medusa: Multiple Decoding Heads

**Paper:** Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774))
**Code:** [github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)

**Core Mechanism:**

Instead of using a separate draft model, Medusa adds K extra "heads" (simple MLP layers) on top of the target model's final hidden states. Each head `h_k` predicts the token at position `t+k`:

```
head_1: predicts token at t+1 (same as original LM head)
head_2: predicts token at t+2
head_3: predicts token at t+3
...
head_K: predicts token at t+K
```

**Tree-Structured Verification:**

Each head produces a top-s_k set of candidate tokens. The candidates across heads form a Cartesian product, creating a tree of candidate continuations. For example, with 3 heads each producing top-2:

```
Position t+1:  [A, B]
Position t+2:  [C, D]  (for each t+1 candidate)
Position t+3:  [E, F]  (for each t+1, t+2 combination)
```

This creates a tree with up to `2 * 2 * 2 = 8` candidate paths. A specially constructed **tree attention mask** restricts each token to attend only to its ancestors in the tree, allowing all candidates to be verified in a single forward pass. Positional indices are set according to tree depth (not sequence position in the flattened batch).

**Two Variants:**
- **Medusa-1**: Heads are trained; uses standard greedy/sampling for acceptance. No distribution guarantees.
- **Medusa-2**: Adds a "typical acceptance" scheme inspired by speculative decoding's rejection sampling to preserve the target distribution, with a tree-based extension.

**Training:**
- Only the Medusa heads are trained; the base model is frozen
- Training data: ~1M tokens from ShareGPT or similar
- Training time: hours on a single GPU (parameter-efficient)
- Recommended: 5 heads maximum

**Performance:**
- Medusa-1: 2.2x speedup (no quality loss for greedy, slight quality shift for sampling)
- Medusa-2: 2.3-3.6x speedup (preserves target distribution)
- Memory overhead: minimal (each head is a small MLP, ~0.1-0.5% of model parameters)
- Per-token overhead: the tree attention is more expensive than a single decode step, but amortized over multiple accepted tokens

**Implementation Complexity for From-Scratch Engine:** **Medium-High**
- Need to add and train MLP heads (requires training infrastructure)
- Need tree attention mask construction and management
- Need efficient tree verification in a single forward pass
- Tree structure optimization (which candidates to include) is non-trivial
- Tighter coupling with the model architecture than standard speculative decoding

---

### 1.3 EAGLE / EAGLE-2 / EAGLE-3: Feature-Level Autoregressive Drafting

**Papers:**
- EAGLE-1: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077), ICML 2024)
- EAGLE-2: Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858), EMNLP 2024)
- EAGLE-3: Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840), NeurIPS 2025)
- **Code:** [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

**Core Mechanism (EAGLE-1):**

EAGLE's key insight: predicting the next **feature vector** (the hidden state from the second-to-last layer) is much easier than predicting the next **token**, because features are continuous and smoother than the discrete, high-entropy token space.

Architecture:
1. During target model inference, capture the feature vector `f_t` from the second-to-last layer
2. A lightweight **auto-regression head** (single transformer layer + embedding) takes `(f_t, token_t)` as input and predicts `f_{t+1}`
3. The predicted feature `f_{t+1}` is passed through the **frozen** original LM head (classification layer) to get the draft token
4. This process repeats autoregressively within the draft model to generate multiple draft tokens
5. Draft tokens form a tree (multiple candidates per position), verified in one target model forward pass

**EAGLE-2 Improvements:**
- **Dynamic draft trees**: Instead of a fixed tree structure, EAGLE-2 uses the draft model's confidence scores to approximate token-level acceptance rates
- Dynamically adjusts the tree width and depth based on context: wider trees when uncertain, deeper trees when confident
- 20-40% faster than EAGLE-1

**EAGLE-3 Improvements (2025):**
- **Removes feature prediction constraint**: Instead of predicting features, directly predicts tokens
- **Multi-layer feature fusion**: Uses features from low, mid, and high layers instead of just the second-to-last layer
- **Training-time test**: A technique that simulates the inference-time feature availability during training
- **Scaling law**: Discovered that increasing training data leads to proportional speedup improvements (not observed in EAGLE-1/2)
- 1.4x faster than EAGLE-2 at batch size 1; 1.38x throughput improvement at batch size 64

**Performance:**
- EAGLE-1: 2.7-3.5x speedup (certified by third-party evaluation as fastest method in 2024)
- EAGLE-2: 3.05-4.26x speedup
- EAGLE-3: 3.0-6.5x speedup (depending on model size and batch)
- Training: ~1-2 days on a single GPU for the auto-regression head
- Memory overhead: auto-regression head is ~0.5-2% of target model parameters

**Implementation Complexity for From-Scratch Engine:** **High**
- Need to train the auto-regression head (requires training loop, data pipeline)
- Need to extract intermediate features from the target model during inference
- Need dynamic tree construction and verification (EAGLE-2/3)
- Need tree attention with dynamic topology
- Most complex of the speculative methods, but highest speedup

---

### 1.4 Self-Speculative Decoding: LayerSkip and SWIFT

#### LayerSkip (Meta, 2024)

**Paper:** Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding" ([arXiv:2404.16710](https://arxiv.org/abs/2404.16710), ACL 2024)
**Code:** [github.com/facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

**Core Mechanism:**

Use the same model for both drafting and verification, eliminating the need for a separate draft model:

1. **Draft phase**: Run only the first N layers (e.g., first 8 of 32) with the shared LM head to generate K draft tokens quickly
2. **Verify phase**: Run the remaining layers (layers 9-32) on the draft tokens to verify and correct

The key advantage is **shared compute**: the draft phase's KV cache for layers 1-N is reused during verification, so verification only needs to compute layers N+1 to L.

**Training Requirements:**
LayerSkip requires a special training recipe:
- Progressive layer dropout: low dropout for early layers, high dropout for later layers
- Early exit loss: all layers share the same exit point (LM head)
- This trains the early layers to produce useful predictions even without later layers

**Performance:**
- 1.6-2.0x speedup on code generation, summarization
- No additional memory for a draft model (single model)
- Sensitive to the choice of exit layer and draft length (task-dependent)
- Integrated into Hugging Face transformers (Nov 2024) and PyTorch torchtune (Dec 2024)

#### SWIFT (2024, ICLR 2025)

**Paper:** "SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration" ([arXiv:2410.06916](https://arxiv.org/abs/2410.06916))
**Code:** [github.com/hemingkx/SWIFT](https://github.com/hemingkx/SWIFT)

**Core Mechanism:**

Unlike LayerSkip, SWIFT requires **no special training**. It is plug-and-play:

1. **Optimization phase**: Before each decoding step, SWIFT identifies which intermediate layers can be skipped with minimal quality loss, using random search with interval Bayesian optimization
2. **Acceleration phase**: Skips the identified layers during drafting, uses the full model for verification
3. The skipped layer set is **adaptive**: it changes based on the input data stream

**Performance:**
- 1.3-1.6x speedup
- No training required (works with any off-the-shelf model)
- Lower speedup than LayerSkip but zero training cost

**Implementation Complexity for From-Scratch Engine:** **Low-Medium**
- LayerSkip: Low implementation complexity but requires training modifications
- SWIFT: Low implementation complexity, no training needed. The layer-skipping draft and full-model verification reuse existing infrastructure. Main challenge is the adaptive layer selection optimization.
- Both benefit from the fact that only one model is loaded

---

### 1.5 Staged/Cascade Speculative Decoding

**Papers:**
- "Cascade Speculative Drafting for Even Faster LLM Inference" (NeurIPS 2024)
- Google Research, "Speculative Cascades: A Hybrid Approach for Smarter, Faster LLM Inference" ([research.google/blog](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/))

**Core Mechanism:**

Instead of a single draft model, use a **cascade of progressively larger models**:

```
Model 0 (tiny, e.g., n-gram / lookup)
  -> drafts tokens for Model 1 (small, e.g., 0.5B)
    -> drafts tokens for Model 2 (medium, e.g., 1.5B)
      -> drafts tokens for Model 3 (target, e.g., 7B)
```

**Two types of cascading (CS Drafting):**

1. **Vertical Cascade**: Eliminates autoregressive generation from neural models entirely at the lowest level. Uses n-gram models or retrieval to generate initial candidates.
2. **Horizontal Cascade**: Optimizes time allocation across drafting stages. Earlier stages generate more candidates (wider), later stages filter them.

**Speculative Cascades (Google):**
Combines speculative decoding with quality-based routing. If a smaller model is "confident enough," its output is used directly without consulting the larger model, saving compute for easy tokens and only escalating to larger models for difficult tokens.

**Performance:**
- Up to 81% speedup over standard speculative decoding
- Best suited for serving scenarios where you already have multiple model sizes deployed

**Implementation Complexity for From-Scratch Engine:** **High**
- Need to load and manage 3+ models simultaneously
- Need sophisticated scheduling to pipeline the cascade stages
- Memory requirements are significant (multiple models in VRAM)
- More relevant for production serving than a learning project
- For a 16GB GPU, likely impractical to fit more than 2 models

---

### 1.6 Lookahead Decoding (Jacobi Iteration)

**Paper:** Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" ([arXiv:2402.02057](https://arxiv.org/abs/2402.02057), ICML 2024)
**Code:** [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding)

**Core Mechanism:**

Lookahead decoding reframes autoregressive generation as solving a system of nonlinear equations via **Jacobi iteration**:

Standard autoregressive decoding is a sequential fixed-point iteration:
```
x_1 = f(prompt)
x_2 = f(prompt, x_1)
x_3 = f(prompt, x_1, x_2)
...
```

Jacobi iteration solves this in parallel by initializing all positions with guesses and iterating:
```
Iteration 0: [guess_1, guess_2, guess_3, ..., guess_K]
Iteration 1: [f(prompt), f(prompt, guess_1), f(prompt, guess_1, guess_2), ...]
Iteration 2: [converged_1, f(prompt, converged_1), f(prompt, converged_1, x_2'), ...]
...until convergence
```

**The Problem with Pure Jacobi:** In practice, most positions do not converge quickly, so raw Jacobi decoding provides little speedup.

**Lookahead's Solution:** Two parallel branches:
1. **Lookahead branch**: Maintains a sliding 2D window of Jacobi iterations, collecting n-grams that appear in the iteration trajectories
2. **Verification branch**: Selects promising n-gram candidates from the collected set and verifies them (like speculative decoding)

The key insight: Jacobi iterations produce useful n-gram candidates as a byproduct, even when individual positions don't converge.

**Performance:**
- 1.5-2.3x speedup on a single GPU
- **No auxiliary model required** (uses only the target model)
- **No training required**
- Trades extra FLOPs (per step) for fewer total steps
- More parallelizable than standard speculative decoding

**Implementation Complexity for From-Scratch Engine:** **Medium-High**
- Need to implement Jacobi iteration with the target model
- Need n-gram collection and caching from iteration trajectories
- Need efficient verification of candidate n-grams
- Need careful KV cache management for the parallel positions
- Conceptually elegant but the 2D window management adds complexity

---

### 1.7 Best Draft Models for Qwen2.5-7B-Instruct

Based on available benchmarks and community results:

| Draft Model | Parameters | Speedup (approx.) | Notes |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 0.5B | 2.0-2.5x | Best bang for buck. Same tokenizer, same architecture family. Fits easily alongside 7B in 16GB. |
| **Qwen2.5-1.5B-Instruct** | 1.5B | 1.5-1.8x | Better acceptance rate but slower drafting. Net speedup is often lower than 0.5B. |
| **Qwen2.5-0.5B** (base) | 0.5B | ~2.0x | Slightly worse match for instruct-tuned target, but still effective. |
| **EAGLE head** (trained) | ~50M | 3.0-4.5x | Best speedup but requires training the auto-regression head specifically for Qwen2.5-7B. |

**Key considerations for your RTX 5080 (16GB):**
- Qwen2.5-7B at FP16 = ~14GB. No room for a draft model at FP16.
- Qwen2.5-7B at INT8 = ~7GB. Qwen2.5-0.5B at FP16 = ~1GB. **This fits** (8GB total, leaving 8GB for KV cache and activations).
- Qwen2.5-7B at FP8 = ~7GB. Qwen2.5-0.5B at FP16 = ~1GB. **This fits** and leverages your Blackwell FP8 tensor cores.
- For EAGLE: the auto-regression head is tiny (~50-100M parameters), so memory is not a concern.

**Recommendation for your project:**
1. Start with **Qwen2.5-0.5B-Instruct** as the draft model (simplest, good speedup, same tokenizer)
2. Then implement **EAGLE** with a trained head for maximum speedup
3. Try **SWIFT self-speculative** as a zero-cost baseline (no extra model needed)

---

## Part 2: Structured / Constrained Generation

### 2.1 XGrammar: The State of the Art

**Paper:** Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" ([arXiv:2411.15100](https://arxiv.org/abs/2411.15100), MLSys 2025)
**Code:** [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
**Blog:** [blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)

**Core Mechanism:**

XGrammar models grammars (JSON schemas, regex, EBNF) as **pushdown automata (PDA)**, which are more expressive than finite state machines (can handle nested structures like JSON). The key innovation is the **context-independent/context-dependent token split**:

**Token Categorization:**

For a PDA with states (position + stack), determine for each vocabulary token whether its validity depends on the stack contents:

1. **Context-Independent Tokens (~99% of vocabulary):** A token's validity depends only on the current PDA position (grammar rule), NOT the stack. Example: in JSON, the token `"name"` is always valid when the grammar expects a string literal, regardless of nesting depth.

2. **Context-Dependent Tokens (~1% of vocabulary):** Validity depends on the full stack. Example: `}` is valid only if there is a matching `{` on the stack.

**Precomputation (Grammar Compilation):**

During a one-time compilation step:
- For each PDA state (ignoring stack), precompute a **bitmask** over the entire vocabulary indicating which context-independent tokens are valid
- Store these bitmasks in an **adaptive token mask cache**
- This is the expensive step (seconds to minutes depending on grammar complexity), but it is done once per grammar

**Runtime (Per-Token Mask Generation):**

At each generation step:
1. Look up the precomputed bitmask for the current grammar position -> handles ~99% of tokens in O(1)
2. For the remaining ~1% context-dependent tokens, execute the PDA to check validity -> O(|context-dependent tokens|)
3. Combine into the final mask and apply to logits before softmax

**Additional Optimizations:**
- **Compiler optimizations**: Inlining grammar rules and merging equivalent PDA states reduces the automaton size
- **Tree-based stack management**: Efficiently manages multiple possible stacks (when grammar is ambiguous) using a persistent tree data structure with O(1) amortized fork/merge
- **Efficient rollback**: The tree structure allows rewinding to a previous state cheaply

**Performance:**
- Up to **100x speedup** over previous solutions (Outlines, llama.cpp grammar)
- Per-token mask generation: **< 40 microseconds** for JSON Schema and CFG tasks
- **Near-zero overhead** in end-to-end LLM serving (mask generation overlaps with GPU LLM execution)
- Grammar compilation: milliseconds to seconds for typical JSON schemas
- As of 2025: default structured generation backend in both **vLLM** and **SGLang**

**Implementation Complexity for From-Scratch Engine:** **High**
- Need to implement a PDA (pushdown automaton) from a grammar specification
- Need the context-independent/context-dependent analysis over the full vocabulary (vocabulary can be 32K-128K tokens, each of which can span multiple characters)
- Need the bitmask precomputation and caching infrastructure
- Need efficient multi-byte token handling (a single token like `": [` may span multiple grammar transitions)
- Recommend: use XGrammar as a library rather than reimplementing from scratch, then study its internals
- For learning: implement a simplified version for JSON-only (without full CFG support)

---

### 2.2 Outlines: FSM-Based Constrained Generation

**Paper:** Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023)
**Code:** [github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)

**Core Mechanism:**

Outlines compiles constraints (regex patterns, JSON schemas) into **finite state machines (FSMs)**, then uses the FSM to mask logits at each generation step:

1. **Schema -> Regex**: Convert JSON schema into an equivalent regular expression
2. **Regex -> FSM**: Convert the regex into a deterministic finite automaton (DFA)
3. **FSM State -> Valid Token Index**: For every FSM state, precompute which vocabulary tokens are valid transitions. This creates an **index** mapping `state -> set of valid token IDs`.
4. **At inference**: Given the current FSM state, look up valid tokens in O(1), mask all other logits to `-inf`, sample from the remaining tokens, advance the FSM state.

**Index Precomputation Details:**

For each FSM state `s` and each vocabulary token `t`:
- Simulate consuming `t` character-by-character from state `s`
- If the FSM can consume all characters of `t` and reach a valid state `s'`, then `t` is valid at state `s`
- This is the expensive step: `O(|states| * |vocabulary| * avg_token_length)`

**Limitations:**
- **FSMs cannot handle nested structures**: JSON with arbitrary nesting depth is not a regular language. Outlines handles this by generating regexes that support a fixed maximum nesting depth.
- **Compilation can be slow**: Complex schemas with `minItems`, `maxItems`, or large enums can cause regex explosion, leading to compilation times of 40 seconds to 10+ minutes.
- **No CFG support**: Limited to regular languages (regex/FSM). Cannot express things like balanced parentheses or recursive structures natively.

**Performance:**
- Per-token overhead: O(1) after precomputation (simple index lookup)
- Recent optimization: 98% reduction in runtime overhead by storing FSM-token-masks as GPU tensors
- Compilation: fast for simple schemas, can be very slow for complex ones
- Speedup: constrained decoding can actually be faster than unconstrained decoding when many tokens are deterministic (the FSM skips them)

**Implementation Complexity for From-Scratch Engine:** **Medium**
- JSON Schema -> Regex conversion requires handling all JSON schema features
- Regex -> DFA is well-understood (use a library like `interegular` or implement Thompson's construction + subset construction)
- Index precomputation is straightforward but can be slow for large vocabularies
- The FSM step logic is simple: ~50 lines of code
- **Good starting point** for learning constrained generation before tackling XGrammar

---

### 2.3 SGLang's Structured Generation

**Paper:** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" ([arXiv:2312.07104](https://arxiv.org/abs/2312.07104), NeurIPS 2024)
**Code:** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
**Blog:** [lmsys.org/blog/2024-01-17-sglang/](https://lmsys.org/blog/2024-01-17-sglang/)

**Core Mechanism:**

SGLang combines two innovations:

**1. RadixAttention (KV Cache Reuse):**

SGLang maintains a **radix tree** of all KV cache blocks across all active and recent requests. When a new request arrives:
- Walk the radix tree to find the longest matching prefix with existing cache entries
- Reuse the matching KV cache blocks, only computing the new suffix
- LRU eviction for cache management

This is particularly powerful for structured generation because:
- Multiple requests with the same system prompt share KV cache
- Branching patterns (e.g., generating multiple JSON fields) can share common prefixes
- Reduces both memory and computation

**2. Compressed FSM for Structured Decoding:**

SGLang introduced a **compressed finite state machine** approach:
- Analyze the FSM to find "singular transition paths" — sequences of states where each state has exactly one valid next state
- Compress these paths: instead of generating one token at a time, jump directly to the end of the deterministic path
- This is the precursor to what XGrammar calls "jump-forward decoding"

**Performance:**
- 3x faster JSON decoding compared to uncompressed FSM approaches
- With RadixAttention: throughput improvements of 2-5x on structured workloads with shared prefixes
- SGLang now integrates XGrammar as its default constrained decoding backend (as of 2025)

**Implementation Complexity for From-Scratch Engine:** **High**
- RadixAttention requires a radix tree for KV cache management (Phase 2 material)
- Compressed FSM is medium complexity
- The combination is a production-grade serving optimization
- Relevant for Phase 5 (serving) and Phase 7 (structured output) together

---

### 2.4 JSON Schema Enforcement: Practical Implementation Approaches

There are three main approaches, in increasing order of expressiveness:

**Approach 1: Regex/FSM (Outlines-style)**

```
JSON Schema -> Regular Expression -> DFA -> Token Mask Index
```

- Pros: Simple, fast per-token, well-understood
- Cons: Cannot handle recursive/nested schemas beyond a fixed depth; regex compilation can explode
- Best for: Simple, flat JSON schemas without deep nesting

**Approach 2: CFG/PDA (XGrammar-style)**

```
JSON Schema -> EBNF Grammar -> Pushdown Automaton -> Token Mask (with CI/CD split)
```

- Pros: Handles arbitrary nesting, recursive structures, full JSON spec
- Cons: More complex implementation, PDA state management
- Best for: Production use, complex schemas, nested structures

**Approach 3: Character-level Checking (lm-format-enforcer-style)**

```
JSON Schema -> Dynamic On-the-Fly Character Checking -> Token Mask
```

- Pros: Most flexible, handles arbitrary formats, no precomputation
- Cons: Slower per token (must check every vocabulary token at every step)
- Best for: Rapid prototyping, unusual formats, when compilation overhead is unacceptable

**Practical Implementation for a From-Scratch Engine (Recommended Path):**

1. **Start simple**: Implement a JSON tokenizer that tracks parser state (are we in a string? in an object? expecting a key?). For each state, compute valid next characters. Map characters to valid tokens.

2. **Add FSM approach**: Convert simple JSON schemas to regex, build a DFA, precompute token masks. This gets you 80% of the way there with moderate effort.

3. **Graduate to PDA**: For full JSON schema support (nested objects, arrays, recursive schemas), implement a pushdown automaton. Use XGrammar's context-independent/context-dependent split for performance.

---

### 2.5 CFG-Based Generation: Pushdown Automata During Decoding

**Core Technical Mechanism:**

A context-free grammar (CFG) is more expressive than regular expressions. It can describe:
- Balanced brackets/braces (JSON, code)
- Recursive structures (nested objects)
- Matching delimiters

A **pushdown automaton (PDA)** is the execution model for CFGs. It extends an FSM with a **stack**:

```
State = (current_grammar_position, stack_contents)
```

At each generation step:
1. Determine the current PDA state (position in grammar + stack)
2. For each vocabulary token, simulate consuming it character-by-character through the PDA
3. If the PDA reaches a valid state after consuming the token, the token is valid
4. Mask invalid tokens, sample from valid ones
5. Update the PDA state

**The Multi-Byte Token Challenge:**

LLM vocabularies contain multi-character tokens (e.g., `"hello"`, `": {"`, `],\n  "`). A single token may:
- Cross multiple grammar rule boundaries
- Push and pop the stack multiple times
- Be partially valid (first 3 chars valid, 4th char invalid)

Handling this correctly requires simulating the PDA for each character within a token, which is the main source of complexity.

**Performance:**
- Naive: O(|vocabulary| * avg_token_length) per generation step — too slow
- XGrammar optimization: O(|context-dependent tokens| * avg_token_length) per step — fast
- Typical: < 100 microseconds per step with XGrammar's approach

**Implementation Complexity for From-Scratch Engine:** **High**
- PDA implementation is well-understood but multi-byte token handling is tricky
- Need to handle ambiguous grammars (multiple possible PDA paths)
- Need efficient stack management (tree-based for multiple paths)
- XGrammar's CI/CD split is the key optimization; without it, performance is poor

---

### 2.6 Jump-Forward Decoding

**Origin:** SGLang's compressed FSM ([LMSYS blog, 2024-02-05](https://lmsys.org/blog/2024-02-05-compressed-fsm/))

**Core Mechanism:**

When the grammar dictates that only one valid token (or one valid character sequence) is possible at the current position, **skip the LLM entirely** and emit the deterministic token directly:

```python
# Standard decoding (one step per token):
Step 1: LLM generates '{' (only valid token) -> waste of a forward pass
Step 2: LLM generates '"' (only valid token) -> waste of a forward pass
Step 3: LLM generates 'name' (one of many valid strings) -> actual generation needed

# Jump-forward decoding:
Step 1: Grammar determines '{"' is deterministic -> emit directly, no LLM call
Step 2: LLM generates 'name' -> actual generation
```

In JSON generation, a significant fraction of tokens are structural boilerplate: `{`, `}`, `[`, `]`, `:`, `,`, `"`, whitespace. Jump-forward decoding emits all of these without invoking the model.

**Implementation:**

1. At each state, check if only one valid token exists (or only one valid character sequence leading to a single token)
2. If so, emit the token directly, advance the grammar state, repeat
3. Only invoke the LLM when multiple valid tokens are possible

**Performance:**
- Can reduce the number of LLM forward passes by 30-60% for JSON generation
- Constrained decoding with jump-forward can be **faster than unconstrained decoding** (fewer forward passes)
- Compressed FSM approach: up to 2x latency reduction and 2.5x throughput improvement

**Implementation Complexity for From-Scratch Engine:** **Low** (given you already have grammar-guided decoding)
- It is a simple optimization on top of any grammar-based system
- Check if the current state has a single valid continuation; if yes, skip the LLM
- ~20 lines of additional code on top of existing constrained decoding

---

### 2.7 lm-format-enforcer and Other Tools

#### lm-format-enforcer

**Code:** [github.com/noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)

**Approach:**
- Dynamic, on-the-fly constraint checking (no precomputation)
- Character-level validation: for each vocabulary token, check if appending it to the current output would violate the schema
- Works by maintaining a "character-level enforcer" that tracks the parser state

**Pros:**
- Most flexible: handles arbitrary JSON schemas, regex, custom formats
- No compilation step (instant startup)
- Low false positive rate in RAG tasks (0.49-0.7% hallucination rate)

**Cons:**
- Slower per-token than precomputed approaches (must check every token at every step)
- No jump-forward optimization
- O(|vocabulary|) per step

**Implementation Complexity:** **Low**
- Conceptually the simplest approach
- Good for prototyping, then optimize with precomputation later

#### llguidance (Guidance AI)

**Code:** [github.com/guidance-ai/llguidance](https://github.com/guidance-ai/llguidance)

- Written in Rust for performance
- Supports CFGs, regex, JSON schema
- Zero compilation timeouts on benchmarks (more robust than XGrammar on complex schemas)
- Used as a backend option in vLLM

#### guidance

**Code:** [github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)

- Python library for constrained generation
- Defines constraints as Python code interleaved with generation calls
- Supports select (choice), regex, JSON schema, CFG
- Good developer experience but tightly coupled to specific model integrations

---

## Comparison Summary

### Speculative Decoding Methods

| Method | Speedup | Training Needed | Extra Memory | Draft Quality | Implementation Complexity |
|---|---|---|---|---|---|
| Standard (Leviathan) | 2-3x | No | Full draft model | Depends on draft | Medium |
| Medusa | 2.2-3.6x | Yes (heads only) | Minimal (~0.1%) | Good | Medium-High |
| EAGLE-1 | 2.7-3.5x | Yes (auto-reg head) | Minimal (~0.5%) | Very good | High |
| EAGLE-2 | 3.0-4.3x | Yes | Minimal | Very good | High |
| EAGLE-3 | 3.0-6.5x | Yes (more data) | Minimal | Excellent | High |
| LayerSkip | 1.6-2.0x | Yes (special recipe) | None | Moderate | Low-Medium |
| SWIFT | 1.3-1.6x | No | None | Moderate | Low-Medium |
| Lookahead | 1.5-2.3x | No | None | N/A | Medium-High |
| Cascade | up to 81% over SD | No | Multiple models | High | High |

### Structured Generation Methods

| Method | Approach | Per-Token Overhead | Compilation Time | Expressiveness | Implementation Complexity |
|---|---|---|---|---|---|
| XGrammar | PDA + CI/CD split | < 40 µs | ms - seconds | Full CFG | High |
| Outlines | Regex -> FSM | O(1) lookup | ms - 10+ min* | Regular (no nesting) | Medium |
| SGLang (compressed FSM) | FSM + jump-forward | O(1) + skips | seconds | Regular + jump | Medium-High |
| lm-format-enforcer | Dynamic checking | O(\|vocab\|) | None | Arbitrary | Low |
| llguidance | Rust CFG engine | Very fast | Fast | Full CFG | N/A (use as library) |

*Outlines compilation time depends heavily on schema complexity.

---

## Recommended Implementation Order for Your Project

### Phase 6 (Speculative Decoding):
1. **Standard speculative decoding** with Qwen2.5-0.5B as draft (start here — the math is foundational)
2. **SWIFT self-speculative** (no training, quick to implement, good baseline comparison)
3. **EAGLE-1** with trained auto-regression head (for maximum speedup)
4. **Lookahead decoding** (no extra model, conceptually interesting)

### Phase 7 (Structured Generation):
1. **Simple JSON parser state tracker** with logit masking (understand the core concept)
2. **Regex -> FSM approach** (Outlines-style, for flat schemas)
3. **Jump-forward optimization** (easy win on top of FSM)
4. **PDA approach** with context-independent/context-dependent split (XGrammar-style, for full JSON support)
5. Integrate XGrammar as a library for production comparison

---

## Key References

### Speculative Decoding
- [Google Research: Looking Back at Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
- [BentoML: Speculative Decoding Handbook](https://bentoml.com/llm/inference-optimization/speculative-decoding)
- [NVIDIA: Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [Jay Mody: Speculative Sampling](https://jaykmody.com/blog/speculative-sampling/)
- [ki-seki: How is the Speculative Decoding Algorithm Constructed?](https://ki-seki.github.io/posts/251226-spec-decoding/)
- [vLLM Blog: How Speculative Decoding Boosts Performance by 2.8x](https://blog.vllm.ai/2024/10/17/spec-decode.html)
- [NVIDIA: Optimizing Qwen2.5-Coder with TensorRT-LLM Lookahead Decoding](https://developer.nvidia.com/blog/optimizing-qwen2-5-coder-throughput-with-nvidia-tensorrt-llm-lookahead-decoding/)

### Structured Generation
- [XGrammar Blog](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
- [LMSYS: Fast JSON Decoding with Compressed FSM](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
- [vLLM Blog: Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [Let's Data Science: How Structured Outputs and Constrained Decoding Work](https://www.letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json)
- [Aidan Cooper: Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/)
- [JSONSchemaBench: Evaluating Constrained Decoding](https://arxiv.org/html/2501.10868v1)
- [SqueezeBits: Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
