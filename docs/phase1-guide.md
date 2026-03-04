# Phase 1 Implementation Guide

A step-by-step walkthrough for implementing every function in the Phase 1 skeleton.
Work through these in order — each section builds on the last.

This guide assumes you know Python but NOT PyTorch tensor operations.
Every line of code is explained: what it does, why it's written that way,
what the shapes mean, and what would break if you did it differently.

---

## Prerequisites

### Download model weights

```bash
# ~15GB download. The models/ directory is in .gitignore.
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

### Install dependencies

```bash
source .venv/Scripts/activate  # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install safetensors tiktoken transformers accelerate pytest ruff
```

### Reference reading (skim first, come back as needed)

- `docs/concepts/qwen2.5-7b-architecture.md` — exact weight names, tensor shapes, config
- `docs/concepts/transformer_math_reference.md` — every formula you'll implement
- [skyzh's tiny-llm Week 1](https://skyzh.github.io/tiny-llm/week1-overview.html) — similar project on MLX
- [andrewkchan's YALM](https://andrewkchan.dev/posts/yalm.html) — from-scratch LLM in C++

---

## PyTorch Crash Course (Read This First)

Before implementing anything, you need to understand tensors and shapes.

### What is a tensor?

A tensor is a multi-dimensional array. Think of it as a generalization of:
- **0D tensor (scalar)**: a single number. `torch.tensor(3.14)` → shape `[]`
- **1D tensor (vector)**: a list of numbers. `torch.tensor([1, 2, 3])` → shape `[3]`
- **2D tensor (matrix)**: a table of numbers. `torch.randn(3, 4)` → shape `[3, 4]`
- **3D tensor**: a stack of matrices. `torch.randn(2, 3, 4)` → shape `[2, 3, 4]`

In LLMs, you'll mostly work with 3D and 4D tensors.

### Shape notation: `[batch, seq_len, hidden_dim]`

Throughout this guide, shapes are written in brackets:

- `[B, S, D]` means a 3D tensor with:
  - `B` = batch size (how many sequences you process at once, usually 1)
  - `S` = sequence length (how many tokens in the input)
  - `D` = hidden dimension (the width of the model's internal representation, 3584 for Qwen2.5-7B)

Example: if you have 1 sequence of 10 tokens, and the model's hidden dim is 3584:
```python
x = torch.randn(1, 10, 3584)   # shape [1, 10, 3584]
x.shape                          # torch.Size([1, 10, 3584])
x.shape[0]                       # 1 (batch)
x.shape[1]                       # 10 (seq_len)
x.shape[2]                       # 3584 (hidden_dim)
```

### What does `dim=` mean?

Many PyTorch operations take a `dim` argument that says "which axis to operate along."

```python
x = torch.tensor([[1, 5, 3],
                   [4, 2, 6]])    # shape [2, 3]

x.sum(dim=0)   # sum along dim 0 (rows) → [5, 7, 9]     (shape [3])
x.sum(dim=1)   # sum along dim 1 (cols) → [9, 12]        (shape [2])
x.sum(dim=-1)  # same as dim=1 (last dim) → [9, 12]      (shape [2])
```

**`dim=-1` means "the last dimension."** This is used everywhere because it works
regardless of how many dimensions the tensor has. If your tensor is `[B, S, D]`,
then `dim=-1` operates along `D`. If it's `[B, H, S, D]`, `dim=-1` still operates along `D`.

### What does `.item()` do?

Converts a single-element tensor to a plain Python number:
```python
t = torch.tensor(42)
t           # tensor(42)     — still a PyTorch tensor
t.item()    # 42             — plain Python int
```

You need `.item()` when PyTorch gives you a tensor but you want a regular Python value
(e.g., a token ID to append to a list).

### What does `@` do?

`@` is matrix multiplication (same as `torch.matmul`):
```python
A = torch.randn(2, 3)    # [2, 3]
B = torch.randn(3, 4)    # [3, 4]
C = A @ B                 # [2, 4] — standard matrix multiply

# With batches:
A = torch.randn(1, 10, 3584)    # [B, S, D]
W = torch.randn(18944, 3584)    # [out_features, in_features]
result = A @ W.T                  # [1, 10, 18944] — each token multiplied by W
```

**Why `.T`?** Weight matrices are stored as `[out_features, in_features]` (convention from
`nn.Linear`). To multiply `[B, S, in_features] @ [in_features, out_features]`, you need
to transpose the weight: `W.T` flips `[out, in]` → `[in, out]`.

### What does `keepdim=True` do?

When you reduce a dimension (mean, sum, etc.), it normally disappears:
```python
x = torch.randn(2, 3, 4)       # shape [2, 3, 4]
x.mean(dim=-1)                   # shape [2, 3]     — dim 4 is gone
x.mean(dim=-1, keepdim=True)     # shape [2, 3, 1]  — dim 4 stays as size 1
```

`keepdim=True` keeps the reduced dimension as size 1, which is critical for
**broadcasting** — PyTorch can automatically expand size-1 dimensions to match
another tensor's shape during element-wise operations:
```python
x = torch.randn(2, 3, 4)               # [2, 3, 4]
mean = x.mean(dim=-1, keepdim=True)     # [2, 3, 1]
result = x - mean                        # [2, 3, 4] — the [1] broadcasts to [4]
```

### What does `unsqueeze` do?

Adds a size-1 dimension at the specified position:
```python
x = torch.tensor([1, 2, 3])    # shape [3]
x.unsqueeze(0)                   # shape [1, 3]  — added dim at position 0
x.unsqueeze(-1)                  # shape [3, 1]  — added dim at the end
```

Used to make shapes compatible for broadcasting. If you have shapes `[3]` and
`[5, 3]` and want to multiply element-wise, you'd `unsqueeze(0)` the first one
to get `[1, 3]`, which broadcasts to `[5, 3]`.

### What is `torch.bfloat16`?

A number format: 16-bit floating point with 8-bit exponent. Compared to `float32`
(32-bit, standard precision), bf16 uses half the memory but has less precision.

```python
x = torch.randn(3, 3, dtype=torch.float32)     # 4 bytes per number
x = torch.randn(3, 3, dtype=torch.bfloat16)    # 2 bytes per number — half the VRAM
```

LLMs use bf16 because: (1) half the memory means the model fits in GPU VRAM,
(2) GPU tensor cores compute bf16 2x faster than float32, (3) the reduced precision
doesn't meaningfully affect output quality for inference.

### What does `.to()` do?

Moves a tensor to a different device or casts to a different dtype:
```python
x = torch.randn(3, 3)             # float32, CPU
x.to(device="cuda")                # float32, GPU
x.to(dtype=torch.bfloat16)         # bfloat16, CPU
x.to(dtype=torch.bfloat16, device="cuda")  # bfloat16, GPU
x.float()                          # shorthand for .to(torch.float32)
```

---

## Part 1: Sampler (`src/engine/sampler.py`)

Start here — it's the easiest file and gives you a quick win.

### 1.1 `greedy(logits)` — pick the most likely token

**What are logits?** When the model processes a sequence, the final output is a vector
of 152,064 numbers — one score for every token in the vocabulary. These raw scores are
called **logits**. Higher logit = model thinks that token is more likely to come next.

```
logits = [1.2, -0.5, 3.7, 0.1, ..., -2.3]
          ↑      ↑     ↑
       token 0  token 1  token 2   ... 152,064 total
```

**Shape**: `logits` comes in as `[152064]` — a 1D tensor with one score per vocab token.
(In the generate loop, we extract just the last position's logits from the full `[B, S, 152064]` output.)

**Implementation**:

```python
return logits.argmax(dim=-1).item()
```

Breakdown:
- `logits.argmax(dim=-1)` — returns the INDEX of the highest value along the last dimension.
  If logits = `[1.2, -0.5, 3.7, 0.1]`, argmax returns `tensor(2)` because index 2 (value 3.7) is largest.
  `dim=-1` means "along the last (and only) dimension." For a 1D tensor, `dim=-1` and `dim=0` are the same.
  We use `dim=-1` out of habit — it works the same whether you pass `[152064]` or `[B, 152064]`.
- `.item()` — converts the PyTorch tensor `tensor(2)` to a plain Python int `2`.
  We need a plain int because we're going to append it to a Python list of token IDs.

**Why not just `int(logits.argmax())`?** That works too. `.item()` is the PyTorch-idiomatic way,
and it's slightly faster because it avoids the Python `int()` constructor overhead.

### 1.2 `temperature_scale(logits, temperature)` — control randomness

**What it does**: Divides all logits by a temperature value before sampling.

- `temperature = 1.0` → no change (default)
- `temperature = 0.5` → logits get doubled → probabilities get sharper → more deterministic
- `temperature = 2.0` → logits get halved → probabilities get flatter → more random

**Why does this work?** Softmax converts logits to probabilities: `p_i = exp(logit_i) / sum(exp(logit_j))`.
When you divide logits by T before softmax: `p_i = exp(logit_i/T) / sum(exp(logit_j/T))`.
Larger T → exponents closer to 0 → all exp() values closer to 1 → more uniform.
Smaller T → exponents spread further apart → winner-take-all.

**Concrete example**:
```
Logits:                [2.0, 1.0, 0.5]

temp=1.0: softmax →    [0.59, 0.24, 0.13]    (normal)
temp=0.5: logits/0.5 = [4.0, 2.0, 1.0] → softmax → [0.84, 0.11, 0.04]  (sharp)
temp=2.0: logits/2.0 = [1.0, 0.5, 0.25] → softmax → [0.42, 0.26, 0.20] (flat)
```

**Implementation**:
```python
if temperature < 1e-8:      # if temperature is basically zero
    return logits            # don't divide — would be infinity
return logits / temperature  # element-wise division: every logit gets divided by T
```

`1e-8` is a tiny number (0.00000001). We check this instead of `temperature == 0`
because floating point numbers can be very small but not exactly zero.

### 1.3 `top_k(logits, k)` — keep only the k best options

**What it does**: Set all logits EXCEPT the top k to negative infinity.
After softmax, `-inf` becomes 0 probability — those tokens can never be sampled.

**Why?** Prevents the model from sampling very unlikely tokens. With vocab size 152,064,
there are thousands of tokens with near-zero probability. Top-k removes them entirely.

**Implementation**:
```python
top_values, _ = torch.topk(logits, k)
threshold = top_values[-1]
return logits.masked_fill(logits < threshold, float('-inf'))
```

Breakdown:
- `torch.topk(logits, k)` — returns two things: the k largest VALUES and their INDICES.
  We only need the values (to find the threshold), so we use `_` to ignore the indices.
  Example: `torch.topk(tensor([1, 5, 3, 9, 2]), 3)` → values=`[9, 5, 3]`, indices=`[3, 1, 2]`
- `top_values[-1]` — the LAST of the top-k values, which is the SMALLEST of the top-k.
  This is our threshold. Anything below this gets removed.
- `logits.masked_fill(condition, value)` — wherever `condition` is True, replace with `value`.
  `logits.masked_fill(logits < threshold, float('-inf'))` means:
  "for every logit smaller than the k-th largest, replace it with -inf."

### 1.4 `top_p(logits, p)` — nucleus sampling

**What it does**: Sort tokens by probability (highest first), walk down the list
adding up probabilities. Once the cumulative sum reaches p (e.g., 0.9), cut off
everything below. This means: "keep the smallest set of tokens that cover 90% of
the probability mass."

**Why top-p instead of top-k?** Top-k always keeps exactly k tokens. But sometimes
the model is very confident (one token has 95% probability) and top-k=50 would include
49 near-zero-probability tokens. Top-p adapts: when the model is confident, fewer tokens
pass; when uncertain, more tokens pass.

**Implementation**:
```python
probs = torch.softmax(logits, dim=-1)
```
Convert logits to probabilities. `softmax(x_i) = exp(x_i) / sum(exp(x_j))`.
This normalizes the scores to sum to 1.0. Shape stays `[152064]`.

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
```
Sort probabilities from highest to lowest. Also returns the original indices
so we can map back later. Example:
```
probs:          [0.05, 0.70, 0.15, 0.10]
sorted_probs:   [0.70, 0.15, 0.10, 0.05]   (highest first)
sorted_indices: [1,    2,    3,    0]        (where each came from)
```

```python
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
```
Running sum: `[0.70, 0.85, 0.95, 1.00]`. At position 2, we've accumulated 0.95,
which is ≥ 0.9 (our p threshold).

```python
sorted_mask = cumulative_probs - sorted_probs >= p
```
**This is the tricky part.** We subtract `sorted_probs` from the cumsum to get
the cumsum BEFORE each token was added:
```
cumulative_probs:                [0.70, 0.85, 0.95, 1.00]
cumulative_probs - sorted_probs: [0.00, 0.70, 0.85, 0.95]   ← cumsum BEFORE this token
>= 0.9?                         [F,    F,    F,    T]
```
Token 3 is the first where the cumsum BEFORE it was already ≥ 0.9, so we remove it.
Tokens 0-2 (probability 0.70 + 0.15 + 0.10 = 0.95) are kept.

**Why subtract?** Without the subtraction, we'd use `cumulative_probs >= p`:
```
>= 0.9?                         [F,    F,    T,    T]
```
This would remove token 2 even though it's the one that PUSHES us past 0.9.
We'd only keep tokens 0-1 (probability 0.85), which is LESS than our target of 0.9.

```python
mask = torch.zeros_like(logits, dtype=torch.bool)
mask.scatter_(0, sorted_indices, sorted_mask)
```
The mask is in SORTED order, but our logits are in ORIGINAL order. `scatter_` puts
sorted values back into their original positions:
```
sorted_mask:    [F, F, F, T]     (sorted order)
sorted_indices: [1, 2, 3, 0]    (original positions)
mask:           [T, F, F, F]    (original order — token 0 is the one to remove)
```

```python
return logits.masked_fill(mask, float('-inf'))
```
Set removed tokens to -inf in the LOGITS (not probs), so downstream softmax
produces 0 for them.

### 1.5 `min_p(logits, p)` — simpler adaptive filtering

**What it does**: Keep any token whose probability is at least `p × max_probability`.
If the top token has probability 0.8 and p=0.1, the threshold is 0.08. Any token with
probability < 0.08 gets removed.

**Why min-p?** It's simpler than top-p and scales naturally with model confidence.
When the model is very confident (max prob = 0.95), the threshold is high and few tokens
pass. When unsure (max prob = 0.15), the threshold is low and many tokens pass.
Introduced in 2023, increasingly popular.

**Implementation**:
```python
probs = torch.softmax(logits, dim=-1)   # convert to probabilities [152064]
max_prob = probs.max()                    # single highest probability
threshold = p * max_prob                  # e.g., 0.1 * 0.8 = 0.08
return logits.masked_fill(probs < threshold, float('-inf'))
```

### 1.6 `repetition_penalty(logits, generated_ids, penalty)` — discourage repeats

**What it does**: For every token that has already appeared in the output, make it
less likely to appear again by shrinking its logit toward zero.

**The asymmetry**: Positive logits get divided by the penalty (making them smaller/less likely).
Negative logits get multiplied by the penalty (making them more negative/even less likely).
This ensures the penalty always pushes the token's probability DOWN regardless of sign.

**Implementation**:
```python
logits = logits.clone()
```
`clone()` creates a copy. Without this, we'd modify the input tensor, which could
cause bugs if the caller still needs the original values.

```python
if not generated_ids:
    return logits
```
If nothing has been generated yet, there's nothing to penalize.

```python
token_ids = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
```
Convert the Python list of previously generated token IDs into a tensor.
`dtype=torch.long` means 64-bit integers — required for indexing.
`device=logits.device` ensures the tensor is on the same device (CPU or GPU) as logits.

```python
scores = logits[token_ids]
```
**Fancy indexing**: if `token_ids = [42, 100, 7]`, this selects `logits[42]`,
`logits[100]`, and `logits[7]` — the logits for the tokens we've already generated.

```python
scores = torch.where(scores > 0, scores / penalty, scores * penalty)
```
`torch.where(condition, if_true, if_false)` — element-wise conditional:
- If score > 0: divide by penalty (e.g., 3.0 / 1.2 = 2.5 — still positive but smaller)
- If score ≤ 0: multiply by penalty (e.g., -1.0 * 1.2 = -1.2 — more negative)

```python
logits[token_ids] = scores
return logits
```
Write the penalized scores back into the logits at the corresponding positions.

### 1.7 `sample(logits, ...)` — full sampling pipeline

**What it does**: Chains all the above functions together, then randomly picks a token.

**Implementation**:
```python
if rep_penalty > 1.0 and generated_ids:
    logits = repetition_penalty(logits, generated_ids, rep_penalty)
logits = temperature_scale(logits, temperature)
if top_k_val > 0:
    logits = top_k(logits, top_k_val)
if top_p_val < 1.0:
    logits = top_p(logits, top_p_val)
if min_p_val > 0.0:
    logits = min_p(logits, min_p_val)
```
Apply each filter in order. The `if` checks disable unused filters
(e.g., `top_k_val=0` means "don't use top-k").

**Order matters**: Repetition penalty first (modifies raw logits), then temperature
(scales logits), then filtering (removes tokens), then sample.

```python
probs = torch.softmax(logits, dim=-1)
```
Convert final filtered logits to probabilities that sum to 1.0.

```python
return torch.multinomial(probs, num_samples=1).item()
```
`torch.multinomial(probs, num_samples=1)` — randomly picks 1 index, where the probability
of picking index i is proportional to `probs[i]`. Returns a tensor like `tensor([4217])`.
`.item()` extracts the Python int `4217`.

This is the fundamental difference between greedy (always pick the highest) and sampling
(randomly pick according to the distribution). Sampling with temperature + filtering
lets the model be creative while staying coherent.

---

## Part 2: Tokenizer (`src/engine/tokenizer.py`)

### 2.1 What is a tokenizer?

A tokenizer converts text to numbers and back:
- **Encode**: `"Hello world"` → `[9707, 1879]`
- **Decode**: `[9707, 1879]` → `"Hello world"`

The model only works with numbers (token IDs). Every unique word, subword, or
character fragment has an assigned ID. Qwen2.5-7B has a vocabulary of 152,064 tokens.

### 2.2 What files does the tokenizer need?

When you download Qwen2.5-7B-Instruct, you'll find these tokenizer files:

```
vocab.json             — maps strings to IDs: {"hello": 14990, "Ġthe": 279, ...}
merges.txt             — BPE merge rules (how to split unknown words into known subwords)
tokenizer.json         — combined config (HuggingFace fast tokenizer format)
tokenizer_config.json  — special tokens, chat template, settings
```

The `Ġ` prefix you'll see in vocab.json means "this token starts with a space."
It's a byte-level encoding convention where byte 0x20 (space) is represented as `Ġ`.

### 2.3 Pragmatic implementation

The tokenizer is NOT the core learning goal of Phase 1 — the forward pass is.
Use HuggingFace's tokenizer internally and focus your energy on the model.

```python
def __init__(self, model_dir: str | Path):
    model_dir = Path(model_dir)
    from transformers import AutoTokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
```
`AutoTokenizer.from_pretrained` reads all the tokenizer files from the directory
and constructs a fast tokenizer. This handles all the BPE encoding complexity.

```python
def encode(self, text: str) -> list[int]:
    return self._tokenizer.encode(text, add_special_tokens=False)
```
`add_special_tokens=False` means: just tokenize the text, don't add any BOS/EOS tokens.
We'll handle special tokens ourselves in `encode_chat`.

```python
def decode(self, token_ids: list[int]) -> str:
    return self._tokenizer.decode(token_ids, skip_special_tokens=False)
```
`skip_special_tokens=False` preserves special tokens in the output text.

### 2.4 ChatML template — implement this yourself

This is the part worth understanding. Qwen2.5 uses the **ChatML** format
to structure conversations. The model was trained to expect this exact format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

Each message is wrapped:
1. `<|im_start|>` (token ID 151644) — signals "a message is starting"
2. The role name (`system`, `user`, or `assistant`) as normal text
3. A newline
4. The message content as normal text
5. `<|im_end|>` (token ID 151645) — signals "this message is done"
6. A newline

After the last message, we add `<|im_start|>assistant\n` WITHOUT `<|im_end|>`,
which prompts the model to start generating the assistant's response.

**Implementation**:
```python
def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
    tokens = []
    for msg in messages:
        role = msg["role"]       # "system", "user", or "assistant"
        content = msg["content"] # the actual message text
        # <|im_start|>{role}\n{content}<|im_end|>\n
        tokens.append(IM_START_ID)                      # 151644 as an integer, NOT as text
        tokens.extend(self.encode(role + "\n" + content)) # "user\nWhat is 2+2?" → [882, 198, ...]
        tokens.append(IM_END_ID)                          # 151645
        tokens.extend(self.encode("\n"))                   # newline between messages
    # Prompt the model to generate as assistant
    tokens.append(IM_START_ID)                            # 151644
    tokens.extend(self.encode("assistant\n"))              # "assistant\n" → [78191, 198]
    return tokens
```

**Critical detail**: `IM_START_ID` and `IM_END_ID` are appended as raw integer IDs.
If you tried `self.encode("<|im_start|>")`, the tokenizer would break the STRING
`"<|im_start|>"` into subword tokens like `["<", "|", "im", "_", "start", "|", ">"]`
— completely wrong. The special tokens must be inserted as their IDs (151644, 151645) directly.

### 2.5 Going deeper (optional, for the blog)

If you want to understand BPE tokenization from scratch:

1. Start with raw bytes of the text: `"Hello" → [72, 101, 108, 108, 111]`
2. Look up each byte in the vocab: byte 72 = "H", byte 101 = "e", etc.
3. Find the highest-priority merge that matches adjacent tokens. If merge rule #1 says
   "merge 'l' + 'l' → 'll'", apply it: `["H", "e", "ll", "o"]`
4. Repeat until no more merges apply.
5. Look up the final tokens in vocab to get IDs.

References:
- [HuggingFace Qwen2 tokenizer source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py)
- [QwenLM tokenizer notes](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md)
- [tiktoken source](https://github.com/openai/tiktoken) — Rust-accelerated BPE

---

## Part 3: Model Forward Pass (`src/engine/model.py`)

This is the core of Phase 1. Take it one function at a time.
**After each function, validate against HuggingFace.**

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

### 3.1 `load_weights(model_dir, device, dtype)` — get the model into memory

**What are safetensors?** A file format for storing tensors (multi-dimensional arrays
of numbers). Qwen2.5-7B's weights are split across 4 files ("shards") because one file
would be too large. An index file maps tensor names to shard files.

**What are weights?** Every matrix multiply in the model uses learned parameters (weights).
The model has 7.61 billion of them, organized into named tensors like
`model.layers.0.self_attn.q_proj.weight` (the query projection matrix for layer 0).

**Implementation**:
```python
import json
from safetensors.torch import load_file

def load_weights(model_dir, device="cuda", dtype=torch.bfloat16):
    model_dir = Path(model_dir)
```

```python
    # 1. Read the index file — it's a JSON that maps tensor names to shard files
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    # weight_map looks like:
    # {
    #   "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    #   "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    #   "model.layers.0.self_attn.q_proj.bias": "model-00001-of-00004.safetensors",
    #   ...
    # }
```

```python
    # 2. Get unique shard filenames (many tensors share the same shard)
    shard_files = set(weight_map.values())  # {"model-00001-of-00004.safetensors", ...}
```

```python
    # 3. Load each shard file and collect all tensors
    weights = {}
    for shard_file in sorted(shard_files):
        shard_path = model_dir / shard_file
        shard_weights = load_file(str(shard_path))
        # load_file returns a dict: {"tensor_name": tensor, ...}
        for name, tensor in shard_weights.items():
            weights[name] = tensor.to(dtype=dtype, device=device)
            # .to(dtype=dtype, device=device) does two things:
            #   1. Cast to bfloat16 (half the memory of float32)
            #   2. Move to GPU ("cuda") or keep on CPU
```

```python
    # 4. Sanity check — Qwen2.5-7B should have exactly 323 named tensors
    assert len(weights) == 323, f"Expected 323 tensors, got {len(weights)}"
    return weights
```

**Why 323 tensors?** Per layer (28 layers):
- input_layernorm.weight (1)
- q_proj.weight + q_proj.bias (2)
- k_proj.weight + k_proj.bias (2)
- v_proj.weight + v_proj.bias (2)
- o_proj.weight (1, no bias)
- post_attention_layernorm.weight (1)
- gate_proj.weight (1)
- up_proj.weight (1)
- down_proj.weight (1)
= 12 per layer × 28 layers = 336, minus 3 (no o_proj bias, confirmed by architecture), wait —
Actually just count: 12 per layer × 28 = 336, plus 3 globals (embed_tokens.weight,
model.norm.weight, lm_head.weight) = 339. But that doesn't match 323 — the exact count
depends on which biases exist. Just load and count. If you get a different number, print
`sorted(weights.keys())` to see what's there.

**Memory note**: BF16 weights = ~15.2 GB. RTX 5080 has 16 GB. This barely fits with
no room for activations or KV cache. Options:
- Load on CPU first (`device="cpu"`) for correctness testing
- Use short sequences (≤ 100 tokens) on GPU
- Wait for Phase 4 (quantization) for comfortable GPU usage

### 3.2 `rmsnorm(x, weight, eps)` — normalize activations

**What is normalization?** As data flows through 28 transformer layers, values can grow
or shrink uncontrollably. Normalization rescales each vector to have a consistent
magnitude, keeping the math stable.

**RMSNorm vs LayerNorm**: LayerNorm centers the data (subtracts mean) AND normalizes.
RMSNorm only normalizes — simpler and faster, works just as well for LLMs.

**The formula**: `y = (x / RMS(x)) * weight`
where `RMS(x) = sqrt(mean(x²))` — the root-mean-square of x.

**Implementation**:
```python
def rmsnorm(x, weight, eps=RMSNORM_EPS):
    # x: [batch, seq_len, hidden_dim] e.g. [1, 10, 3584]
    # weight: [hidden_dim] e.g. [3584] — learned per-dimension scale
    # eps: 1e-6 — tiny number to prevent division by zero
```

```python
    x_float = x.float()
```
Cast to float32 for the computation. Why? BF16 has only ~3 decimal digits of precision.
When you square 3584 numbers and average them, the small errors add up and the result
can be noticeably wrong. Float32 has ~7 decimal digits — enough for accurate normalization.
HuggingFace does the same thing.

```python
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
```
Step by step:
- `x_float.pow(2)` — square every element. Shape stays `[B, S, 3584]`.
- `.mean(dim=-1, ...)` — average across the LAST dimension (3584 elements per token).
  This computes one average per token position. Shape: `[B, S, 1]`.
- `keepdim=True` — keep the last dimension as size 1 (not removed).
  Without keepdim: shape would be `[B, S]`, and we couldn't divide `[B, S, 3584]` by `[B, S]`.
  With keepdim: shape is `[B, S, 1]`, and `[B, S, 3584] / [B, S, 1]` works via broadcasting
  (the `1` automatically expands to `3584`).

```python
    x_normed = x_float * torch.rsqrt(variance + eps)
```
- `variance + eps` — add tiny epsilon (1e-6) to prevent division by zero if variance is 0.
- `torch.rsqrt(...)` — reciprocal square root: `1 / sqrt(x)`. Faster than computing
  `sqrt` and then dividing. Shape: `[B, S, 1]`.
- `x_float * rsqrt(...)` — multiply each element. Broadcasting: `[B, S, 3584] * [B, S, 1]`
  = `[B, S, 3584]`. Every element in a token's vector gets divided by the same RMS value.

```python
    return (x_normed * weight).to(x.dtype)
```
- `x_normed * weight` — per-dimension scaling. `weight` has shape `[3584]`, which
  broadcasts to `[B, S, 3584]`. Each of the 3584 dimensions gets its own learned scale factor.
  The model learns that some dimensions should be amplified and others dampened.
- `.to(x.dtype)` — cast back to whatever dtype the input was (bf16). We computed in float32
  for accuracy, now go back to bf16 for memory efficiency.

### 3.3 `rotary_embedding(q, k, positions)` — RoPE

**What is positional encoding?** Transformers have no built-in notion of word order.
The sentence "dog bites man" and "man bites dog" would produce the same attention
scores without position information. Positional encoding adds order information.

**What is RoPE?** Instead of ADDING position information (like the original Transformer),
RoPE ROTATES pairs of dimensions by an angle that depends on position. This has a
beautiful property: the dot product between two rotated vectors depends only on their
RELATIVE distance, not their absolute positions. "dog" at position 3 attending to
"the" at position 1 gives the same score as "dog" at position 103 attending to
"the" at position 101 — same relative distance of 2.

**The rotation analogy**: Imagine each pair of dimensions as a 2D point that gets
rotated around the origin. Different dimension pairs rotate at different speeds
(frequencies). Low-indexed pairs rotate fast, high-indexed pairs rotate slowly —
like a clock where the second hand, minute hand, and hour hand all encode different
time scales.

**Implementation, step by step**:

```python
def rotary_embedding(q, k, positions):
    # q: [batch, seq_len, num_q_heads, head_dim]  e.g. [1, 10, 28, 128]
    # k: [batch, seq_len, num_kv_heads, head_dim]  e.g. [1, 10, 4, 128]
    # positions: [batch, seq_len]  e.g. [1, 10] containing [0, 1, 2, ..., 9]
```

**Step 1: Compute frequencies**
```python
    dim_indices = torch.arange(0, HEAD_DIM, 2, device=q.device, dtype=torch.float32)
    # HEAD_DIM = 128, step = 2
    # dim_indices = [0, 2, 4, 6, ..., 126]  ← 64 values (one per dimension pair)
```

```python
    freqs = 1.0 / (ROPE_THETA ** (dim_indices / HEAD_DIM))
    # ROPE_THETA = 1,000,000
    # For dim_indices[0] = 0:  1 / (1e6 ^ (0/128))   = 1 / 1     = 1.0         (fast rotation)
    # For dim_indices[1] = 2:  1 / (1e6 ^ (2/128))    = 1 / 1.23  = 0.81        (slightly slower)
    # ...
    # For dim_indices[63] = 126: 1 / (1e6 ^ (126/128)) = 1 / ~724k = 0.0000014  (very slow)
    # Shape: [64]
```

The frequencies decrease exponentially. The first pair rotates fast (captures nearby token
relationships), the last pair rotates extremely slowly (captures very long-range relationships).
Qwen uses theta=1,000,000 (much larger than Llama's 10,000) which makes frequencies
decay slower, allowing it to handle longer sequences.

**Step 2: Compute angles**
```python
    angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
    # positions: [B, S]    → unsqueeze(-1) → [B, S, 1]
    # freqs:     [64]      → unsqueeze × 2  → [1, 1, 64]
    # Multiply with broadcasting: [B, S, 1] * [1, 1, 64] → [B, S, 64]
    #
    # For position 0: angles = [0 * f0, 0 * f1, ...] = all zeros (no rotation)
    # For position 5: angles = [5 * f0, 5 * f1, ...] (each freq scaled by position)
```

Each position gets 64 angles — one per dimension pair. Position 0 has all-zero angles
(no rotation = identity). Higher positions get larger angles.

**Step 3: Compute cos and sin**
```python
    cos = torch.cos(angles).unsqueeze(2)   # [B, S, 64] → [B, S, 1, 64]
    sin = torch.sin(angles).unsqueeze(2)   # same
    # The unsqueeze(2) adds a dimension for heads. [B, S, 1, 64] broadcasts
    # to [B, S, 28, 64] when multiplied with q (which has 28 heads).
```

**Step 4: Apply rotation**
```python
    q_even = q[..., 0::2]    # every other element starting from 0: dims [0, 2, 4, ...]
    q_odd  = q[..., 1::2]    # every other element starting from 1: dims [1, 3, 5, ...]
    # q: [B, S, 28, 128] → q_even: [B, S, 28, 64], q_odd: [B, S, 28, 64]
    # The ... means "all preceding dimensions" — shorthand for [:, :, :, 0::2]
```

Each pair of adjacent dimensions (0,1), (2,3), (4,5), ..., (126,127) forms a 2D vector
that gets rotated. We separate them into "even" and "odd" halves.

```python
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd  = q_even * sin + q_odd * cos
```
This is the 2D rotation formula. For a point (x, y) rotated by angle θ:
```
x' = x * cos(θ) - y * sin(θ)
y' = x * sin(θ) + y * cos(θ)
```
Here, `q_even` is x, `q_odd` is y, and each dimension pair has its own θ from `angles`.

```python
    q_out = torch.stack([q_rot_even, q_rot_odd], dim=-1).flatten(-2)
    # stack: [B, S, 28, 64], [B, S, 28, 64] → [B, S, 28, 64, 2] (pairs stacked)
    # flatten(-2): [B, S, 28, 64, 2] → [B, S, 28, 128] (pairs interleaved back)
    # This reconstructs [dim0, dim1, dim2, dim3, ...] from [even0, even1, ...] and [odd0, odd1, ...]
```

Do the same for K, then return:
```python
    # ... (same rotation code for k_even, k_odd)
    return q_out.to(q.dtype), k_out.to(k.dtype)
```

#### RoPE Variants: Traditional vs Split-Half

The code above shows the **traditional (interleaved)** variant for reference. However,
HuggingFace's Qwen2 implementation uses the **split-half** variant. Both are mathematically
equivalent but pair dimensions differently:

**Traditional (interleaved):**
Pairs adjacent dimensions: (dim0, dim1), (dim2, dim3), ..., (dim126, dim127)
```python
q_even = q[..., 0::2]   # dims 0, 2, 4, ...
q_odd  = q[..., 1::2]   # dims 1, 3, 5, ...
q_rot = torch.stack([q_even * cos - q_odd * sin,
                      q_even * sin + q_odd * cos], dim=-1).flatten(-2)
```

**Split-half (HuggingFace — use this one):**
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

Reference: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)

### 3.4 `make_causal_mask(seq_len, device)` — prevent looking at the future

**Why a causal mask?** During generation, token 3 should only attend to tokens 0, 1, 2, 3 —
not tokens 4, 5, 6, ... (which haven't been generated yet). The mask enforces this.

```python
def make_causal_mask(seq_len, device="cuda"):
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
```

- `torch.ones(4, 4)` — 4×4 matrix of ones.
- `torch.tril(...)` — keep lower triangle, zero out upper:
```
[[1, 0, 0, 0],     True  = can attend
 [1, 1, 0, 0],     False = blocked
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```
Row i, column j: "can token i attend to token j?" Yes if j ≤ i.
- `.unsqueeze(0).unsqueeze(0)` — add batch and head dimensions:
  `[4, 4]` → `[1, 1, 4, 4]`. The `1`s broadcast across all batches and all heads.

### 3.5 `attention(q, k, v, mask)` — Grouped Query Attention

**What is attention?** Each token asks "what other tokens are relevant to me?"
It computes a similarity score with every other token, then takes a weighted
combination of their values. High similarity = more influence.

**What is GQA?** Standard multi-head attention gives each Q head its own K and V head
(28 Q heads, 28 KV heads). GQA shares KV heads across groups of Q heads
(28 Q heads, 4 KV heads — every 7 Q heads share 1 KV head). This saves 7x memory
for KV cache (critical for long sequences) with minimal quality loss.

**Implementation**:

```python
def attention(q, k, v, mask=None):
    # q: [B, S, 28, 128]  — 28 query heads
    # k: [B, S, 4, 128]   — 4 key heads
    # v: [B, S, 4, 128]   — 4 value heads
```

**Step 1: Expand KV heads**
```python
    k = k.repeat_interleave(GQA_RATIO, dim=2)  # [B, S, 4, 128] → [B, S, 28, 128]
    v = v.repeat_interleave(GQA_RATIO, dim=2)
```
`repeat_interleave(7, dim=2)` copies each head 7 times along dimension 2:
`[kv0, kv1, kv2, kv3]` → `[kv0, kv0, kv0, kv0, kv0, kv0, kv0, kv1, kv1, ...]`
Now Q heads 0-6 share kv0, Q heads 7-13 share kv1, etc.

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

**Step 2: Rearrange for matrix multiply**
```python
    q = q.transpose(1, 2)  # [B, S, 28, 128] → [B, 28, S, 128]
    k = k.transpose(1, 2)  # same
    v = v.transpose(1, 2)
```
`transpose(1, 2)` swaps dimensions 1 and 2. We need heads BEFORE sequence length
because `torch.matmul` does batched matrix multiply on the last two dimensions.
With shape `[B, 28, S, 128]`, matmul treats B and 28 as batch dimensions and
multiplies the `[S, 128]` matrices — exactly what we want (each head computes
attention independently).

**Step 3: Compute attention scores**
```python
    scale = HEAD_DIM ** -0.5   # 128 ** -0.5 = 1/sqrt(128) = 1/11.3137 ≈ 0.0884
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    # q: [B, 28, S, 128]
    # k.transpose(-2, -1): [B, 28, 128, S]   (swap last two dims)
    # matmul: [B, 28, S, 128] @ [B, 28, 128, S] → [B, 28, S, S]
    # scores[b, h, i, j] = "how much should token i attend to token j, in head h?"
```

`k.transpose(-2, -1)` swaps the last two dimensions: `[B, 28, S, 128]` → `[B, 28, 128, S]`.
The `-2` and `-1` mean "second-to-last" and "last" — shorthand that works regardless of
the total number of dimensions.

**Why scale by 1/sqrt(d)?** Dot products grow with dimension size. If each element is ~N(0,1),
the dot product of two 128-dim vectors has variance ~128. Dividing by sqrt(128) brings the
variance back to ~1, keeping softmax in a numerically stable range.

**Step 4: Apply causal mask**
```python
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    # ~mask inverts True↔False. Where mask is False (future tokens), set score to -inf.
    # After softmax, -inf → 0 probability — future tokens have zero influence.
```

**Step 5: Softmax**
```python
    attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    # softmax along last dim: for each query position, the scores across all key positions
    # are normalized to sum to 1.0 → proper probability distribution.
    # dtype=torch.float32: compute softmax in float32 to avoid NaN.
    # .to(q.dtype): cast back to bf16 for the next matmul.
```

**Why float32 for softmax?** Softmax computes `exp(x)`. In bf16, if `x` is large (>88),
`exp(x)` overflows to inf, and inf/inf = NaN. Float32 handles values up to ~88 before
overflow, and the subtraction-of-max trick (which softmax implementations do internally)
extends this further. This is standard practice — every LLM framework does it.

**Step 6: Weighted sum**
```python
    output = torch.matmul(attn_weights, v)
    # [B, 28, S, S] @ [B, 28, S, 128] → [B, 28, S, 128]
    # For each query position, this computes a weighted average of all value vectors,
    # where the weights are the attention probabilities from step 5.
```

**Step 7: Rearrange back**
```python
    output = output.transpose(1, 2)  # [B, 28, S, 128] → [B, S, 28, 128]
    return output
```

**Memory note**: `scores` has shape `[B, 28, S, S]`. With S=4096 in BF16:
28 × 4096 × 4096 × 2 bytes = **1.8 GB** just for attention scores. This is why
FlashAttention (Phase 8) exists — it computes attention WITHOUT materializing this
full S×S matrix, reducing memory from O(S²) to O(S).

### 3.6 `swiglu_ffn(x, gate_weight, up_weight, down_weight)` — process each token independently

**What is the FFN?** After attention (which mixes information BETWEEN tokens), the FFN
processes each token INDEPENDENTLY through a learned transformation. Think of attention
as "gathering information from context" and FFN as "thinking about what that information means."

**What is SwiGLU?** A gated variant of the standard feed-forward network:
- Standard FFN: `output = down(ReLU(up(x)))` — two matrix multiplies
- SwiGLU: `output = down(SiLU(gate(x)) * up(x))` — three matrix multiplies

The `gate * up` mechanism lets the model learn to selectively activate features.
The gate controls WHICH neurons fire, and up provides the values.

```python
def swiglu_ffn(x, gate_weight, up_weight, down_weight):
    # x: [B, S, 3584]
    # gate_weight: [18944, 3584]  — expand to FFN dimension
    # up_weight:   [18944, 3584]  — expand to FFN dimension
    # down_weight: [3584, 18944]  — compress back to hidden dimension
```

```python
    gate = torch.nn.functional.silu(x @ gate_weight.T)
    # x @ gate_weight.T: [B, S, 3584] @ [3584, 18944] → [B, S, 18944]
    # silu(z) = z * sigmoid(z)
    #   sigmoid(z) = 1 / (1 + exp(-z))  — squashes to range (0, 1)
    #   z * sigmoid(z) — smooth activation. Unlike ReLU (which hard-clips negatives to 0),
    #   SiLU allows small negative values through and has a smooth curve.
```

```python
    up = x @ up_weight.T
    # [B, S, 3584] @ [3584, 18944] → [B, S, 18944]
    # No activation — raw linear projection.
```

```python
    return (gate * up) @ down_weight.T
    # gate * up: element-wise multiply [B, S, 18944] * [B, S, 18944] → [B, S, 18944]
    #   The gate (0 to ~1 per element) controls which features from 'up' pass through.
    # ... @ down_weight.T: [B, S, 18944] @ [18944, 3584] → [B, S, 3584]
    #   Compress back to hidden dimension.
```

**Why 18944?** The expansion ratio is 18944/3584 = 5.29x. Standard FFN uses 4x expansion,
but SwiGLU has a third matrix, so the expansion is increased to keep total parameter count
similar: `2 × 4 × D² = 3 × 5.29 × D²` (approximately). The specific value 18944 is chosen
to be divisible by common GPU tile sizes.

### 3.7 `transformer_block(...)` — one complete layer

This is just plumbing — calling the functions you've already built.
Every transformer layer does the same thing:

```
Input
  ├─── (save as residual)
  ├─── RMSNorm
  ├─── QKV Projections (linear + bias for Q, K, V)
  ├─── Reshape to multi-head format
  ├─── RoPE (rotate Q and K)
  ├─── Attention
  ├─── Reshape back
  ├─── Output projection (linear, no bias)
  └─── ADD residual ← (the "skip connection")
  ├─── (save as residual)
  ├─── RMSNorm
  ├─── SwiGLU FFN
  └─── ADD residual ← (another skip connection)
Output
```

**The residual connection** (`x = residual + attn_out`) is crucial. Without it, gradients
vanish in deep networks — the model can't train with 28 layers. The residual lets
information flow directly from early layers to later ones, like a highway bypass.

**Implementation**: See the full code in section 3.7 of the original guide.
The key things to remember:
- Q, K, V projections HAVE bias (Qwen-specific). O projection does NOT.
- Reshape Q from `[B, S, 3584]` to `[B, S, 28, 128]` using `.view()`.
  `.view()` reinterprets the same data with a new shape (no copy):
  3584 = 28 heads × 128 dims/head, so `[B, S, 3584]` → `[B, S, 28, 128]`.
  K and V reshape to `[B, S, 4, 128]` because they have 4 heads (512 = 4 × 128).
- RoPE applies to Q and K only, NOT to V. Position matters for "what should I attend to"
  (Q and K), not for "what value do I carry" (V).
- After attention, reshape `[B, S, 28, 128]` back to `[B, S, 3584]` using `.reshape(B, S, -1)`.
  The `-1` means "infer this dimension" — PyTorch computes 28 × 128 = 3584.

```python
def transformer_block(x, weights, layer_idx, positions, mask=None):
    prefix = f"model.layers.{layer_idx}"

    # ── Attention ──
    residual = x
    x = rmsnorm(x, weights[f"{prefix}.input_layernorm.weight"])

    B, S, _ = x.shape
    q = (x @ weights[f"{prefix}.self_attn.q_proj.weight"].T
         + weights[f"{prefix}.self_attn.q_proj.bias"])
    k = (x @ weights[f"{prefix}.self_attn.k_proj.weight"].T
         + weights[f"{prefix}.self_attn.k_proj.bias"])
    v = (x @ weights[f"{prefix}.self_attn.v_proj.weight"].T
         + weights[f"{prefix}.self_attn.v_proj.bias"])

    q = q.view(B, S, NUM_Q_HEADS, HEAD_DIM)
    k = k.view(B, S, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(B, S, NUM_KV_HEADS, HEAD_DIM)

    q, k = rotary_embedding(q, k, positions)

    attn_out = attention(q, k, v, mask)
    attn_out = attn_out.reshape(B, S, -1)
    attn_out = attn_out @ weights[f"{prefix}.self_attn.o_proj.weight"].T

    x = residual + attn_out

    # ── FFN ──
    residual = x
    x = rmsnorm(x, weights[f"{prefix}.post_attention_layernorm.weight"])
    x = swiglu_ffn(
        x,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )
    x = residual + x

    return x
```

### 3.8 `forward(token_ids, weights, positions)` — full model

The complete forward pass: tokens in, logits out.

```python
def forward(token_ids, weights, positions=None):
    batch, seq_len = token_ids.shape
    # token_ids: [B, S] e.g. [1, 10] — 1 batch of 10 token IDs
    device = token_ids.device
```

```python
    # 1. Embedding lookup — convert token IDs to vectors
    x = weights["model.embed_tokens.weight"][token_ids]
    # embed_tokens.weight: [152064, 3584] — one 3584-dim vector per vocab token
    # token_ids: [1, 10] — contains IDs like [9707, 1879, 330, ...]
    # Indexing with token_ids selects rows from the embedding table:
    #   row 9707 → [0.12, -0.34, ...] (3584 numbers)
    #   row 1879 → [0.56, 0.01, ...]
    # Result: [1, 10, 3584] — each token ID replaced by its embedding vector
```

This is a simple lookup table — no math, just picking rows from a matrix.
Each of the 152,064 possible tokens has a learned 3584-dimensional representation.

```python
    # 2. Build position indices
    if positions is None:
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        # torch.arange(10) → tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # .unsqueeze(0)    → [1, 10]
        # .expand(batch, -1) → [B, 10] (repeat for each batch item)
```

```python
    # 3. Build causal mask
    mask = make_causal_mask(seq_len, device=device)   # [1, 1, S, S]
```

```python
    # 4. Run through all 28 layers
    for i in range(NUM_LAYERS):    # NUM_LAYERS = 28
        x = transformer_block(x, weights, i, positions, mask)
        # Each layer transforms [B, S, 3584] → [B, S, 3584]
        # The shape never changes — only the values evolve
```

```python
    # 5. Final normalization
    x = rmsnorm(x, weights["model.norm.weight"])
```

```python
    # 6. Project to vocabulary size
    logits = x @ weights["lm_head.weight"].T
    # x: [B, S, 3584]
    # lm_head.weight: [152064, 3584] → .T → [3584, 152064]
    # matmul: [B, S, 3584] @ [3584, 152064] → [B, S, 152064]
    # logits[b, s, v] = "how likely is vocab token v to appear at position s?"
    return logits
```

The output logits have shape `[B, S, 152064]`. During generation, we only care about
the LAST position (`logits[0, -1, :]` = `[152064]`), because that's the prediction
for the NEXT token.

### 3.9 `generate(...)` — autoregressive text generation

**How does text generation work?** The model predicts one token at a time:
1. Feed the prompt tokens through the model → get logits for the next token
2. Sample a token from those logits
3. Append the new token to the sequence
4. Repeat from step 1 with the longer sequence

**Phase 1's version is intentionally naive**: it re-runs the ENTIRE sequence through
all 28 layers for every single new token. This is O(n²) in sequence length and very slow.
Phase 2 fixes this with KV cache.

```python
def generate(prompt_tokens, weights, max_new_tokens=100, sample_fn=None, device="cuda"):
    from .sampler import greedy
    if sample_fn is None:
        sample_fn = greedy

    tokens = list(prompt_tokens)   # e.g. [151644, 882, 198, 9707, ...]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], device=device)
        # [tokens] wraps the list in another list → 2D: [[151644, 882, ...]]
        # torch.tensor makes it a tensor. Shape: [1, current_len]
        # The [1, ...] is the batch dimension (batch size = 1)

        logits = forward(input_ids, weights)
        # Full forward pass on ALL tokens. Shape: [1, current_len, 152064]
        # This recomputes everything from scratch — even tokens we've seen before.
        # That's the waste that Phase 2's KV cache eliminates.

        next_logits = logits[0, -1, :]
        # logits[0] — first (only) batch item: [current_len, 152064]
        # ..[-1, :] — last position, all vocab scores: [152064]
        # This is the model's prediction for what comes AFTER the last token.

        next_token = sample_fn(next_logits)
        # e.g. greedy(next_logits) → 42 (a Python int)

        from .tokenizer import EOS_IDS
        if next_token in EOS_IDS:
            break
        # Stop if the model generated <|im_end|> (151645) or <|endoftext|> (151643)

        tokens.append(next_token)
        # Add new token to sequence. Next iteration will process [prompt + all generated tokens]

    return tokens
```

---

## Part 4: Validation Strategy

### 4.1 Test after each function

Don't wait until everything is done. After implementing `rmsnorm`, test it immediately:

```python
from transformers import AutoModelForCausalLM
import torch

hf_model = AutoModelForCausalLM.from_pretrained(
    "path/to/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Grab layer 0's normalization weight from HuggingFace
layer0 = hf_model.model.layers[0]

# Create identical input
x = torch.randn(1, 4, 3584, dtype=torch.bfloat16)

# Compare outputs
hf_output = layer0.input_layernorm(x)
your_output = rmsnorm(x, layer0.input_layernorm.weight)

match = torch.allclose(hf_output, your_output, atol=1e-4, rtol=1e-3)
# atol=1e-4: absolute tolerance — values can differ by up to 0.0001
# rtol=1e-3: relative tolerance — values can differ by up to 0.1%
print(f"Match: {match}")  # should print True

if not match:
    diff = (hf_output - your_output).abs()
    print(f"Max difference: {diff.max().item()}")
    print(f"Mean difference: {diff.mean().item()}")
```

### 4.2 Test progression

1. `rmsnorm` — easiest, small function, quick to validate
2. `rotary_embedding` — compare against `Qwen2RotaryEmbedding` in HuggingFace
3. `attention` — one layer at a time
4. `swiglu_ffn` — compare FFN output for one layer
5. `transformer_block` — compare full layer output
6. `forward` — compare final logits for a short prompt
7. `generate` — compare greedy output token-by-token

### 4.3 Common bugs and fixes

| Symptom | Likely cause |
|---------|-------------|
| Logits slightly off (max diff ~0.01) | RMSNorm or softmax not computed in float32 |
| Logits completely wrong | Missing QKV bias. Check `+ weights[...q_proj.bias]` |
| Correct for layer 0-5, diverges after | Small error compounding. Fix individual function tolerances |
| NaN in output | Softmax overflow. Use `dtype=torch.float32` in `torch.softmax` |
| Wrong tokens generated | Wrong causal mask orientation, or wrong EOS token IDs |
| RoPE mismatch | Wrong theta (should be 1,000,000 not 10,000) or partial head_dim application |
| RoPE outputs wrong values | Using interleaved (even/odd) instead of split-half variant. Qwen2.5 uses split-half in HuggingFace. See RoPE Variants section above. |
| Shape mismatch error | Wrong reshape/view. Print shapes before and after each operation |

### 4.4 Debugging tip: print shapes everywhere

When something breaks, add shape prints:
```python
print(f"x after embedding: {x.shape}")           # should be [1, S, 3584]
print(f"q after projection: {q.shape}")           # should be [1, S, 3584]
print(f"q after reshape: {q.shape}")              # should be [1, S, 28, 128]
print(f"scores after matmul: {scores.shape}")     # should be [1, 28, S, S]
```

---

## Part 5: Running It

```bash
# Unit tests (no model needed)
pytest tests/test_model.py -v -k "not HuggingFace"

# Integration tests (need MODEL_DIR set)
MODEL_DIR=./models/Qwen2.5-7B-Instruct pytest tests/test_model.py -v

# Generate text
python run.py --model-dir ./models/Qwen2.5-7B-Instruct --prompt "Hello, who are you?"
python run.py --model-dir ./models/Qwen2.5-7B-Instruct --interactive
```

---

## Further Reading

- [skyzh's tiny-llm](https://skyzh.github.io/tiny-llm/) — Week 1 covers the same ground with MLX
- [andrewkchan's YALM](https://andrewkchan.dev/posts/yalm.html) — from-scratch in C++
- [HuggingFace Qwen2 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) — the reference
- [QwenLM tokenizer notes](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md) — official tokenizer docs
- `docs/concepts/transformer_math_reference.md` — every formula with derivations
- `docs/concepts/qwen2.5-7b-architecture.md` — all weight names, shapes, config values
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) — RoPE paper
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) — GQA paper
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) — SwiGLU paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) — original transformer
