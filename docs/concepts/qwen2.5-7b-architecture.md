# Qwen2.5-7B-Instruct: Complete Architecture Reference

> Technical specifications for building a from-scratch inference engine.
> All values sourced from the official HuggingFace model repository config files.

---

## 1. Architecture Specifications

### 1.1 Model Config (`config.json`)

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 131072,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}
```

### 1.2 Core Dimensions

| Parameter | Value | Notes |
|---|---|---|
| **Hidden dimension** (`hidden_size`) | 3,584 | Token embedding and residual stream width |
| **Intermediate FFN dimension** (`intermediate_size`) | 18,944 | SwiGLU FFN hidden dim (5.29x expansion) |
| **Number of layers** (`num_hidden_layers`) | 28 | Transformer decoder blocks |
| **Number of Q heads** (`num_attention_heads`) | 28 | Query heads per layer |
| **Number of KV heads** (`num_key_value_heads`) | 4 | Key/Value heads per layer (GQA) |
| **GQA group size** | 7 | 28 Q heads / 4 KV heads = 7 Q heads share each KV head |
| **Head dimension** | 128 | `hidden_size / num_attention_heads` = 3584 / 28 = 128 |
| **Vocabulary size** (`vocab_size`) | 152,064 | Including special tokens |
| **Max position embeddings** | 32,768 | Base context window (32K) |
| **RoPE theta** (`rope_theta`) | 1,000,000.0 | Base frequency for rotary embeddings |
| **RMSNorm epsilon** (`rms_norm_eps`) | 1e-6 | Normalization stability constant |
| **Tie word embeddings** | `false` | Separate `embed_tokens` and `lm_head` weights |

### 1.3 Total Parameter Count

| Category | Parameters | Notes |
|---|---|---|
| **Total** | ~7.61B | All parameters |
| **Non-embedding** | ~6.53B | Excluding embed_tokens and lm_head |
| **Embedding (embed_tokens)** | ~545M | 152,064 x 3,584 |
| **LM head (lm_head)** | ~545M | 152,064 x 3,584 (untied, separate weights) |

### 1.4 Context Window and RoPE Scaling

- **Base context**: 32,768 tokens (from `max_position_embeddings`)
- **Extended context**: Up to 131,072 tokens (128K) using YaRN RoPE scaling
- **YaRN config** (add to config.json when needed):
  ```json
  {
    "rope_scaling": {
      "factor": 4.0,
      "original_max_position_embeddings": 32768,
      "type": "yarn"
    }
  }
  ```
- **`sliding_window`: 131,072** is defined but **`use_sliding_window`: false** -- sliding window attention is NOT used by default
- **`max_window_layers`: 28** -- all layers would use windowed attention if enabled

**For Phase 1 implementation**: Implement standard RoPE with `rope_theta=1e6` and `max_position_embeddings=32768`. YaRN scaling is an advanced topic for later phases.

---

## 2. Architecture Details (Layer by Layer)

### 2.1 Forward Pass Pipeline

```
Input token IDs: [batch, seq_len]
        |
  embed_tokens: lookup -> [batch, seq_len, 3584]
        |
  +----- 28x Transformer Blocks -----+
  |                                   |
  |  input_layernorm (RMSNorm)        |
  |        |                          |
  |  Self-Attention (GQA)             |
  |    q_proj: [3584] -> [3584]       |  (28 heads x 128 dim, WITH bias)
  |    k_proj: [3584] -> [512]        |  (4 heads x 128 dim, WITH bias)
  |    v_proj: [3584] -> [512]        |  (4 heads x 128 dim, WITH bias)
  |    Apply RoPE to Q, K             |
  |    Compute GQA attention          |
  |    o_proj: [3584] -> [3584]       |  (NO bias)
  |        |                          |
  |  + Residual connection            |
  |        |                          |
  |  post_attention_layernorm (RMSNorm)|
  |        |                          |
  |  MLP (SwiGLU)                     |
  |    gate_proj: [3584] -> [18944]   |  (NO bias)
  |    up_proj:   [3584] -> [18944]   |  (NO bias)
  |    act_fn:    SiLU(gate) * up     |
  |    down_proj: [18944] -> [3584]   |  (NO bias)
  |        |                          |
  |  + Residual connection            |
  +-----------------------------------+
        |
  model.norm (RMSNorm): [batch, seq_len, 3584]
        |
  lm_head: [3584] -> [152064]  (NO bias)
        |
  Logits: [batch, seq_len, 152064]
```

### 2.2 RMSNorm

```python
# RMSNorm (used for input_layernorm, post_attention_layernorm, model.norm)
# Parameters: weight (gamma), shape [3584]
# Epsilon: 1e-6

def rmsnorm(x, weight, eps=1e-6):
    # x: [batch, seq_len, 3584]
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
```

### 2.3 Attention (Grouped-Query Attention with RoPE)

**Key implementation notes**:
- Q, K, V projections are **separate** (not fused into a single QKV projection in the weights)
- Q, K, V projections **have bias** (`bias=True`)
- O projection has **no bias** (`bias=False`)
- RoPE is applied to Q and K **after** projection, **before** attention computation
- GQA: each KV head is shared by 7 query heads

```python
# Per layer, attention projection shapes:
# q_proj.weight: [3584, 3584]   q_proj.bias: [3584]
# k_proj.weight: [512, 3584]    k_proj.bias: [512]
# v_proj.weight: [512, 3584]    v_proj.bias: [512]
# o_proj.weight: [3584, 3584]   (no bias)

# After projection:
# Q: [batch, seq_len, 28, 128] -> apply RoPE -> [batch, seq_len, 28, 128]
# K: [batch, seq_len, 4, 128]  -> apply RoPE -> [batch, seq_len, 4, 128]
# V: [batch, seq_len, 4, 128]

# GQA expansion (repeat KV heads for grouped attention):
# K: [batch, seq_len, 4, 128] -> expand to [batch, seq_len, 28, 128]  (each KV head repeated 7x)
# V: [batch, seq_len, 4, 128] -> expand to [batch, seq_len, 28, 128]  (each KV head repeated 7x)

# Standard scaled dot-product attention:
# attn = softmax(Q @ K^T / sqrt(128)) @ V
# Output: [batch, seq_len, 28, 128] -> reshape to [batch, seq_len, 3584] -> o_proj
```

### 2.4 Rotary Position Embedding (RoPE)

```python
# RoPE parameters:
# - theta (base frequency): 1,000,000.0
# - head_dim: 128 (applied to the full head dimension)
# - Rotation applied to pairs of dimensions: (0,1), (2,3), ..., (126,127)

# Frequency computation:
# For dimension pair i (0 to 63):
#   freq_i = 1.0 / (theta ^ (2i / head_dim))
#   freq_i = 1.0 / (1000000.0 ^ (2i / 128))

# For position pos:
#   angle = pos * freq_i
#   x_rotated[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
#   x_rotated[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)
```

### 2.5 MLP (SwiGLU)

```python
# SwiGLU FFN -- uses THREE projections instead of two
# gate_proj.weight: [18944, 3584]  (no bias)
# up_proj.weight:   [18944, 3584]  (no bias)
# down_proj.weight: [3584, 18944]  (no bias)

def swiglu_ffn(x, gate_proj, up_proj, down_proj):
    # x: [batch, seq_len, 3584]
    gate = gate_proj(x)          # [batch, seq_len, 18944]
    up = up_proj(x)              # [batch, seq_len, 18944]
    hidden = F.silu(gate) * up   # Element-wise SiLU gating
    output = down_proj(hidden)   # [batch, seq_len, 3584]
    return output
```

**Note on expansion ratio**: The intermediate_size of 18,944 with hidden_size of 3,584 gives a ratio of ~5.29x. This is higher than the standard 4x because SwiGLU uses 3 projections: the effective parameter count per FFN is `3 * hidden * intermediate` vs the standard `2 * hidden * 4*hidden`, so the intermediate dimension is scaled to keep total FFN parameters comparable: `18944 / 3584 ≈ 5.29` compensates for the 3-projection design.

---

## 3. Safetensors Weight Layout

### 3.1 File Structure

The model is split across **4 safetensors files**, totaling **~15.2 GB** (BFloat16):

| File | Layers | Approximate Size |
|---|---|---|
| `model-00001-of-00004.safetensors` | Layers 0-5 + part of 6, embed_tokens | ~3.8 GB |
| `model-00002-of-00004.safetensors` | Layers 6-14 | ~3.8 GB |
| `model-00003-of-00004.safetensors` | Layers 14-22 | ~3.8 GB |
| `model-00004-of-00004.safetensors` | Layers 22-27, model.norm, lm_head | ~3.8 GB |

### 3.2 Complete Weight Tensor Names

**Global tensors:**
- `model.embed_tokens.weight` -- shape: `[152064, 3584]`
- `model.norm.weight` -- shape: `[3584]`
- `lm_head.weight` -- shape: `[152064, 3584]`

**Per-layer tensors** (for layer `i` in 0..27):

| Tensor Name | Shape | Bytes (BF16) | Has Bias |
|---|---|---|---|
| `model.layers.{i}.input_layernorm.weight` | `[3584]` | 7,168 | No |
| `model.layers.{i}.self_attn.q_proj.weight` | `[3584, 3584]` | 25,690,112 | Yes |
| `model.layers.{i}.self_attn.q_proj.bias` | `[3584]` | 7,168 | - |
| `model.layers.{i}.self_attn.k_proj.weight` | `[512, 3584]` | 3,670,016 | Yes |
| `model.layers.{i}.self_attn.k_proj.bias` | `[512]` | 1,024 | - |
| `model.layers.{i}.self_attn.v_proj.weight` | `[512, 3584]` | 3,670,016 | Yes |
| `model.layers.{i}.self_attn.v_proj.bias` | `[512]` | 1,024 | - |
| `model.layers.{i}.self_attn.o_proj.weight` | `[3584, 3584]` | 25,690,112 | No |
| `model.layers.{i}.post_attention_layernorm.weight` | `[3584]` | 7,168 | No |
| `model.layers.{i}.mlp.gate_proj.weight` | `[18944, 3584]` | 135,790,592 | No |
| `model.layers.{i}.mlp.up_proj.weight` | `[18944, 3584]` | 135,790,592 | No |
| `model.layers.{i}.mlp.down_proj.weight` | `[3584, 18944]` | 135,790,592 | No |

**Total tensors**: 323 (12 per layer x 28 layers + 3 global = 339, minus bias-only entries counted separately = 323 tensor entries in safetensors index)

### 3.3 Per-Layer Parameter Count

| Component | Parameters | Notes |
|---|---|---|
| input_layernorm | 3,584 | weight only |
| q_proj (weight + bias) | 12,849,536 | 3584*3584 + 3584 |
| k_proj (weight + bias) | 1,835,520 | 512*3584 + 512 |
| v_proj (weight + bias) | 1,835,520 | 512*3584 + 512 |
| o_proj (weight only) | 12,845,056 | 3584*3584 |
| post_attention_layernorm | 3,584 | weight only |
| gate_proj | 67,895,296 | 18944*3584 |
| up_proj | 67,895,296 | 18944*3584 |
| down_proj | 67,895,296 | 3584*18944 |
| **Layer total** | **233,058,688** | ~233M per layer |
| **28 layers total** | **6,525,643,264** | ~6.53B (matches non-embedding count) |

---

## 4. Tokenizer Details

### 4.1 Tokenizer Type

- **Algorithm**: Byte Pair Encoding (BPE)
- **Implementation**: Based on `tiktoken` (byte-level BPE on UTF-8 bytes)
- **Class**: `Qwen2Tokenizer` (HuggingFace transformers)
- **Vocabulary size**: 152,064 tokens
- **Files**: `vocab.json` + `merges.txt` (HuggingFace format)
- **Model max length**: 131,072 tokens
- **Add BOS token**: `false` (no BOS token prepended automatically)
- **Add prefix space**: `false`

### 4.2 Special Tokens

| Token ID | Token String | Purpose |
|---|---|---|
| 151643 | `<\|endoftext\|>` | End of text / Padding token / BOS token ID in config |
| 151644 | `<\|im_start\|>` | Chat message start (ChatML format) |
| 151645 | `<\|im_end\|>` | Chat message end / EOS token |
| 151646 | `<\|object_ref_start\|>` | Object reference start |
| 151647 | `<\|object_ref_end\|>` | Object reference end |
| 151648 | `<\|box_start\|>` | Bounding box start |
| 151649 | `<\|box_end\|>` | Bounding box end |
| 151650 | `<\|quad_start\|>` | Quad coordinates start |
| 151651 | `<\|quad_end\|>` | Quad coordinates end |
| 151652 | `<\|vision_start\|>` | Vision input start |
| 151653 | `<\|vision_end\|>` | Vision input end |
| 151654 | `<\|vision_pad\|>` | Vision padding |
| 151655 | `<\|image_pad\|>` | Image padding |
| 151656 | `<\|video_pad\|>` | Video padding |
| 151657 | `<tool_call>` | Tool call start |
| 151658 | `</tool_call>` | Tool call end |
| 151659 | `<\|fim_prefix\|>` | Fill-in-middle prefix |
| 151660 | `<\|fim_middle\|>` | Fill-in-middle middle |
| 151661 | `<\|fim_suffix\|>` | Fill-in-middle suffix |
| 151662 | `<\|fim_pad\|>` | Fill-in-middle padding |
| 151663 | `<\|repo_name\|>` | Repository name marker |
| 151664 | `<\|file_sep\|>` | File separator |

**For inference**: The critical tokens are `<|im_start|>` (151644), `<|im_end|>` (151645), and `<|endoftext|>` (151643).

### 4.3 Chat Template (ChatML Format)

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

**With tool calling:**
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

[function definitions in JSON]<|im_end|>
<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Seattle"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 72}
</tool_response><|im_end|>
<|im_start|>assistant
```

### 4.4 Generation Config (Default Sampling Parameters)

```json
{
  "do_sample": true,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "repetition_penalty": 1.05,
  "eos_token_id": [151645, 151643]
}
```

**Note**: Two EOS tokens -- generation stops on either `<|im_end|>` (151645) or `<|endoftext|>` (151643).

---

## 5. Memory Calculations

### 5.1 Model Weight Sizes

| Precision | Bytes/Param | Model Size | Notes |
|---|---|---|---|
| **FP32** | 4 | ~30.4 GB | Not practical for 16GB VRAM |
| **BF16 / FP16** | 2 | ~15.2 GB | Default. Barely fits in 16GB alone |
| **FP8** (E4M3) | 1 | ~7.6 GB | Native RTX 5080 Tensor Core support |
| **INT8** | 1 | ~7.6 GB | Weight-only quantization |
| **FP4** / **INT4** | 0.5 | ~3.8 GB | FP4 native on RTX 5080 (Blackwell) |

Formula: `model_size = num_params * bytes_per_param = 7.61B * bytes_per_param`

### 5.2 KV Cache Size

```
KV cache per token per layer:
  = 2 (K and V) * num_kv_heads * head_dim * bytes_per_element
  = 2 * 4 * 128 * 2  (BF16)
  = 2,048 bytes = 2 KB per token per layer

KV cache per token (all 28 layers):
  = 2,048 * 28
  = 57,344 bytes = 56 KB per token

KV cache for various context lengths (BF16):
  512 tokens:    28.0 MB
  1,024 tokens:  56.0 MB
  2,048 tokens:  112.0 MB
  4,096 tokens:  224.0 MB
  8,192 tokens:  448.0 MB
  16,384 tokens: 896.0 MB
  32,768 tokens: 1,792.0 MB  (1.75 GB)
  65,536 tokens: 3,584.0 MB  (3.5 GB)
  131,072 tokens: 7,168.0 MB (7.0 GB)
```

**With FP8 KV cache** (halved):
```
KV cache per token (all layers): 28 KB
  4,096 tokens:  112 MB
  8,192 tokens:  224 MB
  32,768 tokens: 896 MB
  131,072 tokens: 3.5 GB
```

### 5.3 RTX 5080 VRAM Budget (16 GB)

#### Scenario A: BF16 Weights + BF16 KV Cache

| Component | VRAM | Running Total |
|---|---|---|
| Model weights (BF16) | 15.2 GB | 15.2 GB |
| CUDA/PyTorch overhead | ~0.5 GB | 15.7 GB |
| **DOES NOT FIT** | -- | > 16 GB |

**Verdict**: BF16 model alone nearly fills 16 GB VRAM. No room for KV cache or activations.

#### Scenario B: INT8/FP8 Weights + BF16 KV Cache

| Component | VRAM | Running Total |
|---|---|---|
| Model weights (FP8) | 7.6 GB | 7.6 GB |
| CUDA/PyTorch overhead | ~0.5 GB | 8.1 GB |
| KV cache (4K context, BF16) | 0.22 GB | 8.3 GB |
| KV cache (8K context, BF16) | 0.45 GB | 8.6 GB |
| KV cache (32K context, BF16) | 1.75 GB | 9.9 GB |
| Activation memory (~batch 1) | ~0.5 GB | ~10.4 GB |
| **Remaining headroom** | **~5.6 GB** | -- |

**Verdict**: Very comfortable. Can support up to ~32K context with room to spare.

#### Scenario C: FP4/INT4 Weights + BF16 KV Cache

| Component | VRAM | Running Total |
|---|---|---|
| Model weights (FP4) | 3.8 GB | 3.8 GB |
| CUDA/PyTorch overhead | ~0.5 GB | 4.3 GB |
| KV cache (32K context, BF16) | 1.75 GB | 6.1 GB |
| KV cache (131K context, BF16) | 7.0 GB | 11.3 GB |
| Activation memory | ~0.5 GB | ~11.8 GB |
| **Remaining headroom** | **~4.2 GB** | -- |

**Verdict**: Full 128K context possible with FP4 quantization. Ideal for leveraging Blackwell's native FP4 support.

#### Scenario D: FP8 Weights + FP8 KV Cache

| Component | VRAM | Running Total |
|---|---|---|
| Model weights (FP8) | 7.6 GB | 7.6 GB |
| CUDA/PyTorch overhead | ~0.5 GB | 8.1 GB |
| KV cache (32K context, FP8) | 0.9 GB | 9.0 GB |
| KV cache (65K context, FP8) | 1.75 GB | 9.9 GB |
| Activation memory | ~0.5 GB | ~10.4 GB |
| **Remaining headroom** | **~5.6 GB** | -- |

**Verdict**: Great balance. FP8 everything with long context.

### 5.4 Recommended Configuration for RTX 5080

| Phase | Weights | KV Cache | Max Context | Notes |
|---|---|---|---|---|
| Phase 1 (learning) | FP8 | BF16 | 4-8K | Focus on correctness |
| Phase 3 (kernels) | FP8 | BF16 | 8-16K | Focus on kernel perf |
| Phase 4 (quantization) | FP4 | FP8 | 32K+ | Leverage Blackwell FP4 |
| Phase 5 (serving) | FP8 | FP8 | 32K | Balance speed/context |
| Production target | FP4 | FP8 | 128K | Full capability |

---

## 6. Comparison with Llama-3.1-8B-Instruct

### 6.1 Architecture Comparison

| Feature | Qwen2.5-7B | Llama-3.1-8B | Notes |
|---|---|---|---|
| **Parameters** | 7.61B | 8.03B | Llama is ~5% larger |
| **Hidden size** | 3,584 | 4,096 | Llama has wider hidden dim |
| **Layers** | 28 | 32 | Llama has more layers |
| **Q heads** | 28 | 32 | |
| **KV heads** | 4 | 8 | Qwen has more aggressive GQA |
| **GQA ratio** | 7:1 | 4:1 | Qwen saves more KV cache memory |
| **Head dim** | 128 | 128 | Same |
| **FFN dim** | 18,944 | 14,336 | Qwen has wider FFN |
| **Vocab size** | 152,064 | 128,256 | Qwen has larger vocab |
| **Max context** | 32K (128K w/ YaRN) | 128K (native) | Llama has native 128K |
| **RoPE theta** | 1,000,000 | 500,000 | |
| **RoPE scaling** | YaRN (optional) | llama3 (factor=8) | Different scaling approaches |
| **RMSNorm eps** | 1e-6 | 1e-5 | Qwen uses tighter epsilon |
| **Attention bias** | Q,K,V: **yes**, O: **no** | **None** (all false) | Key difference! |
| **MLP bias** | **None** | **None** | Same |
| **Activation** | SiLU (SwiGLU) | SiLU (SwiGLU) | Same |
| **Tie embeddings** | false | false | Same |
| **Norm type** | RMSNorm (pre-norm) | RMSNorm (pre-norm) | Same |
| **Native dtype** | bfloat16 | bfloat16 | Same |

### 6.2 KV Cache Comparison

| Context | Qwen2.5-7B (BF16) | Llama-3.1-8B (BF16) | Savings |
|---|---|---|---|
| Per token/layer | 2,048 B | 4,096 B | Qwen uses 50% less |
| Per token (all layers) | 56 KB | 128 KB | Qwen uses 56% less (fewer layers too) |
| 4K context | 224 MB | 512 MB | |
| 32K context | 1.75 GB | 4.0 GB | |
| 128K context | 7.0 GB | 16.0 GB | Llama won't fit on 16GB! |

**Qwen's 4 KV heads (vs Llama's 8) and 28 layers (vs 32) mean substantially less KV cache memory**, making it better suited for the 16 GB RTX 5080.

### 6.3 Which is Easier to Implement From Scratch?

**Llama-3.1-8B is slightly simpler** because:
1. **No attention bias**: Llama has `attention_bias=false` and `mlp_bias=false` for all linear layers. Qwen requires bias on Q, K, V projections (but not O), which adds implementation complexity.
2. **More community resources**: Llama has more from-scratch implementations, tutorials, and reference code available.
3. **Cleaner RoPE**: Llama's RoPE uses a well-documented custom scaling (`rope_type: "llama3"`). Qwen's standard RoPE at base 1M is simpler initially, but the YaRN extension is more complex.

**Qwen2.5-7B is the better choice overall** because:
1. **Better VRAM fit**: 50% less KV cache memory due to aggressive GQA (4 KV heads vs 8). Critical for 16 GB VRAM.
2. **No gated access**: Fully open on HuggingFace, no license approval needed (Llama requires Meta's license agreement).
3. **Smaller model**: 7.61B vs 8.03B params, slightly less VRAM for weights.
4. **Better quality per param**: Qwen2.5-7B benchmarks competitively with or above Llama-3.1-8B on most evals.
5. **The bias difference is trivial**: Adding bias to Q/K/V is just `output = weight @ input + bias` instead of `output = weight @ input`. Minimal extra code.
6. **GQA ratio of 7:1 is more educational**: Forces you to properly implement GQA head expansion, teaching you the concept more deeply than Llama's simpler 4:1 ratio.

**Recommendation**: Use **Qwen2.5-7B-Instruct** as the primary target. It fits better on the RTX 5080 and offers richer architectural detail to learn from.

---

## 7. Implementation Checklist for Phase 1

### Weight Loading
- [ ] Parse `model.safetensors.index.json` to get tensor-to-file mapping
- [ ] Load all 4 safetensors files using the `safetensors` library
- [ ] Map tensor names to layer structure (28 layers)
- [ ] Handle separate `embed_tokens` and `lm_head` (not tied)
- [ ] Verify shapes match expected dimensions

### Layer Components
- [ ] RMSNorm: `weight` parameter, `eps=1e-6`
- [ ] Q projection: `weight [3584, 3584]` + `bias [3584]`
- [ ] K projection: `weight [512, 3584]` + `bias [512]`
- [ ] V projection: `weight [512, 3584]` + `bias [512]`
- [ ] O projection: `weight [3584, 3584]` (no bias)
- [ ] RoPE: `theta=1e6`, `head_dim=128`, applied to Q and K
- [ ] GQA: expand 4 KV heads to 28 Q heads (repeat_interleave by 7)
- [ ] Causal mask: upper-triangular mask for autoregressive attention
- [ ] gate_proj: `weight [18944, 3584]` (no bias)
- [ ] up_proj: `weight [18944, 3584]` (no bias)
- [ ] down_proj: `weight [3584, 18944]` (no bias)
- [ ] SiLU activation: `silu(gate) * up`
- [ ] Residual connections (two per layer)

### Generation Loop
- [ ] Prefill: process full prompt in parallel
- [ ] Decode: generate one token at a time
- [ ] Sampling: greedy, temperature, top-k, top-p
- [ ] Stop conditions: check for EOS tokens (151645 or 151643)
- [ ] Chat template: format with `<|im_start|>` / `<|im_end|>` markers

### Validation
- [ ] Compare greedy output token-for-token with HuggingFace `transformers`
- [ ] Check intermediate tensor shapes at every layer
- [ ] Verify RoPE frequencies match reference implementation

---

## Sources

- [Qwen2.5-7B-Instruct HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-7B HuggingFace Base Model](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen2 Technical Report (arXiv)](https://arxiv.org/html/2407.10671v1)
- [Qwen2.5-LLM Blog Post](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Qwen2 HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/qwen2)
- [HuggingFace Transformers Qwen2 Modeling Code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)
- [Tiny LLM - The Qwen2 Model](https://skyzh.github.io/tiny-llm/week1-05-qwen2-model.html)
- [Qwen Tokenization Notes](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md)
- [Qwen Key Concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Llama 3.1 8B Instruct HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Llama 3.1 Blog (HuggingFace)](https://huggingface.co/blog/llama31)
- [Understanding Llama 3.1 Architecture](https://medium.com/@yuxiaojian/understand-how-llama3-1-works-a-deep-dive-into-the-model-flow-b149aba04bed)
- [Meta Llama 3 Introduction](https://ai.meta.com/blog/meta-llama-3/)
- [Qwen2 Transformer Architecture](https://www.emergentmind.com/topics/qwen2-transformer-architecture)
- [SwiGLU Activation in LLMs](https://medium.com/@saeed.mehrang/swiglu-the-activation-function-powering-modern-llms-70ea5cfdeafe)
- [Qwen2.5-7B VRAM Requirements](https://apxml.com/models/qwen2-5-7b)
- [GPU Requirements for Qwen Models](https://apxml.com/posts/gpu-system-requirements-qwen-models)
- [vLLM Qwen Deployment Docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
