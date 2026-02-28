"""Quick-run script for testing your inference engine.

Usage:
  python run.py --model-dir /path/to/Qwen2.5-7B-Instruct --prompt "Hello!"
  python run.py --model-dir /path/to/Qwen2.5-7B-Instruct --interactive
"""

import argparse
import time

import torch

from src.engine.tokenizer import QwenTokenizer
from src.engine.model import load_weights, generate
from src.engine.sampler import sample


def main():
    parser = argparse.ArgumentParser(description="Inference Engine — Phase 1")
    parser.add_argument("--model-dir", required=True, help="Path to Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) filtering")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                        help="Weight precision")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = QwenTokenizer(args.model_dir)

    print(f"Loading weights ({args.dtype}) to {args.device}...")
    t0 = time.perf_counter()
    weights = load_weights(args.model_dir, device=args.device, dtype=dtype)
    print(f"Weights loaded in {time.perf_counter() - t0:.1f}s")

    # Build a sampling function with the user's settings
    def sample_fn(logits: torch.Tensor) -> int:
        if args.temperature == 0 or (args.top_k == 0 and args.top_p == 1.0):
            from src.engine.sampler import greedy
            return greedy(logits)
        return sample(
            logits,
            temperature=args.temperature,
            top_k_val=args.top_k,
            top_p_val=args.top_p,
        )

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        tokens = tokenizer.encode_chat(messages)
        print(f"Prompt: {len(tokens)} tokens")

        t0 = time.perf_counter()
        output_tokens = generate(tokens, weights, args.max_tokens, sample_fn, args.device)
        elapsed = time.perf_counter() - t0

        new_tokens = output_tokens[len(tokens):]
        text = tokenizer.decode(new_tokens)
        tps = len(new_tokens) / elapsed if elapsed > 0 else 0

        print(f"\n{text}")
        print(f"\n--- {len(new_tokens)} tokens in {elapsed:.2f}s ({tps:.1f} tok/s) ---")

    elif args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit"):
                break

            messages = [{"role": "user", "content": user_input}]
            tokens = tokenizer.encode_chat(messages)

            t0 = time.perf_counter()
            output_tokens = generate(tokens, weights, args.max_tokens, sample_fn, args.device)
            elapsed = time.perf_counter() - t0

            new_tokens = output_tokens[len(tokens):]
            text = tokenizer.decode(new_tokens)
            tps = len(new_tokens) / elapsed if elapsed > 0 else 0

            print(f"Assistant: {text}")
            print(f"  [{len(new_tokens)} tokens, {elapsed:.2f}s, {tps:.1f} tok/s]\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
