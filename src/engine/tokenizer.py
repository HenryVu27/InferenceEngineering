"""Qwen2.5 tokenizer — tiktoken BPE with ChatML chat template.

Tokenizer spec (Qwen2.5-7B-Instruct):
  - tiktoken BPE, vocab size 152,064
  - Chat format: ChatML (<|im_start|>, <|im_end|>)
  - Two EOS tokens: <|im_end|> (151645), <|endoftext|> (151643)
  - No BOS token (unlike Llama)

Usage:
  tokenizer = QwenTokenizer("path/to/Qwen2.5-7B-Instruct")
  tokens = tokenizer.encode_chat([{"role": "user", "content": "Hello!"}])
  text = tokenizer.decode(tokens)
"""

from pathlib import Path

import tiktoken


# Special token IDs for Qwen2.5
IM_START_ID = 151644   # <|im_start|>
IM_END_ID = 151645     # <|im_end|>
ENDOFTEXT_ID = 151643  # <|endoftext|>
EOS_IDS = {IM_END_ID, ENDOFTEXT_ID}


class QwenTokenizer:
    """Minimal Qwen2.5 tokenizer wrapping tiktoken."""

    def __init__(self, model_dir: str | Path):
        """Load tokenizer from model directory.

        Args:
            model_dir: Path to Qwen2.5-7B-Instruct directory containing tokenizer files.
        """
        # TODO: Load the tiktoken tokenizer from the model directory.
        # Qwen2.5 uses a tiktoken-compatible BPE vocabulary.
        # Look for the vocab file and special tokens config.
        # Hint: check tokenizer_config.json for special token mappings,
        #       and load the mergeable ranks from the vocab file.
        model_dir = Path(model_dir)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Raw text string.

        Returns:
            List of integer token IDs.
        """
        # TODO: Encode text using tiktoken.
        # Remember: Qwen does NOT prepend a BOS token.
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        # TODO: Decode token IDs using tiktoken.
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply ChatML template and encode.

        ChatML format:
            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            Hello!<|im_end|>
            <|im_start|>assistant


        Args:
            messages: List of {"role": str, "content": str} dicts.

        Returns:
            Token IDs with ChatML special tokens applied.
        """
        # TODO: Build the ChatML string from messages, then encode.
        # Each message: <|im_start|>{role}\n{content}<|im_end|>\n
        # After last message, append <|im_start|>assistant\n to prompt generation.
        # Encode special tokens as their IDs directly (don't try to encode the text form).
        tokens = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            tokens.append(IM_START_ID)
            tokens.extend(self.encode(role + "\n" + content))
            tokens.append(IM_END_ID)
            tokens.extend(self.encode("\n"))
        # prompt model to generate as assistant
        tokens.append(IM_START_ID)
        tokens.extend(self.encode("assistant\n"))
        return tokens

    def is_eos(self, token_id: int) -> bool:
        """Check if token is an end-of-sequence token.

        Qwen2.5 has TWO EOS tokens: <|im_end|> (151645) and <|endoftext|> (151643).
        """
        return token_id in EOS_IDS
