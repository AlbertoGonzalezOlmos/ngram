"""
Chat with a pretrained HuggingFace model augmented with Engram modules.

This uses engram_graft.py (solution 1): a frozen pretrained backbone (e.g. GPT-2)
with Engram N-gram lookup injected at specific layers via forward hooks.
Output is coherent (from the pretrained LM) with Engram influencing hidden states.

Usage:
    uv run python chat_graft.py
    uv run python chat_graft.py --model gpt2 --engram-layers 1 5
    uv run python chat_graft.py --model meta-llama/Llama-3.2-1B-Instruct --max-tokens 128
"""

import argparse
import sys
import io
import time

import torch

from engram_graft import get_model, GraftConfig

import os
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
login(token=os.getenv("HF_TOKEN"))


def ensure_utf8_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_chat_template(tokenizer) -> bool:
    return (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    )


def format_prompt(tokenizer, message: str, history: list[dict]) -> str:
    messages = history + [{"role": "user", "content": message}]
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # fallback for models without a chat template (e.g. gpt2)
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            continue

        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove_mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return input_ids


def chat_loop(model, tokenizer, device, max_new_tokens, temperature, top_k, top_p):
    template_status = "chat template" if has_chat_template(tokenizer) else "raw prompt"
    print(f"\nEngram Graft Chat [{template_status}] (type 'quit' or Ctrl-C to exit)")
    print(f"  backbone: {model.config.model_name_or_path}")
    print(f"  engram layers: {model.config.engram_layer_ids}")
    print(f"  device={device}  max_new_tokens={max_new_tokens}  temperature={temperature}  top_k={top_k}  top_p={top_p}")
    print("-" * 60)

    history: list[dict] = []

    while True:
        try:
            message = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if message.lower() == "/clear":
            history.clear()
            print("(conversation cleared)")
            continue

        prompt = format_prompt(tokenizer, message, history)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        t0 = time.perf_counter()
        generated_ids = generate(
            model, tokenizer, input_ids,
            max_new_tokens, temperature, top_k, top_p,
        )
        elapsed = time.perf_counter() - t0
        new_count = generated_ids.shape[1] - input_ids.shape[1]

        new_tokens = generated_ids[:, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        # truncate at common stop sequences
        for stop in ["User:", "Human:", "<|User|>", "<|Human|>", "\nYou:"]:
            if stop in response:
                response = response[: response.index(stop)]

        tok_per_sec = new_count / elapsed if elapsed > 0 else 0
        print(f"\nEngram: {response}")
        print(f"  [{new_count} tokens in {elapsed:.2f}s — {tok_per_sec:.1f} tok/s]")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})


def main():
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Chat with Engram-grafted pretrained model")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name or path (default: gpt2)")
    parser.add_argument("--engram-layers", nargs="+", type=int, default=[1, 5], help="Layer IDs for Engram injection (default: 1 5)")
    parser.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max new tokens (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, 0 for greedy (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    args = parser.parse_args()

    device = args.device or pick_device()

    model, tokenizer = get_model(
        model_name_or_path=args.model,
        engram_layer_ids=args.engram_layers,
        device=device,
    )

    chat_loop(model, tokenizer, device, args.max_tokens, args.temperature, args.top_k, args.top_p)


if __name__ == "__main__":
    main()
