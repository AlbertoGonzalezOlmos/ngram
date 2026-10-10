"""
Interactive chat interface using the Engram Transformer architecture.

Usage:
    uv run python chat.py
    uv run python chat.py --checkpoint checkpoints/engram_model
    uv run python chat.py --max-tokens 64 --temperature 0.7

NOTE: The demo model uses mocked attention/MoE layers (identity functions)
and random weights, so generated text will NOT be coherent. This exercises
the full Engram N-gram lookup + gated fusion pipeline end-to-end.
"""

import argparse
import sys
import io

import torch

from model_manager import get_model


def ensure_utf8_stdout():
    """Reconfigure stdout/stderr to UTF-8 to handle multilingual output."""
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
    """Autoregressive generation using the EngramTransformer."""
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        # take logits for the last position
        next_logits = logits[:, -1, :]

        # temperature scaling
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # greedy
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            continue

        # top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        # top-p (nucleus) filtering
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
    """Interactive read-eval-print loop."""
    print(f"\nEngram Chat (type 'quit' or Ctrl-C to exit)")
    print(f"  device={device}  max_new_tokens={max_new_tokens}  temperature={temperature}  top_k={top_k}  top_p={top_p}")
    print(
        "  NOTE: attention/MoE layers are mocked — output will not be coherent"
    )
    print("-" * 60)

    history_text = ""

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
            history_text = ""
            print("(conversation cleared)")
            continue

        # build prompt with history
        prompt = history_text + "User: " + message + "\nAssistant:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = generate(
            model, tokenizer, input_ids,
            max_new_tokens, temperature, top_k, top_p,
        )

        # decode only the newly generated tokens
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        print(f"\nEngram: {response}")

        # accumulate history
        history_text = prompt + " " + response + "\n"


def main():
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Chat with the Engram Transformer")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory (default: checkpoints/engram_model)")
    parser.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, 0 for greedy (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    args = parser.parse_args()

    device = args.device or pick_device()

    kwargs = {}
    if args.checkpoint:
        kwargs["checkpoint_dir"] = args.checkpoint

    model, tokenizer = get_model(**kwargs)
    model = model.to(device)

    chat_loop(model, tokenizer, device, args.max_tokens, args.temperature, args.top_k, args.top_p)


if __name__ == "__main__":
    main()
