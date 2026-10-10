"""
Chat with the real Engram Transformer (real attention + SwiGLU FFN + Engram).

This uses engram_transformer.py (solution 2): a from-scratch transformer with
real causal attention, SwiGLU FFN, and Engram modules at configured layers.
Weights are randomly initialized, so output will NOT be coherent until trained.

Usage:
    uv run python chat_transformer.py
    uv run python chat_transformer.py --num-layers 6 --hidden-size 256
    uv run python chat_transformer.py --max-tokens 128 --temperature 0.5
"""

import argparse
import sys
import io
import time

import torch

from engram_transformer import get_model, RealBackBoneConfig


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
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # truncate to max_seq_len to stay within position embedding range
        ctx = input_ids[:, -max_seq_len:]
        logits = model(ctx)
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


def format_params(config: RealBackBoneConfig) -> str:
    total = 0
    # rough estimate: embeddings + layers + head
    D, V, L = config.hidden_size, config.vocab_size, config.num_layers
    FFN = config.ffn_hidden_size
    total += V * D                       # token embed
    total += config.max_seq_len * D      # position embed
    total += L * (3 * D * D + D * D)     # attn QKV + out per layer
    total += L * (D * FFN * 2 + FFN * D) # SwiGLU FFN per layer
    total += V * D                       # LM head
    if total > 1e9:
        return f"{total / 1e9:.1f}B"
    if total > 1e6:
        return f"{total / 1e6:.0f}M"
    return f"{total / 1e3:.0f}K"


def chat_loop(model, tokenizer, device, max_new_tokens, temperature, top_k, top_p):
    cfg = model.config
    print(f"\nEngram Real Transformer Chat (type 'quit' or Ctrl-C to exit)")
    print(f"  architecture: {cfg.num_layers} layers, {cfg.hidden_size}d, {cfg.num_attention_heads} heads, hc_mult={cfg.hc_mult}")
    print(f"  engram layers: {cfg.engram_layer_ids}")
    print(f"  params: ~{format_params(cfg)}  (untrained random weights)")
    print(f"  device={device}  max_new_tokens={max_new_tokens}  temperature={temperature}  top_k={top_k}  top_p={top_p}")
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

        prompt = history_text + "User: " + message + "\nAssistant:"
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

        tok_per_sec = new_count / elapsed if elapsed > 0 else 0
        print(f"\nEngram: {response}")
        print(f"  [{new_count} tokens in {elapsed:.2f}s — {tok_per_sec:.1f} tok/s]")

        history_text = prompt + " " + response + "\n"


def main():
    ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Chat with the real Engram Transformer")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden dim (default: 256)")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers (default: 6)")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--ffn-size", type=int, default=1024, help="FFN hidden size (default: 1024)")
    parser.add_argument("--hc-mult", type=int, default=4, help="Hyper-connection multiplier (default: 4)")
    parser.add_argument("--engram-layers", nargs="+", type=int, default=[1, 4], help="Engram layer IDs (default: 1 4)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir (default: auto)")
    parser.add_argument("--device", default=None, help="Device: cuda, mps, or cpu (auto-detected)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature, 0 for greedy (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore saved checkpoint, rebuild from scratch")
    args = parser.parse_args()

    device = args.device or pick_device()

    cfg = RealBackBoneConfig(
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        ffn_hidden_size=args.ffn_size,
        max_seq_len=args.max_seq_len,
        engram_layer_ids=args.engram_layers,
    )

    kwargs = {"config": cfg, "force_rebuild": args.force_rebuild}
    if args.checkpoint:
        kwargs["checkpoint_dir"] = args.checkpoint

    model, tokenizer = get_model(**kwargs)
    model = model.to(device)

    chat_loop(model, tokenizer, device, args.max_tokens, args.temperature, args.top_k, args.top_p)


if __name__ == "__main__":
    main()
