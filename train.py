"""
Training module for Engram Transformer models.

Implements the training methodology from the Engram paper:
- Dual optimizer: separate optimizers for backbone vs Engram parameters
  - Backbone: AdamW (or Muon at scale) with weight decay
  - Engram embeddings: Adam with 5x learning rate, no weight decay
- Engram convolution parameters zero-initialized (identity mapping at start)
- Step-decay LR schedule
- Standard next-token prediction (cross-entropy) loss

Supports both:
  1. Full pretraining of RealEngramTransformer (engram_transformer.py)
  2. Fine-tuning Engram parameters on a grafted model (engram_graft.py)

Usage:
    from train import EngramTrainer, TrainConfig

    # --- Train the real transformer from scratch ---
    from engram_transformer import get_model
    model, tokenizer = get_model(force_rebuild=True)
    trainer = EngramTrainer(model, tokenizer, TrainConfig())
    trainer.train("path/to/text_files/")

    # --- Fine-tune Engram on a grafted pretrained model ---
    from engram_graft import get_model
    model, tokenizer = get_model("gpt2")
    config = TrainConfig(mode="graft", base_lr=1e-4)
    trainer = EngramTrainer(model, tokenizer, config)
    trainer.train("path/to/text_files/")
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters following the Engram paper (Table 5)."""

    # --- mode ---
    # "full"  = train all parameters (RealEngramTransformer)
    # "graft" = freeze backbone, train only Engram + adapter params (EngramGraft)
    mode: str = "full"

    # --- optimizer (paper: Muon for backbone, Adam for Engram embeddings) ---
    base_lr: float = 4e-4
    engram_lr_multiplier: float = 5.0      # Engram embedding LR = base_lr * 5
    weight_decay: float = 0.1              # backbone weight decay
    engram_weight_decay: float = 0.0       # no weight decay on Engram embeddings
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # --- schedule (paper: step decay from DeepSeek-V2) ---
    warmup_steps: int = 200
    total_steps: int = 5000
    # step decay: multiply LR by decay_factor at each milestone
    lr_decay_factor: float = 0.316         # sqrt(0.1) ≈ two decays = 10x reduction
    lr_decay_milestones: list[int] = field(
        default_factory=lambda: [4000, 4500]
    )

    # --- data ---
    seq_len: int = 512
    batch_size: int = 4
    num_workers: int = 0

    # --- logging / checkpointing ---
    log_interval: int = 50
    save_interval: int = 1000
    save_dir: str = "checkpoints/training"
    grad_clip: float = 1.0

    # --- Engram init (paper: conv params zero-initialized) ---
    zero_init_conv: bool = True


# ---------------------------------------------------------------------------
# Dataset: simple text file(s) → token chunks
# ---------------------------------------------------------------------------

class TextTokenDataset(Dataset):
    """
    Reads text files, tokenizes them into a flat token buffer, and serves
    fixed-length chunks for next-token prediction.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int = 512,
    ):
        self.seq_len = seq_len
        path = Path(path)

        # collect text from file(s)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.suffix in (".txt", ".md", ".jsonl", ".json", ".csv")
            )
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        if not files:
            raise FileNotFoundError(f"No text files found in {path}")

        print(f"Tokenizing {len(files)} file(s)...")
        all_ids: list[int] = []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="replace")
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)

        # store as a flat tensor — each sample is a contiguous slice
        self.tokens = torch.tensor(all_ids, dtype=torch.long)
        # +1 because we need (input, target) pairs shifted by 1
        self.n_samples = max(0, (len(self.tokens) - 1) // self.seq_len)
        print(
            f"  {len(self.tokens):,} tokens → {self.n_samples:,} samples "
            f"(seq_len={self.seq_len})"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]  # (input_ids, labels)


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------

def _is_engram_embedding(name: str) -> bool:
    """Check if a parameter belongs to an Engram embedding table."""
    return "multi_head_embedding" in name or "engram_modules" in name


def _is_engram_param(name: str) -> bool:
    """Check if a parameter belongs to any Engram-related module."""
    engram_keys = (
        "engram", "multi_head_embedding", "short_conv",
        "value_proj", "key_projs", "norm1", "norm2",
        "adapters",
    )
    return any(k in name for k in engram_keys)


def build_param_groups(
    model: nn.Module,
    config: TrainConfig,
) -> list[dict]:
    """
    Split parameters into groups following the paper's dual-optimizer scheme:

    1. Engram embeddings: Adam, LR = base_lr * engram_lr_multiplier, no weight decay
    2. Other Engram params (projections, norms, conv, adapters): Adam, base LR, weight decay
    3. Backbone params: AdamW, base LR, weight decay

    In graft mode, backbone params are excluded (frozen).
    """
    engram_embed_params = []
    engram_other_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_engram_embedding(name):
            engram_embed_params.append(param)
        elif _is_engram_param(name):
            engram_other_params.append(param)
        else:
            backbone_params.append(param)

    groups = []

    if engram_embed_params:
        groups.append({
            "params": engram_embed_params,
            "lr": config.base_lr * config.engram_lr_multiplier,
            "weight_decay": config.engram_weight_decay,
            "label": "engram_embed",
        })

    if engram_other_params:
        groups.append({
            "params": engram_other_params,
            "lr": config.base_lr,
            "weight_decay": config.weight_decay,
            "label": "engram_other",
        })

    if backbone_params and config.mode == "full":
        groups.append({
            "params": backbone_params,
            "lr": config.base_lr,
            "weight_decay": config.weight_decay,
            "label": "backbone",
        })

    return groups


# ---------------------------------------------------------------------------
# LR scheduler: warmup + step decay (paper §4.1, Table 5)
# ---------------------------------------------------------------------------

def _get_lr_scale(step: int, config: TrainConfig) -> float:
    """Compute LR multiplier: linear warmup then step decay at milestones."""
    # warmup
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps

    # step decay
    scale = 1.0
    for milestone in config.lr_decay_milestones:
        if step >= milestone:
            scale *= config.lr_decay_factor
    return scale


def step_scheduler(optimizer: torch.optim.Optimizer, step: int, config: TrainConfig):
    """Update learning rates in-place for all param groups."""
    scale = _get_lr_scale(step, config)
    for group in optimizer.param_groups:
        base_lr = group.get("_base_lr", group["lr"])
        if "_base_lr" not in group:
            group["_base_lr"] = group["lr"]
        group["lr"] = base_lr * scale


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EngramTrainer:
    """
    Trainer for Engram models following the paper's methodology.

    Handles:
    - Dual parameter groups with differentiated LR / weight decay
    - Engram conv zero-initialization
    - Step-decay LR schedule with warmup
    - Gradient clipping
    - Periodic logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: TrainConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainConfig()

        # apply Engram-specific initialization
        if self.config.zero_init_conv:
            self._zero_init_conv()

        # build optimizer
        param_groups = build_param_groups(model, self.config)
        self._log_param_groups(param_groups)

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=self.config.betas,
            eps=self.config.eps,
        )

        self.device = next(model.parameters()).device
        self.global_step = 0

    def _zero_init_conv(self):
        """Zero-initialize Engram ShortConv weights (paper: preserve identity at start)."""
        for name, param in self.model.named_parameters():
            if "short_conv" in name and "conv.weight" in name:
                nn.init.zeros_(param)

    def _log_param_groups(self, groups: list[dict]):
        total = 0
        for g in groups:
            n = sum(p.numel() for p in g["params"])
            total += n
            label = g.get("label", "?")
            print(f"  param group '{label}': {n:,} params, lr={g['lr']:.2e}, wd={g['weight_decay']}")
        print(f"  total trainable: {total:,}")

    # ---- core training loop ----

    def train(
        self,
        data_path: str | Path,
        resume_step: int = 0,
    ) -> dict:
        """
        Train the model on text data.

        Args:
            data_path: path to a text file or directory of text files.
            resume_step: skip this many steps (for resuming).

        Returns:
            dict with final training stats.
        """
        cfg = self.config

        dataset = TextTokenDataset(data_path, self.tokenizer, cfg.seq_len)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.model.train()
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_loss = 0.0
        tokens_seen = 0
        log_loss = 0.0
        log_tokens = 0
        t0 = time.perf_counter()

        step = 0
        data_iter = self._infinite_loader(loader)

        print(f"\nTraining for {cfg.total_steps} steps...")
        print(f"  batch_size={cfg.batch_size}  seq_len={cfg.seq_len}")
        print(f"  effective tokens/step={cfg.batch_size * cfg.seq_len:,}")
        print("-" * 60)

        for step in range(cfg.total_steps):
            if step < resume_step:
                next(data_iter)
                continue

            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # forward
            logits = self.model(input_ids)

            # cross-entropy loss (next-token prediction)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )

            # step optimizer + scheduler
            step_scheduler(self.optimizer, step, cfg)
            self.optimizer.step()

            self.global_step = step + 1
            batch_tokens = input_ids.numel()
            tokens_seen += batch_tokens
            total_loss += loss.item() * batch_tokens
            log_loss += loss.item()
            log_tokens += 1

            # logging
            if (step + 1) % cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = log_loss / log_tokens
                tok_per_sec = tokens_seen / elapsed
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step + 1:>6d}/{cfg.total_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  lr={current_lr:.2e}"
                    f"  tok/s={tok_per_sec:,.0f}"
                    f"  tokens={tokens_seen:,}"
                )
                log_loss = 0.0
                log_tokens = 0

            # checkpointing
            if (step + 1) % cfg.save_interval == 0:
                self._save_checkpoint(save_dir, step + 1)

        # final save
        self._save_checkpoint(save_dir, step + 1)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / tokens_seen if tokens_seen > 0 else float("inf")
        print(f"\nTraining complete: {step + 1} steps, {tokens_seen:,} tokens")
        print(f"  final avg loss: {avg_loss:.4f}")
        print(f"  wall time: {elapsed:.1f}s ({tokens_seen / elapsed:,.0f} tok/s)")

        return {
            "steps": step + 1,
            "tokens": tokens_seen,
            "avg_loss": avg_loss,
            "wall_time": elapsed,
        }

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        """Yield batches forever, cycling through epochs."""
        while True:
            for batch in loader:
                yield batch

    def _save_checkpoint(self, save_dir: Path, step: int):
        path = save_dir / f"step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the step number to resume from."""
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        self.global_step = step
        print(f"Resumed from {path} at step {step}")
        return step


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train an Engram model on text data")
    parser.add_argument("data", help="Path to text file or directory of text files")

    # model selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--transformer", action="store_true", default=True,
        help="Train the RealEngramTransformer from scratch (default)",
    )
    mode_group.add_argument(
        "--graft", type=str, default=None, metavar="MODEL",
        help="Fine-tune Engram on a grafted pretrained model (e.g. --graft gpt2)",
    )

    # architecture (transformer mode)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--engram-layers", nargs="+", type=int, default=[1, 4])

    # training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-dir", default="checkpoints/training")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=1000)

    # device
    parser.add_argument("--device", default=None)

    # resume
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # pick device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # build model
    if args.graft:
        from engram_graft import get_model as get_graft_model
        print(f"Loading grafted model: {args.graft}")
        model, tokenizer = get_graft_model(
            model_name_or_path=args.graft,
            engram_layer_ids=args.engram_layers,
            device=device,
        )
        mode = "graft"
    else:
        from engram_transformer import get_model as get_transformer_model
        from engram_transformer import RealBackBoneConfig
        cfg = RealBackBoneConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            ffn_hidden_size=args.ffn_size,
            engram_layer_ids=args.engram_layers,
        )
        model, tokenizer = get_transformer_model(cfg, force_rebuild=True)
        model = model.to(device)
        mode = "full"

    # build trainer
    train_config = TrainConfig(
        mode=mode,
        base_lr=args.lr,
        warmup_steps=args.warmup,
        total_steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    trainer = EngramTrainer(model, tokenizer, train_config)

    resume_step = 0
    if args.resume:
        resume_step = trainer.load_checkpoint(args.resume)

    trainer.train(args.data, resume_step=resume_step)


if __name__ == "__main__":
    main()
