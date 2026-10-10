"""
Supervised fine-tuning module for question-answering with Engram models.

Trains the model to predict an answer given an input question. The loss is
computed only on the answer tokens (the question serves as context but does
not contribute to the loss), following standard instruction-tuning practice.

Supports two input modalities:
  - Text questions:  {"question": "...", "answer": "..."}  (JSONL / JSON / CSV)
  - Audio questions: {"audio": "path/to/clip.wav", "answer": "..."}  (JSONL)

In audio mode, the question audio is encoded by a frozen Whisper encoder and
prepended as a prefix to the answer tokens (prefix-LM). Loss is masked on
everything before the answer, so only answer tokens contribute to training.

Works with both model types:
  - RealEngramTransformer (full training)
  - EngramGraft (fine-tune Engram params only, backbone frozen)

Usage:
    # Text QA
    uv run python train_qa.py data.jsonl --graft gpt2 --steps 500

    # Audio QA
    uv run python train_qa.py manifest.jsonl --audio --graft gpt2 --steps 500

    # Python API
    from train_qa import QATrainer, QATrainConfig, AudioQATrainer, AudioQAConfig

    trainer = QATrainer(model, tokenizer, QATrainConfig())
    trainer.train("qa_data.jsonl")
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from train import (
    build_param_groups,
    step_scheduler,
    TrainConfig,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100  # PyTorch cross_entropy ignores this label


@dataclass
class QATrainConfig(TrainConfig):
    """Training config with QA-specific defaults."""

    # override defaults for SFT (smaller LR, fewer steps)
    base_lr: float = 2e-4
    total_steps: int = 1000
    warmup_steps: int = 50
    lr_decay_milestones: list[int] = field(default_factory=lambda: [800, 900])
    save_dir: str = "checkpoints/qa_training"
    seq_len: int = 256

    # QA prompt template — {question} and {answer} are replaced at runtime
    prompt_template: str = "Question: {question}\nAnswer: {answer}"
    # separator between prompt context and answer (loss only on tokens after this)
    answer_prefix: str = "\nAnswer: "


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_qa_pairs(path: Path) -> list[dict]:
    """Load QA pairs from JSONL, JSON, or CSV."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        pairs = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
        return pairs

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON file must contain a list of objects, got {type(data)}")

    if suffix == ".csv":
        pairs = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append({"question": row["question"], "answer": row["answer"]})
        return pairs

    raise ValueError(f"Unsupported file type: {suffix} (expected .jsonl, .json, or .csv)")


class QADataset(Dataset):
    """
    Tokenizes question-answer pairs into (input_ids, labels) where the loss
    is masked on the question tokens (only answer tokens contribute).

    Each sample is formatted as:
        "Question: {question}\nAnswer: {answer}<eos>"

    Labels are set to IGNORE_INDEX for all tokens up to and including the
    answer prefix, so the model learns to generate only the answer.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        config: QATrainConfig,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.seq_len = config.seq_len

        path = Path(path)
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.suffix in (".jsonl", ".json", ".csv")
            )
            raw_pairs: list[dict] = []
            for fp in files:
                raw_pairs.extend(_load_qa_pairs(fp))
        else:
            raw_pairs = _load_qa_pairs(path)

        if not raw_pairs:
            raise ValueError(f"No QA pairs found in {path}")

        # pre-tokenize the answer prefix to know where the answer starts
        self.answer_prefix_ids = tokenizer.encode(
            config.answer_prefix, add_special_tokens=False
        )

        # tokenize all pairs
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        skipped = 0
        for pair in raw_pairs:
            result = self._tokenize_pair(pair)
            if result is not None:
                self.samples.append(result)
            else:
                skipped += 1

        print(
            f"  {len(self.samples)} QA samples loaded"
            f" ({skipped} skipped — exceeded seq_len={self.seq_len})"
        )

    def _tokenize_pair(
        self, pair: dict
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tokenize a single QA pair into (input_ids, labels) of length seq_len."""
        q = pair.get("question", pair.get("q", ""))
        a = pair.get("answer", pair.get("a", ""))

        text = self.config.prompt_template.format(question=q, answer=a)

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # append EOS so the model learns to stop
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.append(eos)

        if len(ids) < 2:
            return None

        # find where the answer starts by locating the answer_prefix token span
        answer_start = self._find_answer_start(ids)

        # truncate or pad to seq_len + 1 (need shifted input/label)
        total_len = self.seq_len + 1
        if len(ids) > total_len:
            # skip if the question alone is already too long to leave room for answer
            if answer_start >= self.seq_len:
                return None
            ids = ids[:total_len]
        elif len(ids) < total_len:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = eos if eos is not None else 0
            ids = ids + [pad_id] * (total_len - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)

        # mask question tokens in labels (only train on answer)
        labels = target_ids.clone()
        # everything before answer_start (shifted by 1 for labels) gets masked
        mask_end = max(0, answer_start - 1)
        labels[:mask_end] = IGNORE_INDEX

        # also mask padding tokens
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[target_ids == pad_id] = IGNORE_INDEX

        return input_ids, labels

    def _find_answer_start(self, ids: list[int]) -> int:
        """Find the token index where the answer begins (after the prefix)."""
        prefix = self.answer_prefix_ids
        prefix_len = len(prefix)
        for i in range(len(ids) - prefix_len + 1):
            if ids[i : i + prefix_len] == prefix:
                return i + prefix_len
        # fallback: assume the whole sequence is answer
        return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class QATrainer:
    """
    Supervised fine-tuning trainer for question-answering.

    Like EngramTrainer but uses QADataset with masked loss (only answer tokens
    contribute to the gradient). Reuses the same dual-optimizer / LR schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: QATrainConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or QATrainConfig()

        # zero-init conv if requested
        if self.config.zero_init_conv:
            self._zero_init_conv()

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

    def train(
        self,
        data_path: str | Path,
        resume_step: int = 0,
    ) -> dict:
        """
        Fine-tune on QA data.

        Args:
            data_path: path to a .jsonl, .json, or .csv file (or directory).
            resume_step: skip this many steps when resuming.

        Returns:
            dict with final training stats.
        """
        cfg = self.config

        print(f"Loading QA data from {data_path}...")
        dataset = QADataset(data_path, self.tokenizer, cfg)
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
        total_answer_tokens = 0
        log_loss = 0.0
        log_steps = 0
        t0 = time.perf_counter()

        step = 0
        data_iter = self._infinite_loader(loader)

        print(f"\nQA fine-tuning for {cfg.total_steps} steps...")
        print(f"  batch_size={cfg.batch_size}  seq_len={cfg.seq_len}  samples={len(dataset)}")
        print(f"  loss is masked — only answer tokens contribute")
        print("-" * 60)

        for step in range(cfg.total_steps):
            if step < resume_step:
                next(data_iter)
                continue

            input_ids, labels = next(data_iter)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )

            step_scheduler(self.optimizer, step, cfg)
            self.optimizer.step()

            self.global_step = step + 1

            # count answer tokens (non-ignored)
            n_answer = (labels != IGNORE_INDEX).sum().item()
            total_answer_tokens += n_answer
            total_loss += loss.item() * n_answer
            log_loss += loss.item()
            log_steps += 1

            if (step + 1) % cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = log_loss / log_steps
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step + 1:>6d}/{cfg.total_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  lr={current_lr:.2e}"
                    f"  answer_tokens={total_answer_tokens:,}"
                    f"  {elapsed:.1f}s"
                )
                log_loss = 0.0
                log_steps = 0

            if (step + 1) % cfg.save_interval == 0:
                self._save_checkpoint(save_dir, step + 1)

        self._save_checkpoint(save_dir, step + 1)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / total_answer_tokens if total_answer_tokens > 0 else float("inf")
        print(f"\nQA training complete: {step + 1} steps")
        print(f"  final avg loss (answer tokens): {avg_loss:.4f}")
        print(f"  total answer tokens trained on: {total_answer_tokens:,}")
        print(f"  wall time: {elapsed:.1f}s")

        return {
            "steps": step + 1,
            "answer_tokens": total_answer_tokens,
            "avg_loss": avg_loss,
            "wall_time": elapsed,
        }

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        while True:
            for batch in loader:
                yield batch

    def _save_checkpoint(self, save_dir: Path, step: int):
        path = save_dir / f"qa_step_{step}.pt"
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
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        self.global_step = step
        print(f"Resumed from {path} at step {step}")
        return step

    @torch.no_grad()
    def evaluate(
        self,
        data_path: str | Path,
        max_samples: int = 0,
    ) -> dict:
        """
        Compute average loss on a held-out QA dataset.

        Args:
            data_path: path to evaluation QA data.
            max_samples: limit evaluation to N samples (0 = all).

        Returns:
            dict with eval loss and token counts.
        """
        dataset = QADataset(data_path, self.tokenizer, self.config)
        if max_samples > 0:
            indices = list(range(min(max_samples, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)

        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
            n = (labels != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

        self.model.train()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        return {"eval_loss": avg_loss, "eval_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Audio QA — Config, Dataset, Trainer
# ---------------------------------------------------------------------------

@dataclass
class AudioQAConfig(QATrainConfig):
    """Training config for audio-question → text-answer QA."""

    save_dir: str = "checkpoints/audio_qa_training"
    batch_size: int = 2

    # audio settings
    whisper_model: str = "openai/whisper-tiny"
    freeze_encoder: bool = True
    max_audio_seconds: float = 30.0

    # answer prefix is prepended before the answer text
    answer_prefix: str = "Answer: "


class AudioQADataset(Dataset):
    """
    Loads audio-question + text-answer pairs from a JSONL manifest.

    Each line: {"audio": "path/to/question.wav", "answer": "the answer text"}

    The audio is the question; the model learns to generate only the answer.
    Audio is encoded at training time via WhisperFeatureExtractor.
    Labels are the answer tokens with an "Answer: " prefix; loss is masked on
    audio positions (handled by the trainer, not the dataset).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor,
        config: AudioQAConfig,
    ):
        from transformers import WhisperFeatureExtractor as WFE
        from train_audio import load_audio, WHISPER_SAMPLE_RATE

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.config = config
        self.seq_len = config.seq_len
        self.max_audio_samples = int(config.max_audio_seconds * WHISPER_SAMPLE_RATE)
        self._load_audio = load_audio
        self._sample_rate = WHISPER_SAMPLE_RATE

        manifest_path = Path(manifest_path)
        self.base_dir = manifest_path.parent

        # load manifest
        self.samples: list[dict] = []
        if manifest_path.is_dir():
            files = sorted(manifest_path.rglob("*.jsonl"))
        else:
            files = [manifest_path]

        for fp in files:
            for line in fp.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    audio_path = Path(entry["audio"])
                    if not audio_path.is_absolute():
                        audio_path = self.base_dir / audio_path
                    if audio_path.exists():
                        self.samples.append({
                            "audio": str(audio_path),
                            "answer": entry["answer"],
                        })

        print(f"  {len(self.samples)} audio-QA pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # load and extract audio features
        audio = self._load_audio(sample["audio"])
        audio = audio[: self.max_audio_samples]

        features = self.feature_extractor(
            audio,
            sampling_rate=self._sample_rate,
            return_tensors="np",
        )
        audio_features = features.input_features[0]  # (n_mels, T)

        # tokenize answer with prefix: "Answer: {answer}<eos>"
        answer_text = self.config.answer_prefix + sample["answer"]
        text_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            text_ids.append(eos)

        # also tokenize just the prefix to know where the answer starts
        prefix_ids = self.tokenizer.encode(
            self.config.answer_prefix, add_special_tokens=False
        )
        answer_start = len(prefix_ids)

        # truncate / pad to seq_len + 1
        total_len = self.seq_len + 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos if eos is not None else 0

        if len(text_ids) > total_len:
            text_ids = text_ids[:total_len]
        elif len(text_ids) < total_len:
            text_ids = text_ids + [pad_id] * (total_len - len(text_ids))

        input_ids = torch.tensor(text_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(text_ids[1:], dtype=torch.long)

        # mask: everything before the answer content gets IGNORE_INDEX
        labels = target_ids.clone()
        mask_end = max(0, answer_start - 1)
        labels[:mask_end] = IGNORE_INDEX

        # mask padding
        if pad_id is not None:
            labels[target_ids == pad_id] = IGNORE_INDEX

        return {
            "audio_features": torch.tensor(audio_features, dtype=torch.float32),
            "input_ids": input_ids,
            "labels": labels,
        }


def _audio_qa_collate(batch: list[dict]) -> dict:
    return {
        "audio_features": torch.stack([s["audio_features"] for s in batch]),
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
    }


class AudioQATrainer:
    """
    Trainer for audio-question → text-answer QA.

    Wraps the decoder in AudioEngramModel (from train_audio), then trains
    with masked cross-entropy loss on answer tokens only.
    """

    def __init__(
        self,
        audio_model: nn.Module,  # AudioEngramModel
        tokenizer: PreTrainedTokenizerBase,
        config: AudioQAConfig | None = None,
    ):
        self.model = audio_model
        self.tokenizer = tokenizer
        self.config = config or AudioQAConfig()

        if self.config.zero_init_conv:
            self._zero_init_conv()

        param_groups = build_param_groups(audio_model, self.config)
        self._log_param_groups(param_groups)

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=self.config.betas,
            eps=self.config.eps,
        )

        self.device = next(audio_model.parameters()).device
        self.global_step = 0

    def _zero_init_conv(self):
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

    def train(
        self,
        data_path: str | Path,
        resume_step: int = 0,
    ) -> dict:
        cfg = self.config

        print(f"Loading audio QA data from {data_path}...")
        dataset = AudioQADataset(
            data_path, self.tokenizer,
            self.model.feature_extractor, cfg,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_audio_qa_collate,
            pin_memory=True,
            drop_last=True,
        )

        self.model.train()
        if cfg.freeze_encoder:
            self.model.encoder.eval()

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_loss = 0.0
        total_answer_tokens = 0
        log_loss = 0.0
        log_steps = 0
        t0 = time.perf_counter()

        step = 0
        data_iter = self._infinite_loader(loader)

        print(f"\nAudio QA training for {cfg.total_steps} steps...")
        print(f"  batch_size={cfg.batch_size}  seq_len={cfg.seq_len}")
        print(f"  whisper encoder: {cfg.whisper_model} (frozen={cfg.freeze_encoder})")
        print(f"  loss is masked — only answer tokens contribute")
        print("-" * 60)

        for step in range(cfg.total_steps):
            if step < resume_step:
                next(data_iter)
                continue

            batch = next(data_iter)
            audio_features = batch["audio_features"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # forward: audio prefix + text → logits for text positions
            logits = self.model(audio_features, input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )

            step_scheduler(self.optimizer, step, cfg)
            self.optimizer.step()

            self.global_step = step + 1
            n_answer = (labels != IGNORE_INDEX).sum().item()
            total_answer_tokens += n_answer
            total_loss += loss.item() * n_answer
            log_loss += loss.item()
            log_steps += 1

            if (step + 1) % cfg.log_interval == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = log_loss / log_steps
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step + 1:>6d}/{cfg.total_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  lr={current_lr:.2e}"
                    f"  answer_tokens={total_answer_tokens:,}"
                    f"  {elapsed:.1f}s"
                )
                log_loss = 0.0
                log_steps = 0

            if (step + 1) % cfg.save_interval == 0:
                self._save_checkpoint(save_dir, step + 1)

        self._save_checkpoint(save_dir, step + 1)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / total_answer_tokens if total_answer_tokens > 0 else float("inf")
        print(f"\nAudio QA training complete: {step + 1} steps")
        print(f"  final avg loss (answer tokens): {avg_loss:.4f}")
        print(f"  total answer tokens: {total_answer_tokens:,}")
        print(f"  wall time: {elapsed:.1f}s")

        return {
            "steps": step + 1,
            "answer_tokens": total_answer_tokens,
            "avg_loss": avg_loss,
            "wall_time": elapsed,
        }

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        while True:
            for batch in loader:
                yield batch

    def _save_checkpoint(self, save_dir: Path, step: int):
        path = save_dir / f"audio_qa_step_{step}.pt"
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
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        self.global_step = step
        print(f"Resumed from {path} at step {step}")
        return step

    @torch.no_grad()
    def evaluate(
        self,
        data_path: str | Path,
        max_samples: int = 0,
    ) -> dict:
        dataset = AudioQADataset(
            data_path, self.tokenizer,
            self.model.feature_extractor, self.config,
        )
        if max_samples > 0:
            indices = list(range(min(max_samples, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)

        loader = DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=_audio_qa_collate,
        )

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in loader:
            audio_features = batch["audio_features"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(audio_features, input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            n = (labels != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * n
            total_tokens += n

        self.model.train()
        if self.config.freeze_encoder:
            self.model.encoder.eval()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        return {"eval_loss": avg_loss, "eval_tokens": total_tokens}

    @torch.no_grad()
    def answer(
        self,
        audio_path: str | Path,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """
        Answer a question from an audio file.

        Args:
            audio_path: path to .wav/.mp3/.flac file containing the question
            max_tokens: maximum answer tokens to generate
            temperature: sampling temperature (0 = greedy)

        Returns:
            predicted answer text
        """
        from train_audio import load_audio, WHISPER_SAMPLE_RATE

        self.model.eval()
        device = next(self.model.parameters()).device

        # load and encode audio question
        audio = load_audio(audio_path)
        max_samples = int(self.config.max_audio_seconds * WHISPER_SAMPLE_RATE)
        audio = audio[:max_samples]

        features = self.model.feature_extractor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        )
        audio_features = features.input_features.to(device)

        audio_embeds = self.model.encode_audio(audio_features)
        T_audio = audio_embeds.shape[1]

        # start generation with the answer prefix
        prefix_ids = self.model.tokenizer.encode(
            self.config.answer_prefix, add_special_tokens=False
        )
        generated = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            text_embeds = self.model._get_text_embeddings(generated)
            combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            B = 1
            audio_pad_ids = torch.full(
                (B, T_audio), self.model.pad_id, dtype=torch.long, device=device
            )
            combined_ids = torch.cat([audio_pad_ids, generated], dim=1)

            logits = self.model._forward_with_embeddings(combined_embeds, combined_ids)
            next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.model.tokenizer.eos_token_id:
                break

        # decode only the answer portion (skip the prefix)
        output_ids = generated[0, len(prefix_ids):]
        return self.model.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Engram on QA data")
    parser.add_argument("data", help="Path to QA data (.jsonl, .json, .csv) or audio manifest (.jsonl)")

    # modality
    parser.add_argument("--audio", action="store_true",
                        help="Audio-question mode: data is JSONL with {audio, answer} entries")

    # model
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transformer", action="store_true", default=True)
    mode_group.add_argument("--graft", type=str, default=None, metavar="MODEL")

    # architecture
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=1024)
    parser.add_argument("--engram-layers", nargs="+", type=int, default=[1, 4])

    # audio settings
    parser.add_argument("--whisper-model", default="openai/whisper-tiny",
                        help="Whisper encoder model (default: openai/whisper-tiny)")
    parser.add_argument("--max-audio-sec", type=float, default=30.0)

    # training
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)

    # eval / inference
    parser.add_argument("--eval-data", default=None, help="Eval data (optional)")
    parser.add_argument("--answer-audio", type=str, default=None,
                        help="Answer a question from an audio file (inference, audio mode only)")

    # device / resume
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # build decoder
    if args.graft:
        from engram_graft import get_model as get_graft_model
        print(f"Loading grafted model: {args.graft}")
        decoder, tokenizer = get_graft_model(
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
        decoder, tokenizer = get_transformer_model(cfg, force_rebuild=True)
        decoder = decoder.to(device)
        mode = "full"

    save_dir = args.save_dir or ("checkpoints/audio_qa_training" if args.audio else "checkpoints/qa_training")

    if args.audio:
        # --- Audio QA mode ---
        from train_audio import AudioEngramModel, AudioTrainConfig

        audio_qa_cfg = AudioQAConfig(
            mode=mode,
            whisper_model=args.whisper_model,
            max_audio_seconds=args.max_audio_sec,
            base_lr=args.lr,
            warmup_steps=args.warmup,
            total_steps=args.steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size if args.batch_size != 4 else 2,
            grad_clip=args.grad_clip,
            save_dir=save_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )

        # wrap decoder in AudioEngramModel
        # reuse AudioTrainConfig for Whisper settings
        audio_cfg = AudioTrainConfig(
            whisper_model=args.whisper_model,
            max_audio_seconds=args.max_audio_sec,
            freeze_encoder=audio_qa_cfg.freeze_encoder,
        )
        amodel = AudioEngramModel(decoder, tokenizer, audio_cfg).to(device)

        # inference mode
        if args.answer_audio:
            if args.resume:
                ckpt = torch.load(args.resume, weights_only=False, map_location=device)
                amodel.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded checkpoint: {args.resume}")
            trainer = AudioQATrainer(amodel, tokenizer, audio_qa_cfg)
            answer = trainer.answer(args.answer_audio)
            print(f"\nAnswer: {answer}")
            return

        # training mode
        trainer = AudioQATrainer(amodel, tokenizer, audio_qa_cfg)

        resume_step = 0
        if args.resume:
            resume_step = trainer.load_checkpoint(args.resume)

        trainer.train(args.data, resume_step=resume_step)

        if args.eval_data:
            print("\nRunning evaluation...")
            result = trainer.evaluate(args.eval_data)
            print(f"  eval loss: {result['eval_loss']:.4f} ({result['eval_tokens']:,} tokens)")

    else:
        # --- Text QA mode ---
        config = QATrainConfig(
            mode=mode,
            base_lr=args.lr,
            warmup_steps=args.warmup,
            total_steps=args.steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            save_dir=save_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
        )

        trainer = QATrainer(decoder, tokenizer, config)

        resume_step = 0
        if args.resume:
            resume_step = trainer.load_checkpoint(args.resume)

        trainer.train(args.data, resume_step=resume_step)

        if args.eval_data:
            print("\nRunning evaluation...")
            result = trainer.evaluate(args.eval_data)
            print(f"  eval loss: {result['eval_loss']:.4f} ({result['eval_tokens']:,} tokens)")


if __name__ == "__main__":
    main()
