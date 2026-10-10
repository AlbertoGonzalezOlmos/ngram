"""
Audio-to-text training and inference module for Engram Transformer models.

Uses a frozen Whisper encoder to extract audio features, projects them into
the Engram decoder's hidden space, and trains the decoder to generate text
transcriptions conditioned on audio (prefix-LM approach).

Architecture:
    Audio waveform
    -> Whisper encoder -> (B, T_audio, D_whisper)
    -> Linear projection -> (B, T_audio, D_model)
    -> Prepend as prefix to text token embeddings
    -> Engram Transformer decoder (causal attention covers audio prefix)
    -> Predict text tokens (loss only on text positions)

For Engram N-gram hashing, audio positions use pad tokens (no meaningful
N-gram patterns in the audio prefix).

Data format (JSONL):
    {"audio": "path/to/clip.wav", "text": "the transcription"}

Usage:
    # CLI — fine-tune grafted GPT-2 for speech-to-text
    uv run python train_audio.py manifest.jsonl --graft gpt2 --steps 500

    # CLI — train real Engram transformer
    uv run python train_audio.py manifest.jsonl --steps 2000

    # Python API
    from train_audio import AudioEngramModel, AudioTrainer, AudioTrainConfig

    amodel = AudioEngramModel(decoder_model, tokenizer)
    trainer = AudioTrainer(amodel, tokenizer, AudioTrainConfig())
    trainer.train("manifest.jsonl")

    # Inference
    text = amodel.transcribe("audio.wav")
"""

from __future__ import annotations

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
from transformers import (
    PreTrainedTokenizerBase,
    WhisperModel,
    WhisperFeatureExtractor,
)

from train import build_param_groups, step_scheduler, TrainConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100
WHISPER_SAMPLE_RATE = 16_000


@dataclass
class AudioTrainConfig(TrainConfig):
    """Training config for audio-to-text."""

    base_lr: float = 2e-4
    total_steps: int = 1000
    warmup_steps: int = 50
    lr_decay_milestones: list[int] = field(default_factory=lambda: [800, 900])
    save_dir: str = "checkpoints/audio_training"
    seq_len: int = 256          # max text tokens (audio length is separate)
    batch_size: int = 2

    # audio
    whisper_model: str = "openai/whisper-tiny"  # tiny/base/small/medium/large
    freeze_encoder: bool = True
    max_audio_seconds: float = 30.0

    # prefix token for decoder
    transcript_prefix: str = " "  # single space — minimal prompt before transcript


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and resample to target_sr. Returns float32 numpy array."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy()
    except Exception:
        # fallback to librosa
        import librosa
        audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return audio


# ---------------------------------------------------------------------------
# Audio-conditioned Engram model wrapper
# ---------------------------------------------------------------------------

class AudioEngramModel(nn.Module):
    """
    Wraps an Engram decoder (RealEngramTransformer or EngramGraft) with a
    Whisper audio encoder for speech-to-text.

    The Whisper encoder is frozen. Audio features are projected to the
    decoder's hidden size and prepended as a prefix sequence. The decoder
    then generates text tokens via causal attention that can attend to
    the audio prefix.
    """

    def __init__(
        self,
        decoder: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: AudioTrainConfig | None = None,
    ):
        super().__init__()
        self.config = config or AudioTrainConfig()
        self.decoder = decoder
        self.tokenizer = tokenizer

        # load Whisper encoder + feature extractor
        print(f"Loading Whisper encoder: {self.config.whisper_model}")
        whisper = WhisperModel.from_pretrained(self.config.whisper_model)
        self.encoder = whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model
        )
        self.encoder_hidden_size = whisper.config.d_model

        if self.config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # projection: Whisper dim -> decoder dim
        decoder_hidden_size = self._get_decoder_hidden_size()
        self.audio_proj = nn.Linear(self.encoder_hidden_size, decoder_hidden_size)

        # learnable audio position offset (small, helps the model distinguish
        # audio prefix from text positions)
        self.audio_pos_scale = nn.Parameter(torch.ones(1))

        # pad token id for Engram hashing on audio positions
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id or 0

    def _get_decoder_hidden_size(self) -> int:
        """Extract hidden size from whichever decoder type we have."""
        if hasattr(self.decoder, "config"):
            cfg = self.decoder.config
            if hasattr(cfg, "hidden_size"):
                return cfg.hidden_size
            if hasattr(cfg, "model_name_or_path"):
                return self.decoder.backbone.config.hidden_size
        # fallback: probe the embedding layer
        for name, module in self.decoder.named_modules():
            if isinstance(module, nn.Embedding):
                return module.embedding_dim
        raise ValueError("Cannot determine decoder hidden size")

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio mel features through Whisper encoder + projection.

        Args:
            audio_features: (B, n_mels, T_mel) mel spectrogram from WhisperFeatureExtractor
        Returns:
            (B, T_enc, D_decoder) projected audio features
        """
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            encoder_out = self.encoder(audio_features).last_hidden_state  # (B, T_enc, D_whisper)
        projected = self.audio_proj(encoder_out)  # (B, T_enc, D_decoder)
        return projected * self.audio_pos_scale

    def forward(
        self,
        audio_features: torch.Tensor,
        text_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training: encode audio, prepend to text embeddings,
        run through decoder, return logits for text positions only.

        Args:
            audio_features: (B, n_mels, T_mel) from WhisperFeatureExtractor
            text_input_ids: (B, L_text) token IDs for the transcript

        Returns:
            logits: (B, L_text, vocab_size) — logits aligned with text positions
        """
        B, L_text = text_input_ids.shape
        device = text_input_ids.device

        # encode audio
        audio_embeds = self.encode_audio(audio_features)  # (B, T_audio, D)
        T_audio = audio_embeds.shape[1]

        # get text embeddings from decoder
        text_embeds = self._get_text_embeddings(text_input_ids)  # (B, L_text, D)

        # concatenate: [audio_prefix | text_tokens]
        combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)  # (B, T_audio+L_text, D)

        # build input_ids for Engram hashing: pad tokens for audio, real tokens for text
        audio_pad_ids = torch.full(
            (B, T_audio), self.pad_id, dtype=torch.long, device=device
        )
        combined_input_ids = torch.cat([audio_pad_ids, text_input_ids], dim=1)

        # run through decoder blocks
        logits = self._forward_with_embeddings(
            combined_embeds, combined_input_ids
        )  # (B, T_audio+L_text, vocab_size)

        # return only the text-position logits
        return logits[:, T_audio:, :]

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract raw token embeddings (no position embeddings) from the decoder.

        Position embeddings are omitted here because _forward_with_embeddings
        handles them for the full combined (audio + text) sequence.
        """
        # RealEngramTransformer
        if hasattr(self.decoder, "token_embedding"):
            return self.decoder.token_embedding(input_ids)
        # EngramGraft (wraps a HF model)
        if hasattr(self.decoder, "backbone"):
            backbone = self.decoder.backbone
            if hasattr(backbone, "transformer"):
                # GPT-2 — wte only, wpe applied by backbone forward
                return backbone.transformer.wte(input_ids)
            if hasattr(backbone, "model") and hasattr(backbone.model, "embed_tokens"):
                # Llama / Mistral (RoPE, no absolute pos embed)
                return backbone.model.embed_tokens(input_ids)
        raise ValueError("Cannot extract text embeddings from decoder")

    def _forward_with_embeddings(
        self,
        embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run pre-computed embeddings through the decoder, bypassing its embedding layer."""
        # RealEngramTransformer
        if hasattr(self.decoder, "blocks"):
            B, L, D = embeds.shape
            positions = torch.arange(L, device=embeds.device).unsqueeze(0)
            hidden_states = embeds + self.decoder.position_embedding(positions)
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.decoder.config.hc_mult, -1
            ).contiguous()
            for block in self.decoder.blocks:
                hidden_states = block(input_ids=input_ids, hidden_states=hidden_states)
            hidden_states = hidden_states[:, :, 0, :]
            hidden_states = self.decoder.final_norm(hidden_states)
            return self.decoder.lm_head(hidden_states)

        # EngramGraft — use HF's inputs_embeds to bypass the embedding layer
        # while keeping all internal routing (causal masks, etc.) intact
        if hasattr(self.decoder, "backbone"):
            self.decoder._input_ids_ref[0] = input_ids
            backbone = self.decoder.backbone
            outputs = backbone(inputs_embeds=embeds)
            self.decoder._input_ids_ref[0] = None
            return outputs.logits

        raise ValueError("Unsupported decoder type for embedding injection")

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str | Path,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: path to .wav/.mp3/.flac file
            max_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)

        Returns:
            transcribed text string
        """
        self.eval()
        device = next(self.parameters()).device

        # load and extract features
        audio = load_audio(audio_path)
        max_samples = int(self.config.max_audio_seconds * WHISPER_SAMPLE_RATE)
        audio = audio[:max_samples]

        features = self.feature_extractor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        )
        audio_features = features.input_features.to(device)  # (1, n_mels, T)

        # encode audio
        audio_embeds = self.encode_audio(audio_features)  # (1, T_audio, D)
        T_audio = audio_embeds.shape[1]

        # start with prefix token
        prefix_ids = self.tokenizer.encode(
            self.config.transcript_prefix, add_special_tokens=False
        )
        generated = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            B, L_text = generated.shape
            text_embeds = self._get_text_embeddings(generated)
            combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

            audio_pad_ids = torch.full(
                (B, T_audio), self.pad_id, dtype=torch.long, device=device
            )
            combined_ids = torch.cat([audio_pad_ids, generated], dim=1)

            logits = self._forward_with_embeddings(combined_embeds, combined_ids)
            next_logits = logits[:, -1, :]  # last position

            if temperature > 0:
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # decode, skipping the prefix
        output_ids = generated[0, len(prefix_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioTextDataset(Dataset):
    """
    Loads audio-transcript pairs from a JSONL manifest.

    Each line: {"audio": "path/to/file.wav", "text": "transcription"}
    Audio paths can be absolute or relative to the manifest's directory.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor: WhisperFeatureExtractor,
        config: AudioTrainConfig,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.config = config
        self.seq_len = config.seq_len
        self.max_audio_samples = int(config.max_audio_seconds * WHISPER_SAMPLE_RATE)

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
                            "text": entry["text"],
                        })

        print(f"  {len(self.samples)} audio-text pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # load audio
        audio = load_audio(sample["audio"])
        audio = audio[: self.max_audio_samples]

        # extract mel features
        features = self.feature_extractor(
            audio,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="np",
        )
        audio_features = features.input_features[0]  # (n_mels, T)

        # tokenize text
        text = sample["text"]
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            text_ids.append(eos)

        # truncate / pad text to seq_len + 1
        total_len = self.seq_len + 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos if eos is not None else 0

        if len(text_ids) > total_len:
            text_ids = text_ids[:total_len]
        elif len(text_ids) < total_len:
            text_ids = text_ids + [pad_id] * (total_len - len(text_ids))

        input_ids = torch.tensor(text_ids[:-1], dtype=torch.long)
        labels = torch.tensor(text_ids[1:], dtype=torch.long)

        # mask padding in labels
        labels[labels == pad_id] = IGNORE_INDEX

        return {
            "audio_features": torch.tensor(audio_features, dtype=torch.float32),
            "input_ids": input_ids,
            "labels": labels,
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Collate audio-text samples into a batch."""
    return {
        "audio_features": torch.stack([s["audio_features"] for s in batch]),
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AudioTrainer:
    """Trainer for audio-to-text with Engram models."""

    def __init__(
        self,
        model: AudioEngramModel,
        tokenizer: PreTrainedTokenizerBase,
        config: AudioTrainConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AudioTrainConfig()

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
        cfg = self.config

        print(f"Loading audio data from {data_path}...")
        dataset = AudioTextDataset(
            data_path, self.tokenizer,
            self.model.feature_extractor, cfg,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

        self.model.train()
        # keep encoder frozen even in train mode
        if cfg.freeze_encoder:
            self.model.encoder.eval()

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        total_loss = 0.0
        total_tokens = 0
        log_loss = 0.0
        log_steps = 0
        t0 = time.perf_counter()

        step = 0
        data_iter = self._infinite_loader(loader)

        print(f"\nAudio-to-text training for {cfg.total_steps} steps...")
        print(f"  batch_size={cfg.batch_size}  seq_len={cfg.seq_len}")
        print(f"  whisper encoder: {cfg.whisper_model} (frozen={cfg.freeze_encoder})")
        print("-" * 60)

        for step in range(cfg.total_steps):
            if step < resume_step:
                next(data_iter)
                continue

            batch = next(data_iter)
            audio_features = batch["audio_features"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

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
            n_tokens = (labels != IGNORE_INDEX).sum().item()
            total_tokens += n_tokens
            total_loss += loss.item() * n_tokens
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
                    f"  tokens={total_tokens:,}"
                    f"  {elapsed:.1f}s"
                )
                log_loss = 0.0
                log_steps = 0

            if (step + 1) % cfg.save_interval == 0:
                self._save_checkpoint(save_dir, step + 1)

        self._save_checkpoint(save_dir, step + 1)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        print(f"\nAudio training complete: {step + 1} steps")
        print(f"  avg loss: {avg_loss:.4f}")
        print(f"  text tokens: {total_tokens:,}")
        print(f"  wall time: {elapsed:.1f}s")

        return {
            "steps": step + 1,
            "tokens": total_tokens,
            "avg_loss": avg_loss,
            "wall_time": elapsed,
        }

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        while True:
            for batch in loader:
                yield batch

    def _save_checkpoint(self, save_dir: Path, step: int):
        path = save_dir / f"audio_step_{step}.pt"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Engram for audio-to-text")
    parser.add_argument("data", help="Path to JSONL manifest with audio/text pairs")

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

    # audio
    parser.add_argument("--whisper-model", default="openai/whisper-tiny",
                        help="Whisper encoder model (default: openai/whisper-tiny)")
    parser.add_argument("--max-audio-sec", type=float, default=30.0)

    # training
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-dir", default="checkpoints/audio_training")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)

    # inference
    parser.add_argument("--transcribe", type=str, default=None,
                        help="Transcribe a single audio file instead of training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint for transcription")

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

    audio_config = AudioTrainConfig(
        whisper_model=args.whisper_model,
        max_audio_seconds=args.max_audio_sec,
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

    # build decoder
    if args.graft:
        from engram_graft import get_model as get_graft_model
        print(f"Loading grafted model: {args.graft}")
        decoder, tokenizer = get_graft_model(
            model_name_or_path=args.graft,
            engram_layer_ids=args.engram_layers,
            device=device,
        )
        audio_config.mode = "graft"
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
        audio_config.mode = "full"

    # build audio model
    amodel = AudioEngramModel(decoder, tokenizer, audio_config).to(device)

    # transcribe mode
    if args.transcribe:
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
            amodel.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")
        text = amodel.transcribe(args.transcribe)
        print(f"\nTranscription: {text}")
        return

    # training mode
    trainer = AudioTrainer(amodel, tokenizer, audio_config)

    resume_step = 0
    if args.resume:
        resume_step = trainer.load_checkpoint(args.resume)

    trainer.train(args.data, resume_step=resume_step)


if __name__ == "__main__":
    main()
