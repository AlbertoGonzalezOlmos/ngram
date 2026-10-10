"""
Build, save, and load a Transformer with Engram modules.

Usage:
    from model_manager import get_model

    model, tokenizer = get_model()                # default path: checkpoints/engram_model
    model, tokenizer = get_model("my/path")       # custom path
    model, tokenizer = get_model(force_rebuild=True)  # ignore any saved checkpoint
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from engram_demo_v1 import (
    EngramConfig,
    BackBoneConfig,
    TransformerBlock,
    engram_cfg,
    backbone_config,
)

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/engram_model")
_WEIGHTS_FILE = "model.pt"
_CONFIG_FILE = "config.pt"


class EngramTransformer(nn.Module):
    """Full Transformer backbone with Engram modules at configured layers."""

    def __init__(
        self,
        engram_config: EngramConfig | None = None,
        bb_config: BackBoneConfig | None = None,
    ):
        super().__init__()
        self.engram_config = engram_config or engram_cfg
        self.bb_config = bb_config or backbone_config

        self.embedding = nn.Embedding(
            self.bb_config.vocab_size, self.bb_config.hidden_size
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(layer_id=i)
                for i in range(self.bb_config.num_layers)
            ]
        )
        self.head = nn.Linear(
            self.bb_config.hidden_size, self.bb_config.vocab_size
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids
        Returns:
            logits: (B, L, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        # mock hyper-connection: expand to (B, L, hc_mult, D)
        hidden_states = hidden_states.unsqueeze(2).expand(
            -1, -1, self.bb_config.hc_mult, -1
        )

        for block in self.blocks:
            hidden_states = block(input_ids=input_ids, hidden_states=hidden_states)

        # mock hyper-connection: collapse back to (B, L, D)
        hidden_states = hidden_states[:, :, 0, :]
        logits = self.head(hidden_states)
        return logits


def build_model(
    engram_config: EngramConfig | None = None,
    bb_config: BackBoneConfig | None = None,
) -> EngramTransformer:
    """Build a fresh model with random weights."""
    return EngramTransformer(engram_config, bb_config)


def save_model(model: EngramTransformer, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Save model weights and config needed to rebuild it."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / _WEIGHTS_FILE)

    config_data = {
        "engram_config": model.engram_config,
        "bb_config": model.bb_config,
    }
    torch.save(config_data, checkpoint_dir / _CONFIG_FILE)

    return checkpoint_dir


def load_model(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> EngramTransformer:
    """Rebuild the model architecture from config, then load saved weights."""
    checkpoint_dir = Path(checkpoint_dir)

    config_data = torch.load(checkpoint_dir / _CONFIG_FILE, weights_only=False)
    model = build_model(
        engram_config=config_data["engram_config"],
        bb_config=config_data["bb_config"],
    )

    state = torch.load(checkpoint_dir / _WEIGHTS_FILE, weights_only=True)
    model.load_state_dict(state)

    return model


def get_model(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    force_rebuild: bool = False,
) -> tuple[EngramTransformer, AutoTokenizer]:
    """
    Load a saved model if one exists, otherwise build and save a new one.

    Args:
        checkpoint_dir: where to look for / store the checkpoint.
        force_rebuild: if True, always build from scratch and overwrite.

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / _WEIGHTS_FILE

    if not force_rebuild and weights_path.exists():
        print(f"Loading model from {checkpoint_dir}")
        model = load_model(checkpoint_dir)
    else:
        print("Building new model...")
        model = build_model()
        save_model(model, checkpoint_dir)
        print(f"Model saved to {checkpoint_dir}")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model.engram_config.tokenizer_name_or_path, trust_remote_code=True
    )

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = get_model()

    text = "Only Alexander the Great could tame the horse Bucephalus."
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        logits = model(input_ids)

    print(f"input_ids.shape = {input_ids.shape}")
    print(f"logits.shape    = {logits.shape}")
