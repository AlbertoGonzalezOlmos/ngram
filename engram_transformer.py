"""
Full Transformer with real attention and FFN layers plus Engram modules.

Unlike the demo (which mocks attention/MoE as identity), this module provides
real causal multi-head attention and a SwiGLU feed-forward network that operate
on the hyper-connection shape (B, L, hc_mult, D).

Usage:
    from engram_transformer import get_model

    model, tokenizer = get_model()                        # default config
    model, tokenizer = get_model(force_rebuild=True)      # fresh weights

Weights are randomly initialized — you must train this model before it
produces coherent text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import engram_demo_v1
from engram_demo_v1 import Engram, EngramConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RealBackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    num_attention_heads: int = 8
    ffn_hidden_size: int = 4096       # typically 4x hidden_size
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # Engram overrides (forwarded to EngramConfig globals)
    engram_layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"


# ---------------------------------------------------------------------------
# Attention operating on (B, L, hc_mult, D)
# ---------------------------------------------------------------------------

class HCCausalAttention(nn.Module):
    """
    Causal multi-head self-attention that accepts (B, L, hc_mult, D).

    All hc_mult streams share the same QKV projection weights but have
    independent RMSNorm parameters.  The hc_mult dimension is folded into
    the batch dimension so standard MHA mechanics apply.
    """

    def __init__(self, hidden_size: int, num_heads: int, hc_mult: int,
                 dropout: float = 0.0, norm_eps: float = 1e-5):
        super().__init__()
        self.hc_mult = hc_mult
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, G, D) -> (B, L, G, D)"""
        B, L, G, D = x.shape

        # per-stream norm then flatten into batch
        normed = torch.stack(
            [self.norms[i](x[:, :, i, :]) for i in range(G)], dim=2
        )
        x_flat = normed.reshape(B * G, L, D)               # (B*G, L, D)

        qkv = self.qkv(x_flat)                              # (B*G, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head: (B*G, L, nH, hD) -> (B*G, nH, L, hD)
        q = q.view(B * G, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * G, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * G, L, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product with causal mask (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                    # (B*G, nH, L, hD)

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B * G, L, D)
        )
        out = self.out_proj(attn_out)                        # (B*G, L, D)
        return out.view(B, G, L, D).permute(0, 2, 1, 3)     # (B, L, G, D)


# ---------------------------------------------------------------------------
# SwiGLU FFN operating on (B, L, hc_mult, D)
# ---------------------------------------------------------------------------

class HCFFN(nn.Module):
    """
    SwiGLU feed-forward network for (B, L, hc_mult, D) tensors.

    Shared weights across hc_mult streams, per-stream RMSNorm.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, hc_mult: int,
                 dropout: float = 0.0, norm_eps: float = 1e-5):
        super().__init__()
        self.hc_mult = hc_mult
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, G, D) -> (B, L, G, D)"""
        B, L, G, D = x.shape

        normed = torch.stack(
            [self.norms[i](x[:, :, i, :]) for i in range(G)], dim=2
        )
        x_flat = normed.reshape(B * L * G, D)

        out = self.down_proj(
            self.dropout(F.silu(self.gate_proj(x_flat)) * self.up_proj(x_flat))
        )
        return out.view(B, L, G, D)


# ---------------------------------------------------------------------------
# Transformer block with real attention + FFN + optional Engram
# ---------------------------------------------------------------------------

class RealTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: RealBackBoneConfig):
        super().__init__()
        D = config.hidden_size
        self.attn = HCCausalAttention(
            D, config.num_attention_heads, config.hc_mult,
            dropout=config.dropout, norm_eps=config.norm_eps,
        )
        self.ffn = HCFFN(
            D, config.ffn_hidden_size, config.hc_mult,
            dropout=config.dropout, norm_eps=config.norm_eps,
        )
        self.engram = None
        if layer_id in config.engram_layer_ids:
            self.engram = Engram(layer_id=layer_id)

    def forward(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.engram is not None:
            hidden_states = (
                self.engram(hidden_states=hidden_states, input_ids=input_ids)
                + hidden_states
            )
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.ffn(hidden_states) + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class RealEngramTransformer(nn.Module):
    """
    Transformer with real causal attention, SwiGLU FFN, and Engram modules.

    Shape flow:
        input_ids (B, L)
        -> token embed (B, L, D)
        -> + positional embed
        -> expand to (B, L, hc_mult, D)
        -> N x RealTransformerBlock
        -> collapse to (B, L, D)
        -> LM head -> logits (B, L, vocab_size)
    """

    def __init__(self, config: RealBackBoneConfig | None = None):
        super().__init__()
        self.config = config or RealBackBoneConfig()
        cfg = self.config

        # --- patch global singletons so Engram.__init__ reads correct values ---
        engram_demo_v1.backbone_config.hidden_size = cfg.hidden_size
        engram_demo_v1.backbone_config.hc_mult = cfg.hc_mult
        engram_demo_v1.backbone_config.vocab_size = cfg.vocab_size
        engram_demo_v1.backbone_config.num_layers = cfg.num_layers

        engram_demo_v1.engram_cfg.layer_ids = list(cfg.engram_layer_ids)
        engram_demo_v1.engram_cfg.tokenizer_name_or_path = cfg.tokenizer_name_or_path
        engram_demo_v1.engram_cfg.kernel_size = 4

        # --- layers ---
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.hidden_size)

        self.blocks = nn.ModuleList(
            [RealTransformerBlock(i, cfg) for i in range(cfg.num_layers)]
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L)
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        # expand to hyper-connection shape
        hidden_states = hidden_states.unsqueeze(2).expand(
            -1, -1, self.config.hc_mult, -1
        ).contiguous()

        for block in self.blocks:
            hidden_states = block(input_ids=input_ids, hidden_states=hidden_states)

        # collapse: take stream 0
        hidden_states = hidden_states[:, :, 0, :]
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Save / Load / Get
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/engram_real_transformer")
_WEIGHTS_FILE = "model.pt"
_CONFIG_FILE = "config.pt"


def build_model(config: RealBackBoneConfig | None = None) -> RealEngramTransformer:
    return RealEngramTransformer(config)


def save_model(
    model: RealEngramTransformer,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / _WEIGHTS_FILE)
    torch.save({"config": model.config}, checkpoint_dir / _CONFIG_FILE)
    return checkpoint_dir


def load_model(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> RealEngramTransformer:
    checkpoint_dir = Path(checkpoint_dir)
    config_data = torch.load(checkpoint_dir / _CONFIG_FILE, weights_only=False)
    model = build_model(config_data["config"])
    state = torch.load(checkpoint_dir / _WEIGHTS_FILE, weights_only=True)
    model.load_state_dict(state)
    return model


def get_model(
    config: RealBackBoneConfig | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    force_rebuild: bool = False,
) -> tuple[RealEngramTransformer, AutoTokenizer]:
    """
    Load a saved model or build a new one.

    Returns:
        (model, tokenizer)
    """
    checkpoint_dir = Path(checkpoint_dir)
    weights_path = checkpoint_dir / _WEIGHTS_FILE

    if not force_rebuild and weights_path.exists():
        print(f"Loading real Engram transformer from {checkpoint_dir}")
        model = load_model(checkpoint_dir)
    else:
        print("Building new real Engram transformer...")
        model = build_model(config)
        save_model(model, checkpoint_dir)
        print(f"Model saved to {checkpoint_dir}")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model.config.tokenizer_name_or_path, trust_remote_code=True
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = RealBackBoneConfig(
        hidden_size=256,
        hc_mult=4,
        vocab_size=129280,
        num_layers=6,
        num_attention_heads=4,
        ffn_hidden_size=1024,
        engram_layer_ids=[1, 4],
    )
    model, tokenizer = get_model(cfg, force_rebuild=True)

    text = "Only Alexander the Great could tame the horse Bucephalus."
    ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        logits = model(ids)

    print(f"input_ids.shape = {ids.shape}")
    print(f"logits.shape    = {logits.shape}")
    print("RealEngramTransformer forward pass OK")
