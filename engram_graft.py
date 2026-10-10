"""
Graft Engram modules onto a pretrained HuggingFace causal language model.

The pretrained backbone is frozen; only the Engram parameters are trainable.
This gives you coherent generation from the pretrained LM with the Engram
N-gram lookup/fusion pipeline running at configured layers.

Usage:
    from engram_graft import get_model

    model, tokenizer = get_model("gpt2")
    model, tokenizer = get_model("meta-llama/Llama-3.2-1B-Instruct")

    # fine-tune only Engram parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import engram_demo_v1
from engram_demo_v1 import Engram, EngramConfig




# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GraftConfig:
    """Configuration for grafting Engram onto a pretrained model."""
    model_name_or_path: str = "gpt2"
    engram_layer_ids: List[int] = field(default_factory=lambda: [1, 5])
    engram_vocab_size_factor: int = 5
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    hc_mult: int = 4
    kernel_size: int = 4
    freeze_backbone: bool = True
    torch_dtype: str = "auto"


# ---------------------------------------------------------------------------
# Hyper-connection adapter
# ---------------------------------------------------------------------------

class HyperConnectionAdapter(nn.Module):
    """Bridges (B, L, D) <-> (B, L, hc_mult, D) around an Engram module.

    Initialized as a no-op: collapse extracts stream 0 (identity) and the
    gate starts at 0, so the pretrained backbone is unperturbed until the
    Engram parameters are fine-tuned.
    """

    def __init__(self, hidden_size: int, hc_mult: int):
        super().__init__()
        self.hc_mult = hc_mult

        # Learnable gate initialized to 0 → Engram contribution starts at zero
        self.gate = nn.Parameter(torch.zeros(1))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, L, hc_mult, D) via broadcast."""
        return x.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def collapse(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, hc_mult, D) -> (B, L, D) by taking stream 0."""
        return x[:, :, 0, :]


# ---------------------------------------------------------------------------
# Helpers to locate decoder layers across HF architectures
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the list of decoder blocks for common HF architectures."""
    # GPT-2 / GPT-Neo
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Llama / Mistral / Qwen / DeepSeek / Gemma / Phi
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-NeoX / Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        f"Cannot find decoder layers for {type(model).__name__}. "
        "Add support in _get_decoder_layers()."
    )


# ---------------------------------------------------------------------------
# Main grafted model
# ---------------------------------------------------------------------------

class EngramGraft(nn.Module):
    """Pretrained HF causal LM with Engram modules injected at specific layers."""

    def __init__(self, config: GraftConfig, device: str | None = None):
        super().__init__()
        self.config = config

        # --- load pretrained model + tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        )

        hidden_size = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        # --- patch global configs so Engram.__init__ picks up the right sizes ---
        engram_demo_v1.backbone_config.hidden_size = hidden_size
        engram_demo_v1.backbone_config.hc_mult = config.hc_mult
        engram_demo_v1.backbone_config.vocab_size = vocab_size

        engram_demo_v1.engram_cfg.tokenizer_name_or_path = config.model_name_or_path
        engram_demo_v1.engram_cfg.layer_ids = list(config.engram_layer_ids)
        engram_demo_v1.engram_cfg.max_ngram_size = config.max_ngram_size
        engram_demo_v1.engram_cfg.n_embed_per_ngram = config.n_embed_per_ngram
        engram_demo_v1.engram_cfg.n_head_per_ngram = config.n_head_per_ngram
        engram_demo_v1.engram_cfg.kernel_size = config.kernel_size
        engram_demo_v1.engram_cfg.pad_id = pad_id
        engram_demo_v1.engram_cfg.engram_vocab_size = [
            vocab_size * config.engram_vocab_size_factor
        ] * (config.max_ngram_size - 1)

        # --- freeze backbone ---
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- create Engram + adapter per target layer and register hooks ---
        decoder_layers = _get_decoder_layers(self.backbone)
        num_layers = len(decoder_layers)

        self.engram_modules = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        self._hooks: list = []

        # mutable container so hooks can read input_ids set in forward()
        self._input_ids_ref: list[torch.Tensor | None] = [None]

        for layer_id in config.engram_layer_ids:
            if layer_id >= num_layers:
                raise ValueError(
                    f"engram_layer_ids contains {layer_id} but model only "
                    f"has {num_layers} layers (0-indexed)."
                )
            key = str(layer_id)
            self.engram_modules[key] = Engram(layer_id=layer_id)
            self.adapters[key] = HyperConnectionAdapter(hidden_size, config.hc_mult)

            hook = decoder_layers[layer_id].register_forward_hook(
                self._make_hook(key)
            )
            self._hooks.append(hook)

    # ---- hook factory ----

    def _make_hook(self, key: str):
        """Create a post-forward hook that runs Engram on the layer output."""

        def hook(module, args, output):
            engram_mod = self.engram_modules[key]
            adapter = self.adapters[key]
            input_ids = self._input_ids_ref[0]

            # HF layers return a tuple; hidden_states is always element 0
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            hs_4d = adapter.expand(hidden_states)                # (B,L,hc,D)
            engram_out = engram_mod(hs_4d, input_ids)            # (B,L,hc,D)
            # gate starts at 0 → tanh(0)=0 → no Engram influence until trained
            hs_4d = hs_4d + torch.tanh(adapter.gate) * engram_out
            hidden_states = adapter.collapse(hs_4d)              # (B,L,D)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook

    # ---- forward ----

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run the backbone forward pass with Engram hooks active.

        Returns:
            logits: (B, L, vocab_size)
        """
        self._input_ids_ref[0] = input_ids
        outputs = self.backbone(input_ids=input_ids, **kwargs)
        self._input_ids_ref[0] = None
        return outputs.logits

    def remove_hooks(self):
        """Remove all registered hooks (useful before serialization)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Save / load (Engram params only)
# ---------------------------------------------------------------------------

def save_engram_params(model: EngramGraft, path: str) -> None:
    """Save only the trainable Engram + adapter parameters."""
    state = {
        "engram_modules": model.engram_modules.state_dict(),
        "adapters": model.adapters.state_dict(),
        "config": model.config,
    }
    torch.save(state, path)


def load_engram_params(model: EngramGraft, path: str) -> None:
    """Load previously saved Engram + adapter parameters into an existing model."""
    state = torch.load(path, weights_only=False)
    model.engram_modules.load_state_dict(state["engram_modules"])
    model.adapters.load_state_dict(state["adapters"])


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def get_model(
    model_name_or_path: str = "gpt2",
    engram_layer_ids: list[int] | None = None,
    device: str | None = None,
    **graft_kwargs,
) -> tuple[EngramGraft, AutoTokenizer]:
    """
    Build an EngramGraft model ready for inference or fine-tuning.

    Args:
        model_name_or_path: any HuggingFace causal LM identifier or local path.
        engram_layer_ids: which decoder layers to inject Engram into.
            Defaults to [1, 5] (early layers, where N-gram patterns help most).
        device: target device (auto-detected if None).
        **graft_kwargs: extra fields forwarded to GraftConfig.

    Returns:
        (model, tokenizer)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    cfg = GraftConfig(
        model_name_or_path=model_name_or_path,
        engram_layer_ids=engram_layer_ids or [1, 5],
        **graft_kwargs,
    )

    model = EngramGraft(cfg, device=device)
    model = model.to(device)
    model.eval()

    return model, model.tokenizer


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model, tokenizer = get_model("gpt2", engram_layer_ids=[1, 5])
    device = next(model.parameters()).device

    text = "The quick brown fox"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(ids)

    print(f"input_ids.shape = {ids.shape}")
    print(f"logits.shape    = {logits.shape}")
    print("EngramGraft forward pass OK")
