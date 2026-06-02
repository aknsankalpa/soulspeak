"""
models.py — SoulSpeak Personality Model Definitions
====================================================
Defines PyTorch model classes for Big Five personality prediction:
  - Wav2Vec2PersonalityModel  → acoustic features from raw audio
  - BertPersonalityModel      → linguistic features from transcript text

Both models output 5 scores in [0,1] for:
  [Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism]
"""

import os
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict

log = logging.getLogger("soulspeak.models")

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# ── Wav2Vec2 personality model ─────────────────────────────────────────────────

class Wav2Vec2PersonalityModel(nn.Module):
    """
    Wav2Vec2 backbone + mean-pool attention + MLP head for Big Five prediction.
    Matches the common fine-tuning approach used in speech personality research.

    Architecture:
        Wav2Vec2Model (frozen or partially frozen backbone)
        → mean pooling over time frames (with optional attention mask)
        → LayerNorm
        → Dropout(0.1)
        → Linear(hidden_size → 256)
        → GELU
        → Dropout(0.1)
        → Linear(256 → 5)
        → Sigmoid
    """

    def __init__(self, pretrained_name: str = "facebook/wav2vec2-base", num_traits: int = 5):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_name)
        hidden = self.wav2vec2.config.hidden_size   # 768 for base, 1024 for large

        self.norm     = nn.LayerNorm(hidden)
        self.drop1    = nn.Dropout(0.1)
        self.fc1      = nn.Linear(hidden, 256)
        self.act      = nn.GELU()
        self.drop2    = nn.Dropout(0.1)
        self.fc2      = nn.Linear(256, num_traits)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state            # (B, T, H)

        # Attention-masked mean pooling
        if attention_mask is not None:
            # Propagate input attention mask to the transformer output frames
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                _compute_mask_indices,
            )
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)                # (B, H)

        x = self.norm(pooled)
        x = self.drop1(x)
        x = self.act(self.fc1(x))
        x = self.drop2(x)
        return self.sigmoid(self.fc2(x))              # (B, 5)


# ── BERT personality model ─────────────────────────────────────────────────────

class BertPersonalityModel(nn.Module):
    """
    BERT backbone + CLS-token pooling + MLP head for Big Five prediction.

    Architecture:
        BertModel (backbone)
        → CLS token hidden state
        → LayerNorm
        → Dropout(0.1)
        → Linear(hidden_size → 256)
        → GELU
        → Dropout(0.1)
        → Linear(256 → 5)
        → Sigmoid
    """

    def __init__(self, pretrained_name: str = "bert-base-uncased", num_traits: int = 5):
        super().__init__()
        from transformers import BertModel
        self.bert    = BertModel.from_pretrained(pretrained_name)
        hidden       = self.bert.config.hidden_size    # 768 base, 1024 large

        self.norm    = nn.LayerNorm(hidden)
        self.drop1   = nn.Dropout(0.1)
        self.fc1     = nn.Linear(hidden, 256)
        self.act     = nn.GELU()
        self.drop2   = nn.Dropout(0.1)
        self.fc2     = nn.Linear(256, num_traits)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls = outputs.last_hidden_state[:, 0, :]       # CLS token  (B, H)

        x = self.norm(cls)
        x = self.drop1(x)
        x = self.act(self.fc1(x))
        x = self.drop2(x)
        return self.sigmoid(self.fc2(x))              # (B, 5)


# ── Smart loader ───────────────────────────────────────────────────────────────

def _infer_pretrained(state_dict: dict, model_type: str) -> str:
    """Guess the backbone variant from weight shapes inside the state dict."""
    if model_type == "wav2vec2":
        # Check projection layer hidden size
        for k, v in state_dict.items():
            if "wav2vec2.encoder.layers.0.attention.q_proj.weight" in k:
                return "facebook/wav2vec2-large" if v.shape[0] == 1024 else "facebook/wav2vec2-base"
        return "facebook/wav2vec2-base"
    else:  # bert
        for k, v in state_dict.items():
            if "bert.encoder.layer.0.attention.self.query.weight" in k:
                return "bert-large-uncased" if v.shape[0] == 1024 else "bert-base-uncased"
        return "bert-base-uncased"


def _try_load_state_dict(model: nn.Module, state_dict: dict, path: str) -> bool:
    """Try strict, then non-strict loading; return True on success."""
    # Strip common prefixes that differ between saving conventions
    cleaned = {}
    for k, v in state_dict.items():
        # Some checkpoints wrap everything under "model." or "module."
        key = k
        for prefix in ("model.", "module.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    for strict in (True, False):
        try:
            missing, unexpected = model.load_state_dict(cleaned, strict=strict)
            if not strict and missing:
                log.warning("  %d missing keys (non-strict load): %s …",
                            len(missing), missing[:3])
            log.info("✅ Loaded %s (strict=%s)", path, strict)
            return True
        except RuntimeError as e:
            log.debug("  strict=%s failed: %s", strict, e)
    return False


def load_wav2vec2_model(path: str) -> Wav2Vec2PersonalityModel | None:
    """Load wav2vec2_personality_best.pt into the personality model."""
    if not os.path.exists(path):
        log.warning("⚠️  wav2vec2 model not found: %s", path)
        return None
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.error("❌ Could not torch.load %s: %s", path, e)
        return None

    # Case 1: already a full nn.Module
    if isinstance(raw, nn.Module):
        raw.eval()
        log.info("✅ Loaded full wav2vec2 model from %s", path)
        return raw

    # Case 2: state dict
    state_dict = raw if isinstance(raw, dict) else raw.get("state_dict", raw)
    pretrained  = _infer_pretrained(state_dict, "wav2vec2")
    log.info("   Detected backbone: %s", pretrained)

    try:
        model = Wav2Vec2PersonalityModel(pretrained_name=pretrained)
    except Exception as e:
        log.error("❌ Could not instantiate Wav2Vec2PersonalityModel: %s", e)
        return None

    if _try_load_state_dict(model, state_dict, path):
        model.eval()
        return model

    log.error("❌ State dict mismatch for %s — falling back to simulation", path)
    return None


def load_bert_model(path: str) -> BertPersonalityModel | None:
    """Load bert_personality_best.pt into the personality model."""
    if not os.path.exists(path):
        log.warning("⚠️  BERT model not found: %s", path)
        return None
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.error("❌ Could not torch.load %s: %s", path, e)
        return None

    if isinstance(raw, nn.Module):
        raw.eval()
        log.info("✅ Loaded full BERT model from %s", path)
        return raw

    state_dict = raw if isinstance(raw, dict) else raw.get("state_dict", raw)
    pretrained  = _infer_pretrained(state_dict, "bert")
    log.info("   Detected backbone: %s", pretrained)

    try:
        model = BertPersonalityModel(pretrained_name=pretrained)
    except Exception as e:
        log.error("❌ Could not instantiate BertPersonalityModel: %s", e)
        return None

    if _try_load_state_dict(model, state_dict, path):
        model.eval()
        return model

    log.error("❌ State dict mismatch for %s — falling back to simulation", path)
    return None


# ── Session data model ─────────────────────────────────────────────────────────

@dataclass
class Session:
    id: Optional[int] = None
    user: str = 'anonymous'
    created_at: Optional[str] = None
    traits: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)
