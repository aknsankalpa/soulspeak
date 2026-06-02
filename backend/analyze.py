"""
analyze.py — SoulSpeak Audio Analysis Pipeline
===============================================
Pipeline:
  1. Load & resample audio to 16 kHz mono
  2. ACOUSTIC branch  → Wav2Vec2 model → raw trait probabilities
  3. LINGUISTIC branch → Whisper transcription → BERT model → raw trait probabilities
  4. Fuse scores (weighted average)
  5. Generate insights & recommendations

All heavy objects (models, tokenizers, whisper) are loaded once at startup
and cached as module-level singletons.
"""

import os
import logging
import random
import math
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("soulspeak.analyze")

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# ── Singleton model cache ──────────────────────────────────────────────────────

_wav2vec2_model  = None   # Wav2Vec2PersonalityModel
_wav2vec2_proc   = None   # Wav2Vec2Processor
_bert_model      = None   # BertPersonalityModel
_bert_tokenizer  = None   # BertTokenizer / AutoTokenizer
_whisper_model   = None   # whisper model for transcription
_device          = None

MODEL_STATUS = {
    "wav2vec2":  False,
    "bert":      False,
    "whisper":   False,
    "device":    "cpu",
}


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ── Startup loader (called once from app.py) ───────────────────────────────────

def load_all_models(wav2vec2_path: str = "wav2vec2_personality_best.pt",
                    bert_path:     str = "bert_personality_best.pt",
                    whisper_size:  str = "base") -> dict:
    """
    Load all models into module singletons.
    Call this once at Flask startup.
    Returns MODEL_STATUS dict.
    """
    global _wav2vec2_model, _wav2vec2_proc, _bert_model, _bert_tokenizer, _whisper_model

    device = get_device()
    MODEL_STATUS["device"] = str(device)
    log.info("🚀 Loading models on device: %s", device)

    # ── Wav2Vec2 ──────────────────────────────────────────────────────────────
    try:
        from models import load_wav2vec2_model
        _wav2vec2_model = load_wav2vec2_model(wav2vec2_path)
        if _wav2vec2_model:
            _wav2vec2_model.to(device)

            # Load matching processor (feature extractor + normaliser)
            from transformers import Wav2Vec2Processor, AutoFeatureExtractor
            backbone = _wav2vec2_model.wav2vec2.config._name_or_path \
                       if hasattr(_wav2vec2_model, "wav2vec2") else "facebook/wav2vec2-base"
            try:
                _wav2vec2_proc = Wav2Vec2Processor.from_pretrained(backbone)
            except Exception:
                _wav2vec2_proc = AutoFeatureExtractor.from_pretrained(backbone)

            MODEL_STATUS["wav2vec2"] = True
            log.info("✅ Wav2Vec2 ready")
    except Exception as e:
        log.error("❌ Wav2Vec2 load error: %s", e)

    # ── BERT ──────────────────────────────────────────────────────────────────
    try:
        from models import load_bert_model
        _bert_model = load_bert_model(bert_path)
        if _bert_model:
            _bert_model.to(device)

            from transformers import AutoTokenizer
            backbone = _bert_model.bert.config._name_or_path \
                       if hasattr(_bert_model, "bert") else "bert-base-uncased"
            _bert_tokenizer = AutoTokenizer.from_pretrained(backbone)

            MODEL_STATUS["bert"] = True
            log.info("✅ BERT ready")
    except Exception as e:
        log.error("❌ BERT load error: %s", e)

    # ── Whisper (transcription) ───────────────────────────────────────────────
    try:
        import whisper
        log.info("   Loading Whisper '%s' …", whisper_size)
        _whisper_model = whisper.load_model(whisper_size, device=str(device))
        MODEL_STATUS["whisper"] = True
        log.info("✅ Whisper ready")
    except ImportError:
        log.warning("⚠️  openai-whisper not installed — transcription unavailable")
    except Exception as e:
        log.error("❌ Whisper load error: %s", e)

    return MODEL_STATUS


# ── Audio loading ──────────────────────────────────────────────────────────────

def _load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load and resample audio to target sample rate, return mono float32 array."""
    import librosa, shutil, subprocess

    load_path = path
    tmp_wav = None
    ext = os.path.splitext(path)[-1].lower()
    if ext not in ('.wav', '.flac', '.aiff', '.aif') and shutil.which('ffmpeg'):
        tmp_wav = path + '_tmp.wav'
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-ar', str(target_sr),
                 '-ac', '1', '-f', 'wav', tmp_wav],
                capture_output=True, timeout=60, check=True
            )
            load_path = tmp_wav
            log.info("Pre-converted %s → WAV via ffmpeg", ext)
        except Exception as e:
            log.warning("ffmpeg pre-conversion failed (%s); using audioread fallback", e)
            tmp_wav = None

    try:
        audio, _ = librosa.load(load_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ── Acoustic branch (Wav2Vec2) ─────────────────────────────────────────────────

def _acoustic_scores(audio: np.ndarray) -> Optional[np.ndarray]:
    """Run audio through the Wav2Vec2 personality model. Returns shape (5,) or None."""
    if _wav2vec2_model is None or _wav2vec2_proc is None:
        return None

    device = get_device()
    try:
        inputs = _wav2vec2_proc(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = _wav2vec2_model(input_values, attention_mask=attention_mask)

        scores = logits.squeeze(0).cpu().numpy()   # (5,)
        log.debug("Acoustic raw scores: %s", scores)
        return scores
    except Exception as e:
        log.error("❌ Acoustic inference failed: %s", e)
        return None


# ── Transcription (Whisper) ────────────────────────────────────────────────────

def _transcribe(audio: np.ndarray) -> str:
    """Transcribe audio to text using Whisper."""
    if _whisper_model is None:
        return ""
    try:
        # Whisper expects float32 at 16 kHz
        result = _whisper_model.transcribe(audio, fp16=False, language="en")
        text = result.get("text", "").strip()
        log.debug("Transcript (%d chars): %s …", len(text), text[:80])
        return text
    except Exception as e:
        log.error("❌ Transcription failed: %s", e)
        return ""


# ── Linguistic branch (BERT) ───────────────────────────────────────────────────

def _linguistic_scores(text: str) -> Optional[np.ndarray]:
    """Run transcript through the BERT personality model. Returns shape (5,) or None."""
    if _bert_model is None or _bert_tokenizer is None or not text:
        return None

    device = get_device()
    try:
        # Tokenize — BERT max is 512 tokens; truncate long transcripts
        enc = _bert_tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            logits = _bert_model(input_ids, attention_mask, token_type_ids)

        scores = logits.squeeze(0).cpu().numpy()   # (5,)
        log.debug("Linguistic raw scores: %s", scores)
        return scores
    except Exception as e:
        log.error("❌ Linguistic inference failed: %s", e)
        return None


# ── Fallback simulation ────────────────────────────────────────────────────────

_PROFESSION_BASELINES = {
    "Healthcare Professional": [0.70, 0.75, 0.60, 0.88, 0.28],
    "Software Engineer":       [0.82, 0.78, 0.45, 0.65, 0.35],
    "Teacher / Educator":      [0.75, 0.72, 0.68, 0.82, 0.32],
    "Manager / Executive":     [0.68, 0.80, 0.72, 0.70, 0.30],
    "Sales / Marketing":       [0.65, 0.65, 0.85, 0.75, 0.38],
    "Lawyer / Legal":          [0.72, 0.85, 0.60, 0.58, 0.32],
    "Public Speaker / Presenter": [0.78, 0.70, 0.90, 0.72, 0.25],
    "Creative / Artist":       [0.92, 0.55, 0.62, 0.78, 0.42],
    "Student":                 [0.80, 0.62, 0.58, 0.76, 0.40],
    "Entrepreneur":            [0.85, 0.72, 0.78, 0.68, 0.35],
    "HR / Counsellor":         [0.74, 0.74, 0.65, 0.90, 0.28],
}
_DEFAULT_BASELINE = [0.68, 0.65, 0.60, 0.72, 0.35]


def _simulate_scores(duration: float, profession: str) -> np.ndarray:
    """Deterministic fallback when neither model is available."""
    base = _PROFESSION_BASELINES.get(profession, _DEFAULT_BASELINE)
    random.seed(int(duration * 31 + sum(ord(c) for c in profession)))
    noise = [random.uniform(-0.08, 0.08) for _ in range(5)]
    random.seed()
    scores = np.array([max(0.15, min(0.97, b + n)) for b, n in zip(base, noise)],
                      dtype=np.float32)
    return scores


# ── Score fusion ───────────────────────────────────────────────────────────────

def _fuse(acoustic: Optional[np.ndarray],
          linguistic: Optional[np.ndarray],
          acoustic_weight: float = 0.5) -> np.ndarray:
    """
    Weighted average of acoustic and linguistic predictions.
    Falls back gracefully if one branch is unavailable.
    """
    if acoustic is not None and linguistic is not None:
        w = acoustic_weight
        return w * acoustic + (1 - w) * linguistic
    if acoustic is not None:
        return acoustic
    if linguistic is not None:
        return linguistic
    return None   # both missing — caller will simulate


# ── Insight generation ────────────────────────────────────────────────────────

def _trait_label(score: float) -> str:
    if score >= 0.80: return "Very High"
    if score >= 0.65: return "High"
    if score >= 0.45: return "Moderate"
    if score >= 0.30: return "Low"
    return "Very Low"


def _build_insights(scores_pct: dict, profession: str) -> list:
    insights = []
    O, C, E, A, N = [scores_pct[t] for t in TRAITS]

    if A >= 78:
        insights.append({"type": "strength", "icon": "🤝",
            "text": f"Agreeableness ({A}%) — your warmth builds trust effortlessly. Lean into empathic listening before presenting key points."})
    if N <= 35:
        insights.append({"type": "strength", "icon": "🧘",
            "text": f"Low Neuroticism ({N}%) — you project calm under pressure, a rare and powerful communication asset."})
    if O >= 75:
        insights.append({"type": "strength", "icon": "💡",
            "text": f"Openness ({O}%) — your curiosity shows in your speech. Use vivid analogies to make complex ideas land."})
    if E >= 78:
        insights.append({"type": "strength", "icon": "🔊",
            "text": f"Extraversion ({E}%) — your natural energy commands rooms. Vary your pace to give listeners time to absorb key ideas."})
    if C < 65:
        insights.append({"type": "improve", "icon": "📐",
            "text": f"Conscientiousness ({C}%) — try the PEEL method: Point → Evidence → Explain → Link. Structured speech increases listener retention by ~30%."})
    if E < 50:
        insights.append({"type": "improve", "icon": "🎯",
            "text": f"Extraversion ({E}%) — practise deliberate eye contact and intentional pausing. Brief silences signal confidence, not hesitation."})
    if N > 55:
        insights.append({"type": "improve", "icon": "🌬️",
            "text": f"Neuroticism ({N}%) — try the 4-7-8 breath (inhale 4 counts, hold 7, exhale 8) before important conversations to ground your voice."})
    if A < 55:
        insights.append({"type": "improve", "icon": "🤲",
            "text": f"Agreeableness ({A}%) — mirror your listener's language and acknowledge their perspective before asserting your own."})
    return insights[:4]


# ── Main entry point ───────────────────────────────────────────────────────────

def analyze_audio(audio_path: str, duration: float, profession: str,
                  acoustic_weight: float = 0.5) -> dict:
    """
    Full analysis pipeline. Returns a result dict ready for the API response.

    Parameters
    ----------
    audio_path      : Path to audio file (webm, wav, mp3, ogg …)
    duration        : Recording length in seconds (used for fallback)
    profession      : User's declared profession
    acoustic_weight : Weight given to acoustic vs linguistic scores (0–1)
    """
    transcript = ""
    source_used = []

    # ── 1. Load audio ──────────────────────────────────────────────────────
    try:
        audio = _load_audio(audio_path)
    except Exception as e:
        log.error("❌ Audio load failed: %s", e)
        audio = None

    # Model-specific clip lengths to keep CPU inference under ~2 minutes
    # Wav2Vec2 attention is O(n²) in token length — 10 s keeps it manageable
    # Whisper handles longer audio efficiently — 30 s gives enough transcript
    WAV2VEC2_MAX = 10 * 16_000
    WHISPER_MAX  = 30 * 16_000

    # ── 2. Acoustic branch ─────────────────────────────────────────────────
    acoustic = None
    if audio is not None and _wav2vec2_model is not None:
        acoustic_audio = audio[:WAV2VEC2_MAX]
        log.info("Wav2Vec2 input: %.1f s", len(acoustic_audio) / 16_000)
        acoustic = _acoustic_scores(acoustic_audio)
        if acoustic is not None:
            source_used.append("acoustic (Wav2Vec2)")

    # ── 3. Transcription ───────────────────────────────────────────────────
    if audio is not None and _whisper_model is not None:
        whisper_audio = audio[:WHISPER_MAX]
        log.info("Whisper input: %.1f s", len(whisper_audio) / 16_000)
        transcript = _transcribe(whisper_audio)

    # ── 4. Linguistic branch ───────────────────────────────────────────────
    linguistic = None
    if transcript and _bert_model is not None:
        linguistic = _linguistic_scores(transcript)
        if linguistic is not None:
            source_used.append("linguistic (BERT)")

    # ── 5. Fuse ────────────────────────────────────────────────────────────
    fused = _fuse(acoustic, linguistic, acoustic_weight)
    if fused is None:
        log.warning("⚠️  Both models unavailable — using profession-based simulation")
        fused = _simulate_scores(duration, profession)
        source_used.append("simulation (fallback)")

    # ── 6. Convert to percentage integers ─────────────────────────────────
    scores_pct = {t: int(round(float(fused[i]) * 100)) for i, t in enumerate(TRAITS)}

    # ── 7. Build response dict ─────────────────────────────────────────────
    overall  = int(round(sum(scores_pct.values()) / len(TRAITS)))
    insights = _build_insights(scores_pct, profession)

    result = {
        "scores":      scores_pct,
        "labels":      {t: _trait_label(v / 100) for t, v in scores_pct.items()},
        "overall":     overall,
        "transcript":  transcript,
        "insights":    insights,
        "source":      ", ".join(source_used) if source_used else "simulation",
        "acoustic_weight": acoustic_weight if acoustic is not None else None,
        "models_used": {
            "wav2vec2": acoustic is not None,
            "bert":     linguistic is not None,
            "whisper":  bool(transcript),
        },
    }
    return result
