"""Runtime configuration from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ANVESH_DATA_DIR", "./data")).resolve())
    db_url: str = os.getenv("ANVESH_DB_URL", "")
    data_key: str | None = os.getenv("ANVESH_DATA_KEY")  # legacy evidence key (read-only, for re-encryption)
    # exact frontend origins allowed to call the API with credentials (also checked on WebSocket handshakes)
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv("ANVESH_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if o.strip() and o.strip() != "*")

    # Speech models (all optional; the API reports what is loaded at /api/health)
    asr_model: str = os.getenv("ANVESH_ASR_MODEL", "small")
    asr_device: str = os.getenv("ANVESH_ASR_DEVICE", "cpu")
    asr_compute: str = os.getenv("ANVESH_ASR_COMPUTE", "int8")
    enable_asr: bool = _bool("ANVESH_ENABLE_ASR", True)
    # "hi" transcribes Hindi/Hinglish faithfully (Devanagari, then transliterated to Roman Hinglish);
    # "en" makes Whisper TRANSLATE Hindi into English, so wording is lost;
    # "auto" detects but only among ANVESH_ASR_LANGS (prevents mislabels such as 'jw').
    # "auto" (default) detects the caller's language once per call among ANVESH_ASR_LANGS and keeps it; the officer
    # can also set it at intake. Indian languages other than Hindi go to IndicConformer (speech/indic_asr.py).
    asr_language: str = os.getenv("ANVESH_ASR_LANGUAGE", "auto")
    asr_languages: tuple[str, ...] = tuple(os.getenv(
        "ANVESH_ASR_LANGS", "hi,en,bn,ta,te,mr,gu,kn,ml,pa,ur,as,ne,sa,sd,fr,de,es,ru,ar,zh").split(","))
    # Spoken Hindi and Urdu are close; Hinglish callers are often detected as Urdu. Treat Urdu as Hindi unless off.
    asr_urdu_as_hindi: bool = _bool("ANVESH_ASR_URDU_AS_HINDI", True)
    enable_indic_asr: bool = _bool("ANVESH_ENABLE_INDIC_ASR", True)
    indic_asr_dir: str = os.getenv("ANVESH_INDIC_ASR_DIR", "")
    # Hindi stays on Whisper by default: the Hinglish layer (Roman script, English loanwords) is built around it
    indic_asr_for_hindi: bool = _bool("ANVESH_INDIC_ASR_HINDI", False)
    asr_report_language: str = os.getenv("ANVESH_ASR_REPORT_LANGUAGE", "")
    # accurate model used after a live call ends and for uploads (re-transcribes the full recording)
    asr_final_model: str = os.getenv("ANVESH_ASR_FINAL_MODEL", "large-v3-turbo")
    enable_asr_final: bool = _bool("ANVESH_ENABLE_ASR_FINAL", True)
    emotion_model: str = os.getenv("ANVESH_EMOTION_MODEL", "iic/emotion2vec_plus_base")
    enable_emotion: bool = _bool("ANVESH_ENABLE_EMOTION", True)
    # telephone-adapted head on emotion2vec+ embeddings ("auto" = use it when its weights file is present, "off" = raw model)
    ser_head: str = os.getenv("ANVESH_SER_HEAD", "auto")
    enable_speaker: bool = _bool("ANVESH_ENABLE_SPEAKER", True)
    # synthetic / cloned voice detection (audio-classification model with real vs fake labels)
    spoof_model: str = os.getenv("ANVESH_SPOOF_MODEL", "MelodyMachine/Deepfake-audio-detection-V2")
    enable_spoof: bool = _bool("ANVESH_ENABLE_SPOOF", True)
    text_embedder: str = os.getenv("ANVESH_TEXT_EMBEDDER", "")  # e.g. paraphrase-multilingual-MiniLM-L12-v2

    live_chunk_seconds: float = float(os.getenv("ANVESH_LIVE_CHUNK_SECONDS", "5"))
    # Voice distress per language (EVALUATION.md, "Languages"): VALIDATED languages passed the pre-stated test
    # (held out of training, phone audio) and may change priority; MEASURED ones were tested but did not pass, so
    # the voice reading is shown as indicative only and never changes priority; any other language abstains.
    validated_languages: tuple[str, ...] = tuple(os.getenv("ANVESH_VALIDATED_LANGS", "en,bn,mni").split(","))
    measured_languages: tuple[str, ...] = tuple(os.getenv(
        "ANVESH_MEASURED_LANGS",
        "hi,as,brx,doi,gu,kn,ks,kok,mai,ml,mr,ne,or,pa,sa,sat,sd,ta,te,ur,fr,it,pl,ru,es,de").split(","))

    @property
    def database_url(self) -> str:
        return self.db_url or f"sqlite:///{self.data_dir / 'anvesh.db'}"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
