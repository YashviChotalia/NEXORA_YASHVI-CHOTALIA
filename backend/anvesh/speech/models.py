"""Replaceable model adapters: ASR, voice emotion, speaker embedding.

Models load in a background thread at startup (`warmup()`), so a first live
call never waits on a download. Until a model is ready its signal is simply
absent and downstream logic abstains; /api/health reports loading / ready /
error. Nothing is ever simulated.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings

log = logging.getLogger("anvesh.models")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


@dataclass
class ModelStatus:
    name: str
    model: str
    loaded: bool
    state: str  # disabled | idle | loading | ready | error
    error: str | None = None
    license: str | None = None


class _Lazy:
    name = "model"
    license = None

    def __init__(self, model_id: str, enabled: bool):
        self.model_id = model_id
        self.enabled = enabled
        self._model = None
        self._error: str | None = None if enabled else "disabled by configuration"
        self._lock = threading.Lock()
        self._loading = False

    def _load(self):  # pragma: no cover - needs model files
        raise NotImplementedError

    def _do_load(self) -> None:
        with self._lock:
            if self._model is not None or self._error is not None:
                return
            self._loading = True
            try:
                self._model = self._load()
                log.info("%s ready: %s", self.name, self.model_id)
            except Exception as e:  # ImportError, download failure, ...
                self._error = f"{type(e).__name__}: {e}"[:300]
                log.warning("%s unavailable: %s", self.name, self._error)
            finally:
                self._loading = False

    def start_loading(self) -> None:
        if self.enabled and self._model is None and self._error is None and not self._loading:
            self._loading = True
            threading.Thread(target=self._do_load, name=f"load-{self.name}", daemon=True).start()

    def get(self, block: bool = True):
        """Return the model. With block=False, never wait: start loading and return None until ready."""
        if not self.enabled or self._error is not None:
            return None
        if self._model is None:
            if block:
                self._do_load()
            else:
                self.start_loading()
        return self._model

    @property
    def state(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._model is not None:
            return "ready"
        if self._error:
            return "error"
        return "loading" if self._loading else "idle"

    def status(self) -> ModelStatus:
        return ModelStatus(self.name, self.model_id, self._model is not None, self.state, self._error, self.license)


HOTWORDS = ("UPI PhonePe Paytm GooglePay GPay SBI HDFC ICICI Axis Kotak PNB OTP KYC AnyDesk CBI customs "
            "lakh hazaar rupaye account number transfer at the rate okaxis ybl")


def _clean_segments(segs, offset: float, hotwords: str, stats: dict | None = None) -> list[dict]:
    """Drop typical Whisper hallucinations: text on silence, prompt echoes, looping text."""
    hw = set(hotwords.lower().split())
    out = []
    drop = (stats or {}).setdefault("dropped", {}) if stats is not None else {}
    for s in segs:
        text = s.text.strip()
        if not text:
            continue
        words = text.lower().replace(",", " ").split()
        if s.no_speech_prob > 0.6 and s.avg_logprob < -0.7:
            drop["no_speech"] = drop.get("no_speech", 0) + 1
            continue
        if getattr(s, "compression_ratio", 1.0) > 2.4:
            drop["repetitive"] = drop.get("repetitive", 0) + 1
            continue
        if len(words) >= 4 and sum(w in hw for w in words) / len(words) > 0.6:
            drop["hotword_echo"] = drop.get("hotword_echo", 0) + 1
            continue  # echo of the vocabulary hint
        out.append({
            "start": round(s.start + offset, 2), "end": round(s.end + offset, 2), "text": text,
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "words": [{"word": w.word.strip(), "start": round(w.start + offset, 2), "end": round(w.end + offset, 2),
                       "p": round(w.probability, 3)} for w in (s.words or [])],
        })
    return out


class ASR(_Lazy):
    name = "asr"
    live = True  # the live model favours speed: one decoding pass, no temperature fallback
    last_stats: dict = {}
    license = "MIT (faster-whisper, Whisper weights)"

    def _load(self):  # pragma: no cover
        from faster_whisper import WhisperModel
        return WhisperModel(self.model_id, device=settings.asr_device, compute_type=settings.asr_compute)

    def _language(self, m, audio: np.ndarray) -> str:
        """Fixed language, or auto-detect restricted to the allowed set (avoids 'jw' ... mislabels)."""
        cfg = settings.asr_language
        if cfg != "auto":
            return cfg
        allowed = [x for x in settings.asr_languages if x]
        try:  # pragma: no cover - needs model
            _, _, probs = m.detect_language(audio)
            probs = dict(probs)
            lang = max(allowed, key=lambda l: probs.get(l, 0.0))
        except Exception:
            lang = allowed[0] if allowed else "hi"
        return "hi" if (lang == "ur" and settings.asr_urdu_as_hindi) else lang

    @staticmethod
    def use_indic(lang: str | None) -> bool:
        from .indic_asr import LANGUAGES
        return bool(lang) and lang in LANGUAGES and (lang != "hi" or settings.indic_asr_for_hindi) \
            and indic.get(block=True) is not None

    def transcribe(self, audio: np.ndarray, offset: float = 0.0, language: str | None = None,
                   previous_text: str = "") -> tuple[list[dict], str | None]:
        """Returns segments with Roman-script `text` and the original in `text_native`."""
        from .translit import to_roman
        m = self.get(block=True)
        if m is None:
            raise RuntimeError(f"{self.name} model not available ({self._error})")
        lang = language or self._language(m, audio)
        if not language and lang != "en" and indic.get(block=True) is not None:
            # Whisper hears that speech is Indian but often not which language; IndicConformer decides among its
            # 22 languages (see indic_asr.route_language; measured in eval/run_lid_eval.py)
            from .indic_asr import route_language
            lang = route_language(lang, indic.get(block=True).identify(audio))
        if self.use_indic(lang):
            segs = indic.get(block=True).transcribe(audio, lang, offset)
            self.last_stats = {"engine": "indicconformer", "lang": lang, "kept": len(segs)}
            for sg in segs:
                sg["text_native"] = sg["text"]
                sg["text"] = to_roman(sg["text"])  # Devanagari-script languages get Roman text; others stay native
            return segs, lang
        # Fallback re-decoding at higher temperatures made one live chunk take ~50 s when Whisper looped on
        # Hindi; for live audio use a single pass and stop loops with no-repeat n-grams instead.
        kwargs = dict(language=lang, task="transcribe", vad_filter=True, word_timestamps=not self.live,
                      beam_size=3 if self.live else 5, condition_on_previous_text=False,
                      temperature=0.0 if self.live else (0.0, 0.2, 0.4), no_repeat_ngram_size=3,
                      repetition_penalty=1.1, compression_ratio_threshold=2.4,
                      vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200})
        if previous_text:
            kwargs["initial_prompt"] = previous_text[-200:]
        try:  # hotwords bias recognition of domain words without the echo problems of a prompt
            segs, info = m.transcribe(audio, hotwords=HOTWORDS, **kwargs)
        except TypeError:  # pragma: no cover - older faster-whisper
            segs, info = m.transcribe(audio, **kwargs)
        raw = list(segs)
        stats: dict = {"raw": len(raw), "lang": getattr(info, "language", lang),
                       "speech_s": round(float(getattr(info, "duration_after_vad", 0) or 0), 1)}
        out = _clean_segments(raw, offset, HOTWORDS, stats)
        prev = previous_text.strip().lower()
        before = len(out)
        # prompt echo: only whole repeats of an earlier line (short real replies like "haan ji" are kept)
        out = [s for s in out if not prev or len(s["text"]) < 12 or s["text"].strip().lower() not in prev]
        if before - len(out):
            stats.setdefault("dropped", {})["prompt_echo"] = before - len(out)
        stats["kept"] = len(out)
        self.last_stats = stats
        for s in out:
            s["text_native"] = s["text"]
            s["text"] = to_roman(s["text"])
        return out, settings.asr_report_language or getattr(info, "language", lang)


class FinalASR(ASR):
    name = "asr_final"
    live = False


EMO_IGNORE = {"other", "<unk>", "unk", "unknown"}


class Emotion(_Lazy):
    name = "voice_emotion"
    license = "Check the emotion2vec+ model card (model-license) before commercial use"

    def _load(self):  # pragma: no cover
        from funasr import AutoModel
        hub = "hf" if self.model_id.startswith("emotion2vec/") else "ms"
        return AutoModel(model=self.model_id, hub=hub, disable_update=True, disable_pbar=True)

    def head(self):
        """The telephone-adapted head (see speech/ser_head.py), or None when it is off or not built."""
        if settings.ser_head.lower() in ("off", "0", "false", "none"):
            return None
        from . import ser_head
        return ser_head.load(None if settings.ser_head.lower() in ("auto", "on", "1", "true") else settings.ser_head)

    def predict_full(self, audio: np.ndarray, block: bool = False) -> tuple[dict[str, float] | None, np.ndarray | None]:
        """Raw emotion2vec+ distribution (renormalised over real emotions) and its 768-d utterance embedding."""
        m = self.get(block=block)  # live calls never stall waiting for a download
        if m is None:
            return None, None
        res = m.generate(audio.astype(np.float32), granularity="utterance", extract_embedding=True, disable_pbar=True)
        r = res[0] if isinstance(res, list) else res
        labels = [str(lbl).split("/")[-1].lower() for lbl in r["labels"]]
        scores = {lbl: float(s) for lbl, s in zip(labels, r["scores"]) if lbl not in EMO_IGNORE}
        total = sum(scores.values())
        feats = np.asarray(r["feats"], dtype=np.float32).reshape(-1) if r.get("feats") is not None else None
        if total <= 0:
            return None, feats
        return {lbl: round(v / total, 4) for lbl, v in scores.items()}, feats

    def predict(self, audio: np.ndarray) -> dict[str, float] | None:
        raw, feats = self.predict_full(audio)
        h = self.head()
        if h is not None and feats is not None:
            adapted = h.predict(feats)
            if adapted:
                return adapted
        return raw

    def status(self) -> ModelStatus:
        st = super().status()
        h = self.head()
        if h is not None and st.loaded:
            st.model = f"{st.model} + {h.spec.get('name', 'telephone head')}"
        return st


class Speaker(_Lazy):
    name = "speaker_embedding"
    license = "Apache-2.0 (SpeechBrain ECAPA-TDNN, VoxCeleb)"

    def _load(self):  # pragma: no cover
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):  # removed in torchaudio 2.9; SpeechBrain 1.0.x still calls it
            torchaudio.list_audio_backends = lambda: ["soundfile"]
        kw = {}
        try:  # Windows without Developer Mode cannot create symlinks: copy the files instead
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except ImportError:  # older speechbrain
            pass
        return EncoderClassifier.from_hparams(source=self.model_id,
                                              savedir=str(settings.data_dir / "models" / "ecapa"), **kw)

    def embed(self, audio: np.ndarray) -> list[float] | None:
        m = self.get(block=True)
        if m is None:
            return None
        import torch
        with torch.no_grad():
            e = m.encode_batch(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)).squeeze().cpu().numpy()
        e = e / (np.linalg.norm(e) + 1e-9)
        return [round(float(v), 6) for v in e]


FAKE_WORDS = ("fake", "spoof", "synth", "deepfake", "generated", "tts", "clone")


class SpoofDetector(_Lazy):
    """Synthetic / cloned-voice detector (audio classification model, labels like real vs fake)."""
    name = "voice_authenticity"
    license = "Check the model card of ANVESH_SPOOF_MODEL before deployment"

    def _load(self):  # pragma: no cover - needs model files
        from transformers import pipeline
        return pipeline("audio-classification", model=self.model_id, device=-1)

    def p_fake(self, audio: np.ndarray, sr: int = 16000) -> float | None:
        """Probability the speech is machine-generated, averaged over up to three 4 s windows."""
        m = self.get(block=False)
        if m is None:
            return None
        n = 4 * sr
        starts = [0] if len(audio) <= n else list(np.linspace(0, len(audio) - n, num=min(3, len(audio) // n + 1)).astype(int))
        scores = []
        for st in starts:
            res = m({"raw": audio[st:st + n].astype(np.float32), "sampling_rate": sr}, top_k=None)
            fake = sum(r["score"] for r in res if any(w in str(r["label"]).lower() for w in FAKE_WORDS))
            real = sum(r["score"] for r in res if not any(w in str(r["label"]).lower() for w in FAKE_WORDS))
            if fake + real > 0:
                scores.append(fake / (fake + real))
        return float(np.mean(scores)) if scores else None


class IndicASR(_Lazy):
    """AI4Bharat IndicConformer for 22 Indian languages (see speech/indic_asr.py)."""
    name = "asr_indic"
    license = "MIT (AI4Bharat IndicConformer)"

    def _load(self):  # pragma: no cover - needs model files
        from pathlib import Path

        from .indic_asr import IndicConformer
        if not (Path(self.model_id) / "assets" / "encoder.onnx").exists():
            # gated on Hugging Face: accept the model's terms and sign in (huggingface-cli login) first
            from huggingface_hub import snapshot_download
            snapshot_download("ai4bharat/indic-conformer-600m-multilingual", local_dir=self.model_id,
                              allow_patterns=["assets/*", "config.json", "README.md"])
        return IndicConformer(self.model_id)


def _indic_dir() -> str:
    from .indic_asr import default_dir
    return settings.indic_asr_dir or str(default_dir())


asr = ASR(settings.asr_model, settings.enable_asr)
asr_final = FinalASR(settings.asr_final_model, settings.enable_asr and settings.enable_asr_final
                     and settings.asr_final_model != settings.asr_model)
emotion = Emotion(settings.emotion_model, settings.enable_emotion)
speaker = Speaker("speechbrain/spkrec-ecapa-voxceleb", settings.enable_speaker)
spoof = SpoofDetector(settings.spoof_model, settings.enable_spoof)
indic = IndicASR(_indic_dir(), settings.enable_indic_asr and settings.enable_asr)


def warmup() -> None:
    """Start loading every enabled model in the background (downloads on first run)."""
    for m in (asr, emotion, speaker, asr_final, spoof, indic):
        m.start_loading()


def best_asr() -> ASR:
    """The accurate model when it is ready, otherwise the live model."""
    return asr_final if asr_final.state == "ready" else asr


def statuses() -> list[dict]:
    return [s.status().__dict__ for s in (asr, asr_final, indic, emotion, speaker, spoof)]
