"""Live speech-emotion tracking for calls.

Two signals, both indicative only:

1. Model emotion (emotion2vec+) on overlapping ~3 s voiced windows, smoothed over time so one noisy
   window cannot flip the reading, and reported as "uncertain" when the model is not confident.
2. Vocal arousal relative to THIS caller's own baseline (loudness, pitch movement, speaking rate),
   which needs no model, works from the first seconds of a call, and is robust to accent and language.

They are combined into the four signals the problem statement asks for (fear, anger, distress,
nervousness). Nothing here is evidence of intent, honesty or guilt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .speech import audio as A

SR = A.SR
WINDOW_S = 3.0
HOP_S = 1.5
MIN_VOICED = 0.35
UNCERTAIN_P = 0.45
EMA = 0.4  # weight of the newest window in the smoothed distribution


def _z(v: float, mu: float, sd: float) -> float:
    return 0.0 if sd <= 1e-6 else max(-3.0, min(3.0, (v - mu) / sd))


@dataclass
class Baseline:
    """Running estimate of the caller's normal voice (first voiced windows weigh most)."""
    rms: deque = field(default_factory=lambda: deque(maxlen=12))
    pitch: deque = field(default_factory=lambda: deque(maxlen=12))
    f0: deque = field(default_factory=lambda: deque(maxlen=12))
    rate: deque = field(default_factory=lambda: deque(maxlen=12))

    def ready(self) -> bool:
        return len(self.rms) >= 2

    def add(self, f: dict) -> None:
        self.rms.append(f.get("rms", 0.0))
        self.pitch.append(f.get("pitch_std", 0.0))
        self.f0.append(f.get("pitch_mean", 0.0))
        self.rate.append(f.get("speech_rate", 0.0))

    def arousal(self, f: dict) -> float:
        def stat(q, floor):
            a = np.array(q, dtype=float)
            return float(np.median(a)), max(float(np.std(a)), floor)
        (mr, sr_), (mp, sp), (mf, sf), (mt, st) = (stat(self.rms, 0.01), stat(self.pitch, 8.0),
                                                   stat(self.f0, 12.0), stat(self.rate, 0.4))
        # arousal cues: louder, higher and more varied pitch, faster speech (only increases count for pitch range)
        z = (0.40 * _z(f.get("rms", 0), mr, sr_) + 0.25 * _z(f.get("pitch_mean", 0), mf, sf)
             + 0.20 * max(0.0, _z(f.get("pitch_std", 0), mp, sp)) + 0.15 * _z(f.get("speech_rate", 0), mt, st))
        return round(float(1 / (1 + np.exp(-1.6 * z))), 3)  # 0..1, 0.5 = the caller's usual voice


POSITIVE = ("happy", "surprised")


def adjust_for_role(emotions: dict[str, float] | None, role: str) -> dict[str, float] | None:
    """Common-sense prior: someone calling a cybercrime helpline to report harm is not 'happy'.

    For victims and informants, emotion2vec's 'happy' on a tense phone line is almost always nervous
    laughter, forced politeness or a telephone-band artefact, and 'surprised' is shock. So 'happy' is never
    reported for these roles: 40% of its mass is read as nervous arousal (added to 'fearful'), the rest is
    dropped; 'surprised' is shared between 'fearful' and 'neutral'. Suspect recordings are left untouched.
    """
    if not emotions or role not in ("victim", "informant"):
        return emotions
    e = {k: v for k, v in emotions.items() if k not in POSITIVE}
    e["fearful"] = e.get("fearful", 0.0) + 0.4 * emotions.get("happy", 0.0) + 0.5 * emotions.get("surprised", 0.0)
    e["neutral"] = e.get("neutral", 0.0) + 0.5 * emotions.get("surprised", 0.0)
    total = sum(e.values())
    return {k: round(v / total, 4) for k, v in e.items()} if total > 0 else None


def indicators(emotions: dict[str, float] | None, arousal: float | None) -> dict[str, float]:
    """Map model classes + arousal to fear / anger / distress / nervousness (0..1, indicative)."""
    e = emotions or {}
    fear, anger = e.get("fearful", 0.0), e.get("angry", 0.0) + 0.5 * e.get("disgusted", 0.0)
    sad, neutral = e.get("sad", 0.0), e.get("neutral", 0.0)
    a = 0.5 if arousal is None else arousal
    out = {
        "fear": fear,
        "anger": anger,
        "distress": min(1.0, 0.6 * (fear + sad) + 0.4 * max(0.0, a - 0.5) * 2 * (1 - neutral)),
        "nervousness": min(1.0, (0.5 * fear + 0.5 * e.get("surprised", 0.0) + max(0.0, a - 0.55) * 1.6) * (1 - anger)),
    }
    if not emotions:  # model not loaded: arousal alone can only say "raised voice", not which emotion
        out = {"fear": 0.0, "anger": 0.0, "distress": round(max(0.0, a - 0.5) * 1.4, 3),
               "nervousness": round(max(0.0, a - 0.55) * 1.6, 3)}
    return {k: round(min(1.0, v), 3) for k, v in out.items()}


class LiveEmotionTracker:
    def __init__(self, predict=None, role: str = "victim"):
        self.predict = predict  # callable(np.ndarray) -> dict | None
        self.role = role
        self.baseline = Baseline()
        self.smoothed: dict[str, float] | None = None
        self.history: list[dict] = []

    def analyse(self, x: np.ndarray, t_start: float) -> dict:
        """One window (float32 mono 16 kHz). Returns a window record for the case and the live UI."""
        t_end = t_start + len(x) / SR
        f = A.features(x) or {}
        voiced = 1 - f.get("pause_ratio", 1.0) if f else 0.0
        silent = not f or f.get("rms", 0.0) < 0.006 or voiced < MIN_VOICED
        rec = {"t_start": round(t_start, 2), "t_end": round(t_end, 2), "acoustic": f, "silent": silent,
               "emotions": None, "label": None, "p": None, "arousal": None, "indicators": None}
        if silent:
            return rec
        arousal = self.baseline.arousal(f) if self.baseline.ready() else None
        if len(self.baseline.rms) < 4 or (arousal is not None and arousal < 0.7):
            self.baseline.add(f)  # learn the usual voice; do not learn from very agitated windows
        emo = None
        if self.predict:
            try:
                emo = adjust_for_role(self.predict(x), self.role)
            except Exception:  # pragma: no cover - model failure must never break the call
                emo = None
        if emo:
            prev = self.smoothed or emo
            keys = set(prev) | set(emo)
            self.smoothed = {k: EMA * emo.get(k, 0.0) + (1 - EMA) * prev.get(k, 0.0) for k in keys}
            label, p = max(self.smoothed.items(), key=lambda kv: kv[1])
            rec["label"] = label if p >= UNCERTAIN_P else "uncertain"
            rec["p"] = round(p, 3)
        rec.update({"emotions": emo, "arousal": arousal,
                    "indicators": indicators(self.smoothed if emo else None, arousal)})
        self.history.append(rec)
        return rec


class LatencyStats:
    """Rolling live-latency figures (process-wide), reported by /api/health and shown in the live view."""

    def __init__(self, n: int = 500):
        self.data: dict[str, deque] = {"emotion_ms": deque(maxlen=n), "transcript_s": deque(maxlen=n)}

    def add(self, key: str, value: float) -> None:
        self.data[key].append(float(value))

    def summary(self) -> dict:
        out = {}
        for k, q in self.data.items():
            if q:
                a = np.sort(np.array(q))
                out[k] = {"n": len(a), "median": round(float(np.median(a)), 2),
                          "p95": round(float(a[int(0.95 * (len(a) - 1))]), 2)}
        return out


LATENCY = LatencyStats()


class Chunker:
    """Decides where to cut live audio for transcription: at a pause, using an adaptive noise floor."""

    def __init__(self, target_s: float = 5.0, min_s: float = 1.6, max_s: float = 6.5):
        self.target, self.min, self.max = target_s, min_s, max_s
        self.floor = deque(maxlen=200)  # recent 100 ms frame energies

    def observe(self, pcm16: bytes) -> None:
        x = A.pcm16_to_float(pcm16)
        n = SR // 10
        for i in range(0, len(x) - n + 1, n):
            self.floor.append(float(np.sqrt(np.mean(x[i:i + n] ** 2))))

    def threshold(self) -> float:
        if len(self.floor) < 10:
            return 0.012
        return float(min(0.03, max(0.006, np.percentile(self.floor, 20) * 2.2)))

    def should_cut(self, buf_seconds: float, tail_pcm16: bytes) -> bool:
        if buf_seconds >= self.max:
            return True
        if buf_seconds < self.min:
            return False
        tail = A.pcm16_to_float(tail_pcm16)
        quiet = len(tail) > 0 and float(np.sqrt(np.mean(tail ** 2))) < self.threshold()
        return quiet and (buf_seconds >= self.target * 0.5)
